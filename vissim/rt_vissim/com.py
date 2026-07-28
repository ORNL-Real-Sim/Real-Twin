##############################################################################
# Copyright (c) 2024-, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of RealTwin and is distributed under a GPL               #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# Contributors: ORNL Real-Twin Team                                          #
# Contact: realtwin@ornl.gov                                                 #
##############################################################################
"""Thin wrapper around the PTV Vissim COM API.

This is the only module in the package that talks to Vissim.  Everything else
works on plain dataframes and dataclasses so that the bulk of the pipeline can
be developed and tested on a machine without a Vissim licence.

Typical use::

    from rt_vissim.com import VissimSession

    with VissimSession(visible=True) as sess:
        sess.new_net()
        sess.import_opendrive("chatt.xodr")
        sess.save_net_as("chatt.inpx")

Notes:
    Vissim exposes several COM ProgIDs -- an unversioned ``VISSIM.Vissim`` that
    points at whichever release registered itself last, plus versioned ones such
    as ``VISSIM.Vissim-64.2600`` (Vissim 2026, 64 bit).  Pinning the version is
    strongly preferred when more than one release is installed, which is the
    normal state of affairs on a modelling workstation.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Preferred ProgIDs, most specific first.  ``VissimSession`` walks this list
#: when no explicit ProgID is given.  Vissim 2026 is what the RealTwin VISSIM
#: pipeline is developed against.
DEFAULT_PROGIDS = (
    "VISSIM.Vissim-64.2600",   # Vissim 2026, 64 bit
    "VISSIM.Vissim.2026",
    "VISSIM.Vissim-64.2200",   # Vissim 2022, 64 bit
    "VISSIM.Vissim.2022",
    "VISSIM.Vissim",           # whatever registered last
)


class VissimComError(RuntimeError):
    """Raised when a Vissim COM call fails or Vissim is unavailable."""


def available_progids() -> list[str]:
    """Return the Vissim COM ProgIDs registered on this machine.

    Reads ``HKEY_CLASSES_ROOT`` directly rather than trying to instantiate each
    one, so it is cheap and does not consume a licence.

    Returns:
        Registered ProgIDs, e.g. ``["VISSIM.Vissim-64.2600", ...]``.  Empty on
        non-Windows platforms or when Vissim is not installed.
    """
    try:
        import winreg
    except ImportError:  # pragma: no cover - non-Windows
        return []

    found: list[str] = []
    try:
        with winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, "") as root:
            i = 0
            while True:
                try:
                    name = winreg.EnumKey(root, i)
                except OSError:
                    break
                if name.upper().startswith("VISSIM.VISSIM"):
                    found.append(name)
                i += 1
    except OSError:  # pragma: no cover - unusual registry state
        return []
    return sorted(found)


class VissimSession:
    """A live Vissim COM session, usable as a context manager.

    Args:
        progid: COM ProgID to instantiate.  When ``None``, the first entry of
            :data:`DEFAULT_PROGIDS` that instantiates successfully is used.
        visible: Whether to show the Vissim GUI.  Keeping it visible is handy
            while developing; batch runs should pass ``False``.
        quit_on_exit: Whether to shut Vissim down when leaving the ``with``
            block.  Set ``False`` to leave the GUI open for inspection.

    Attributes:
        vissim: The raw COM object.  Reach for it when this wrapper does not
            expose what you need -- it is a normal Vissim ``IVissim``.
        progid: The ProgID that was actually instantiated.
    """

    def __init__(self, progid: str | None = None, *, visible: bool = True,
                 quit_on_exit: bool = True):
        self.progid = progid
        self.visible = visible
        self.quit_on_exit = quit_on_exit
        self.vissim = None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def start(self) -> "VissimSession":
        """Launch Vissim and return ``self``.

        Raises:
            VissimComError: If ``pywin32`` is missing or no ProgID could be
                instantiated (Vissim not installed, or no licence available).
        """
        try:
            import win32com.client
        except ImportError as exc:  # pragma: no cover - environment specific
            raise VissimComError(
                "pywin32 is required to drive Vissim over COM. "
                "Install it with: pip install pywin32"
            ) from exc

        candidates = [self.progid] if self.progid else list(DEFAULT_PROGIDS)
        errors: list[str] = []
        for progid in candidates:
            try:
                self.vissim = win32com.client.Dispatch(progid)
                self.progid = progid
                break
            except Exception as exc:  # noqa: BLE001 - COM raises many types
                errors.append(f"{progid}: {exc}")
        else:
            detail = "\n    ".join(errors) or "no ProgIDs tried"
            raise VissimComError(
                "Could not start Vissim over COM. Tried:\n    " + detail
                + "\n  Registered ProgIDs on this machine: "
                + (", ".join(available_progids()) or "<none>")
            )

        try:
            self.vissim.Visible = self.visible
        except Exception:  # noqa: BLE001 - not fatal, some builds disallow it
            pass
        return self

    def close(self) -> None:
        """Shut down the Vissim instance, ignoring teardown errors."""
        if self.vissim is not None and self.quit_on_exit:
            try:
                self.vissim.Exit()
            except Exception:  # noqa: BLE001 - teardown is best effort
                pass
        self.vissim = None

    def __enter__(self) -> "VissimSession":
        return self.start()

    def __exit__(self, *exc_info) -> None:
        self.close()

    @property
    def net(self):
        """The Vissim ``Net`` object.

        Raises:
            VissimComError: If the session has not been started.
        """
        self._require_started()
        return self.vissim.Net

    def _require_started(self) -> None:
        if self.vissim is None:
            raise VissimComError("Vissim session is not started; call start() "
                                 "or use the session as a context manager.")

    # ------------------------------------------------------------------ #
    # Network I/O
    # ------------------------------------------------------------------ #
    def new_net(self) -> None:
        """Discard the current network and start an empty one."""
        self._require_started()
        self.vissim.New()

    def load_net(self, path: str | Path, additive: bool = False) -> None:
        """Load an ``.inpx`` network.

        Args:
            path: Path to the ``.inpx`` file.
            additive: Whether to merge into the current network instead of
                replacing it.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
        """
        self._require_started()
        path = Path(path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Vissim network not found: {path}")
        self.vissim.LoadNet(str(path), additive)

    def save_net_as(self, path: str | Path) -> Path:
        """Save the current network to ``path`` and return the resolved path."""
        self._require_started()
        path = Path(path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.vissim.SaveNetAs(str(path))
        return path

    def import_opendrive(self, path: str | Path, **options) -> None:
        """Import an OpenDRIVE (``.xodr``) file into the current network.

        The OpenDRIVE importer is exposed under different names depending on the
        Vissim release, so this tries the known entry points in turn and reports
        all of them if none works.  On success Vissim builds links for the
        OpenDRIVE roads and connectors for the junction paths -- **renumbering
        everything**, which is why :mod:`rt_vissim.network` derives the RealTwin
        matchup table from the imported network rather than from the OpenDRIVE
        IDs.

        Args:
            path: Path to the ``.xodr`` file.
            **options: Reserved for importer options in a future release; passing
                any is currently an error so that typos surface immediately.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            TypeError: If unsupported ``options`` are passed.
            VissimComError: If no known import entry point succeeded.
        """
        self._require_started()
        if options:
            raise TypeError(f"Unsupported import_opendrive options: {sorted(options)}")

        path = Path(path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"OpenDRIVE file not found: {path}")
        target = str(path)

        attempts: list[tuple[str, str]] = []

        # Entry points seen across Vissim releases, most likely first.
        for owner_name, owner in (("Net", self.vissim.Net), ("Vissim", self.vissim)):
            for method in ("ImportOpenDrive", "ImportOpenDRIVE", "OpenDriveImport"):
                fn = getattr(owner, method, None)
                if fn is None:
                    continue
                try:
                    fn(target)
                    return
                except Exception as exc:  # noqa: BLE001 - COM raises many types
                    attempts.append((f"{owner_name}.{method}", str(exc)))

        detail = "\n    ".join(f"{name}: {err}" for name, err in attempts)
        raise VissimComError(
            "Could not import OpenDRIVE via COM.\n"
            f"  File: {target}\n"
            "  Tried:\n    " + (detail or "<no matching COM method found>")
            + "\n  Fall back to the GUI: File > Import > OpenDRIVE, then save the "
              ".inpx and use load_net()."
        )

    # ------------------------------------------------------------------ #
    # Simulation
    # ------------------------------------------------------------------ #
    def configure_simulation(self, *, start_time: float, end_time: float,
                             resolution: int = 10, seed: int = 42) -> None:
        """Set the simulation window and stochastics.

        Args:
            start_time: Simulation start, in seconds after midnight.  Written to
                ``StartTm`` so the clock lines up with the GridSmart and Synchro
                time-of-day data.
            end_time: Simulation end, in seconds after midnight.
            resolution: Simulation steps per second.
            seed: Random seed.
        """
        self._require_started()
        sim = self.vissim.Simulation
        sim.SetAttValue("SimPeriod", int(round(end_time - start_time)))
        sim.SetAttValue("StartTm", int(round(start_time)))
        sim.SetAttValue("SimRes", int(resolution))
        sim.SetAttValue("RandSeed", int(seed))

    def run_simulation(self) -> None:
        """Run the loaded scenario to completion."""
        self._require_started()
        self.vissim.Simulation.RunContinuous()


def com_objects_to_records(collection, attributes: list[str]) -> list[dict]:
    """Read a Vissim COM collection into plain dicts.

    ``GetMultipleAttributes`` is dramatically faster than looping over items and
    calling ``AttValue`` one at a time -- the difference is minutes on a network
    the size of Chattanooga -- so use it when available and fall back only when
    the collection does not support it.

    Args:
        collection: Any Vissim COM collection, e.g. ``Vissim.Net.Links``.
        attributes: Attribute names to read, e.g. ``["No", "Name", "IsConn"]``.

    Returns:
        One dict per object, keyed by attribute name.
    """
    try:
        rows = collection.GetMultipleAttributes(attributes)
        return [dict(zip(attributes, row)) for row in rows]
    except Exception:  # noqa: BLE001 - fall back to the slow path
        out = []
        for obj in collection:
            out.append({attr: obj.AttValue(attr) for attr in attributes})
        return out


def is_windows() -> bool:
    """Return whether we are on Windows, where the Vissim COM API exists."""
    return os.name == "nt"
