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

"""Test the find_executable_from_PATH_on_win function"""

import os
import stat
from pathlib import Path

import pytest

from tests.proj_add_sys_path import add_pkg_to_sys_path

add_pkg_to_sys_path("realtwin")

from realtwin.util_lib.find_executable_from_PATH_on_win import (  # noqa: E402
    find_executable_from_PATH_on_win,
)


class TestFindExeOnWin:
    """Test the find_executable_from_PATH_on_win function"""

    def setup_class(self):
        """Set up the class"""
        self.EXE_NAME = "aconsole.exe"
        self.EXE_NAME_NOT_FOUND = "fake_exe"

    def test_error_exe_name_error(self):
        """Test the error when exe_name is not a string"""
        with pytest.raises(ValueError, match="exe_name should be a string."):
            find_executable_from_PATH_on_win(exe_name=123)

    def test_ext_error(self):
        """Test the error when ext is not a string"""
        with pytest.raises(ValueError, match="ext should be a string."):
            find_executable_from_PATH_on_win(exe_name=self.EXE_NAME, ext=123)

    def test_sel_dir_error(self):
        """Test the error when sel_dir is not a list"""
        with pytest.raises(ValueError, match="sel_dir should be a list."):
            find_executable_from_PATH_on_win(exe_name=self.EXE_NAME, sel_dir=123)

    def test_exe_not_found(self, monkeypatch, tmp_path):
        """Test the case when the exe is not found"""
        empty_path = tmp_path / "empty-path"
        empty_drive = tmp_path / "empty-drive"
        empty_path.mkdir()
        empty_drive.mkdir()
        monkeypatch.setenv("PATH", str(empty_path))
        monkeypatch.setenv("SystemDrive", str(empty_drive))

        res = find_executable_from_PATH_on_win(
            exe_name=self.EXE_NAME_NOT_FOUND,
            verbose=False,
        )
        assert res is None

    def test_exe_found_in_path(self, monkeypatch, tmp_path):
        """Test finding an executable in PATH without a drive search."""
        path_dir = tmp_path / "path-bin"
        path_dir.mkdir()
        executable = path_dir / self.EXE_NAME
        executable.touch()
        executable.chmod(executable.stat().st_mode | stat.S_IEXEC)
        monkeypatch.setenv("PATH", str(path_dir))

        res = find_executable_from_PATH_on_win(
            exe_name=self.EXE_NAME,
            verbose=False,
        )

        assert res is not None
        assert Path(res[0]).resolve() == executable.resolve()

    @pytest.mark.skipif(os.name != "nt", reason="Windows system-drive search")
    def test_exe_found_on_system_drive_without_sel_dir(self, monkeypatch, tmp_path):
        """Find an executable recursively when it is not present in PATH."""
        empty_path = tmp_path / "empty-path"
        fake_drive = tmp_path / "fake-drive"
        executable = (
            fake_drive / "Program Files" / "Aimsun" / "Aimsun Next 23" / "AConsole.EXE"
        )
        empty_path.mkdir()
        executable.parent.mkdir(parents=True)
        executable.touch()
        executable.chmod(executable.stat().st_mode | stat.S_IEXEC)
        monkeypatch.setenv("PATH", str(empty_path))
        monkeypatch.setenv("SystemDrive", str(fake_drive))

        res = find_executable_from_PATH_on_win(
            exe_name="aconsole",
            verbose=False,
        )

        assert res is not None
        assert Path(res[0]).resolve() == executable.resolve()
