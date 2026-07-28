# RealTwin VISSIM pipeline (development sandbox)

A PTV Vissim counterpart to the RealTwin SUMO pipeline, driven over the Vissim
COM API. Same steps and same method as SUMO — network ingestion, a MatchupTable
join, demand ingestion, signal control ingestion — but **standalone after the
OpenDRIVE import**, because Vissim renumbers the whole network on import.

Developed against **PTV Vissim 2026** (COM ProgID `VISSIM.Vissim-64.2600`).

---

## Why standalone, and not chained off the SUMO IDs

The SUMO MatchupTable's `FromRoadID_OpenDrive` / `ToRoadID_OpenDrive` columns are
**not** durable OpenDRIVE IDs. They are SUMO edge IDs with the leading `-`
stripped: RealTwin renames every SUMO edge to `-<n>` in
[`parse_SUMO_ID`](../realtwin/util_lib/mapping_SUMO_OpenDrive_ID.py), and the
OpenDRIVE road IDs that `netconvert` emits are a separate running counter.

Measured on the Chattanooga dataset: regenerating the `.xodr` with SUMO 1.24
instead of the 1.21 that produced the committed file shifted every road ID
(`280…331` became `390…575`) — **zero** overlap with the MatchupTable. Vissim
then renumbers again on import. So the pipeline re-derives its own junctions,
bearings and turns from the Vissim network itself.

---

## Pipeline stages

| SUMO (`realtwin`) | VISSIM (`rt_vissim`) |
|---|---|
| `parse_SUMO_to_OpenDrive` → `netconvert` | `scripts/01_import_opendrive.py` → `netconvert` + COM import |
| parse `net.xml` junctions/edges | [`network.py`](rt_vissim/network.py) reads `Vissim.Net.Links` |
| `format_junction_bearing` (bearings from lane shape) | bearings from Vissim link polylines |
| junction = `<junction>` element | junction = SUMO internal edge name in the Vissim link name |
| turn from `connection dir` attribute | turn classified from bearing change |
| `generate_matchup_table` → `MatchupTable.xlsx` | same layout, `*_Vissim` link-number columns |
| GridSmart → `.flow.xml` / `.turn.xml` → `jtrrouter` | GridSmart → vehicle inputs + static routing decisions |
| Synchro UTDF → NEMA `tlLogic` | Synchro UTDF → `.prbc` Ring Barrier Controller files |

---

## Two findings that make the import work

### 1. Georeferencing — the network lands at the origin

SUMO normalises coordinates to near `(0, 0)` and records the shift in
`<location netOffset="-667733.00,-3878704.47">`. `netconvert` faithfully writes
local coordinates to the `.xodr` and parks the true origin in
`<header><offset x="667733.00" y="3878704.47"/>` — **which Vissim's OpenDRIVE
importer ignores**, dropping the network next to `(0, 0)` instead of at the site.

`--offset.disable-normalization` does not help (the net is already normalised).
The fix is to shift the network back before export:

```bash
netconvert -s chatt.net.xml --opendrive-output chatt.xodr \
  --output.original-names true --junctions.scurve-stretch 1.0 \
  --offset.x 667733.00 --offset.y 3878704.47
```

`scripts/01_import_opendrive.py` reads `netOffset` from the network and applies
the negated values automatically.

### 2. Topology — a turn is not a connector

Vissim's OpenDRIVE importer turns **every** OpenDRIVE road into a link, including
the connecting roads inside junctions. Vissim *connectors* are only ~1.5 m
stitches between consecutive roads. So one movement is a path:

```
approach link --conn--> internal link(s) --conn--> exit link
```

Treating each connector as a movement classifies almost everything as "thru"
(measured: 241 thru / 8 left / 1 right — obviously wrong).

Vissim also names each link after the OpenDRIVE road it came from:

| Link name | Meaning |
|---|---|
| `390-0-Right` | OpenDRIVE road 390, lane section 0, right side |
| `473: :12_0-0-Right` | road 473, OpenDRIVE road name `:12_0` |

`:12_0` is a SUMO **internal edge** — a path inside SUMO junction 12 — carried
across by `netconvert --output.original-names`. That yields an exact junction
grouping straight from the Vissim model, with no geometric clustering.
Networks imported without original names fall back to spatial clustering.

---

## Validation against the SUMO pipeline

Running stage 1 on `datasets/chattanooga/updated_net/chatt.net.xml` and comparing
the derived movement table with RealTwin's hand-curated
`datasets/chattanooga/updated_net/MatchupTable.xlsx`:

| | VISSIM-derived | SUMO MatchupTable |
|---|---|---|
| Movements | 104 | 104 |
| Junctions | 10 | 10 |
| Junction IDs | `2,3,4,7,8,9,10,11,12,18` | `2,3,4,7,8,9,10,11,12,18` |
| Movements per junction | `4,5,7,8,8,8,16,16,16,16` | `4,5,7,8,8,8,16,16,16,16` |
| Legs per junction | `3,3,3,3,3,3,4,4,4,4` | `3,3,3,3,3,3,4,4,4,4` |
| Turn mix | R28 T28 L26 U22 | R28 T30 L24 U22 |

The structure matches exactly. Two movements are labelled `left` where SUMO says
`thru` — skewed approaches near the 20° classification threshold. As with the
SUMO flow, the MatchupTable is the place to correct them by hand.

---

## Layout

```
vissim/
  rt_vissim/
    com.py          COM session, OpenDRIVE import, collection reads
    network.py      links -> junctions, bearings, turn movements
    ir.py           simulator-agnostic scenario IR (vehicle inputs, routes, signals)
    matchup.py      generate / read the Vissim MatchupTable      [in progress]
    demand.py       GridSmart turn counts -> IR                  [todo]
    signal.py       Synchro UTDF -> IR                           [todo]
    rbc.py          IR -> .prbc Ring Barrier Controller files    [todo]
    writer.py       IR -> Vissim, over COM                       [todo]
    pipeline.py     orchestrator                                 [todo]
  scripts/
    01_import_opendrive.py    stage 1: netconvert + import + inspect
  tests/                                                          [todo]
  work/                       generated artefacts (gitignored)
  VISSIM_previous/            prior ORNL VISSIM work, kept for reference
```

`rt_vissim` is deliberately split so that only `com.py` and `writer.py` need
Vissim. Everything else is plain pandas/JSON and is unit-testable without a
licence.

---

## Signal control: `.prbc` Ring Barrier Controller

`VISSIM_previous/` contains working `.prbc` files, which settles the format:
**`.prbc` is JSON**, and every time is in **tenths of a second**
(`CycleLength: 1000` = 100 s, `Split: 550` = 55 s, `Yellow: 40` = 4.0 s).

```jsonc
{"Controller": {
  "ExecutionFrequency": 1,
  "OffsetReference": "LeadingStartOfGreen",
  "Sequence": {"BarrierGroups": [                 // <- Synchro BRP barrier/ring
    {"RingGroups": [{"VehicleSignalGroups": [1,2]}, {"VehicleSignalGroups": [5,6]}]},
    {"RingGroups": [{"VehicleSignalGroups": [3,4]}, {"VehicleSignalGroups": [7,8]}]}]},
  "VehicleSignalGroups": [                        // <- Synchro Phases table
    {"ID": 1, "Name": "1", "MinGreen": 40, "MaxGreen1": 70, "Yellow": 40,
     "RedClearance": 0, "VehExtension": 10,
     "MinRecall": false, "MaxRecall": false, "DualEntry": false, "StartUp": false}],
  "Patterns": [                                   // <- Synchro Timeplans
    {"ID": 1, "CycleLength": 1000, "Offset": 160,
     "MaxGreenMode": "InhibitMaxGreen", "PermissiveMode": "SingleBand",
     "VehicleSignalGroupsInPattern": [
       {"VehicleSignalGroup": 2, "Split": 550, "Coordinated": true, "MinRecall": true}]}],
  "PatternSchedule": {"PatternScheduleItems": [{"Pattern": 1, "StartTime": 0}]}
}}
```

The Synchro → RBC mapping is therefore direct:

| Synchro UTDF | `.prbc` |
|---|---|
| `Phases.BRP` (barrier/ring/position) | `Sequence.BarrierGroups[].RingGroups[]` |
| `Phases.MinGreen` / `MaxGreen` | `MinGreen` / `MaxGreen1` |
| `Phases.Yellow` / `AllRed` | `Yellow` / `RedClearance` |
| `Phases.VehExt` | `VehExtension` |
| `Phases.Recall` (`1` min, `3` max) | `MinRecall` / `MaxRecall` |
| `Timeplans.Cycle Length` / `Offset` | `Pattern.CycleLength` / `Offset` |
| `Timeplans.Reference Phase` | `VehicleSignalGroupsInPattern[].Coordinated` |

This preserves actuation, which a fixed-time conversion would throw away.

---

## Running it

```bash
# from the repository root, with the venv active
.venv/Scripts/python.exe vissim/scripts/01_import_opendrive.py --open-gui
```

Options: `--net` (source SUMO network), `--name`, `--outdir`, `--progid`
(pin a Vissim version), `--skip-netconvert`, `--visible`, `--open-gui`.

Outputs into `vissim/work/<scenario>/`:

- `<name>.xodr` — OpenDRIVE, georeferenced to true UTM
- `<name>.inpx` — the imported Vissim network
- `<name>_links.csv` — every link/connector with parsed names and geometry
- `<name>_movements.csv` — junctions, bearings and turn movements

A Vissim instance started over COM terminates when Python releases it, so
`--open-gui` launches the saved `.inpx` in a standalone GUI instead.

---

## Status

**Working:** OpenDRIVE conversion with correct georeferencing, COM import into
Vissim 2026, link/connector extraction, junction derivation, approach bearings,
turn classification — validated against the SUMO MatchupTable on Chattanooga.

**Next:** MatchupTable generation/read-back, then demand and signal ingestion,
then the COM writer.

## Open questions

- **Signal heads.** Placement needs a lane-level mapping from movement to
  approach lane. Synchro gives lane groups, not lanes; the lane assignment for
  shared lanes needs a rule.
- **Detectors.** RBC actuation needs vehicle detectors. Synchro carries
  `DetectSize1`/`FirstDetect`; `.prbc` has an empty `VehicleDetectors` list to
  populate, and detectors also have to be created on the Vissim links.
- **Turn threshold.** The 20° thru/turn boundary mislabels two skewed movements
  in Chattanooga relative to SUMO. Worth checking against another network before
  tuning.
