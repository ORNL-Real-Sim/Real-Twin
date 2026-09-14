# RealTwin VISSIM pipeline (development sandbox)

A PTV Vissim counterpart to the RealTwin SUMO pipeline, driven over the Vissim
COM API. Same steps and same method as SUMO — network ingestion, a MatchupTable
join, demand ingestion, signal control ingestion — but **standalone after the
OpenDRIVE import**, because Vissim renumbers the whole network on import.

Developed against **PTV Vissim 2026** (COM ProgID `VISSIM.Vissim-64.2600`).

---

## Running it end to end

Nothing here needs the SUMO pipeline to have been run. The inputs are a SUMO
network file and the two data folders; the Vissim MatchupTable is generated
here, and the SUMO one is only ever used for **comparison**.

### What you need

| | |
|---|---|
| PTV Vissim 2026, licensed | the COM API is used for every write |
| SUMO's `netconvert` on `PATH` | stage 1 only, to make the OpenDRIVE file |
| Python 3.11+ | `pandas`, `openpyxl`, `pywin32` |

Inputs:

```
<name>.net.xml        RealTwin SUMO network
Traffic/              GridSmart turning-movement counts (.xls, one per intersection)
Control/              Synchro UTDF export (.csv, one file for the corridor)
```

### Step 1 — import the network

```bash
python vissim/scripts/01_import_opendrive.py     --net datasets/example1/updated_net/chatt.net.xml     --outdir vissim/work/mynet --name chatt
```

`netconvert` writes the `.xodr`, Vissim imports it, and the link table is read
back out. Writes `chatt.inpx`, `chatt_links.csv`, `chatt_movements.csv`,
`chatt_junctions.csv`.

Run this **first even if you already have an `.inpx`**: `chatt_links.csv`
carries the `FromPos` column, and that is what places signal heads upstream of
the stop line in stage 4.

### Step 2 — MatchupTable, and the one manual step

```bash
python vissim/scripts/02_matchup_table.py     --movements vissim/work/mynet/chatt_movements.csv     --outdir vissim/work/mynet
```

This derives the junctions, approaches, bearings and turns on its own, then
stops and asks for the only thing it cannot know — **which data file belongs to
which junction**. Open `MatchupTable.xlsx` and fill three columns for each
signalised junction:

| column | what to put |
|---|---|
| `File_GridSmart` | the GridSmart workbook for that intersection |
| `File_Synchro` | the Synchro UTDF file |
| `IntersectionID_Synchro` | that intersection's Synchro `INTID` |

Each GridSmart workbook names the intersection it covers, so opening one tells
you which junction it belongs to. On Chattanooga that is **6 surveyed junctions
× 3 values**; `File_Synchro` is the same file for the whole corridor.

This is the same manual step the SUMO pipeline has —
`rt_matchup_table_generation.py` writes those three columns blank too, which is
why the repository carries a hand-filled `MatchupTable_updated.xlsx`.

**Then run the same command again.** The second run keeps what you typed and
derives the rest: on Chattanooga 80 of 104 `Turn_GridSmart` and `Turn_Synchro`
codes, the intersection names read out of the workbooks themselves, and a
flow-continuity check between adjacent junctions. Pass `--regenerate` only to
rebuild the table from the movement table, discarding your entries.

It also says which junctions name nothing:

```
:6 junctions name their data files.
:4 name none and will carry no counts: 5, 7, 13, 15. That is right for an
 intersection nobody surveyed, and wrong if one was missed.
```

Worth reading: a junction left out here carries no demand, and that surfaces
much later as an entry link written at volume zero.

*Shortcut:* `--seed-from <a filled MatchupTable>` copies those values from a
table that already has them. It expects one keyed on **SUMO** junction IDs,
which is what RealTwin's SUMO pipeline produces — hand it a Vissim-keyed table
and it translates IDs that need no translating, seeding the wrong junctions.
Convenience only; the pipeline does not need it.

### Step 3 — demand

```bash
python vissim/scripts/03_write_demand.py     --inpx vissim/work/mynet/chatt.inpx     --matchup vissim/work/mynet/MatchupTable.xlsx     --links vissim/work/mynet/chatt_links.csv     --movements vissim/work/mynet/chatt_movements.csv     --start 08:00 --end 09:00
```

Turning-movement counts become vehicle inputs on the entry links and one static
routing decision per approach, carrying the counts as relative flows. Writes
`chatt_demand.inpx`.

Consecutive routing decisions are **combined** by default
(`CombineStaRoutDec`), which Vissim resolves at simulation start: a vehicle
passing the first decision already knows the turn it will make at the second,
and changes lanes for it in time.

Without that a vehicle learns a turn only on reaching its approach, and a short
approach leaves no room to cross into the turn lane — Vissim deletes it.  On
Chattanooga that is the difference between **93 vehicles destroyed and 4**, out
of 4,061:

```
                  stranded   waited   total
separate decisions      93        8     101   2.49%
combined  decisions      4        8      12   0.30%
```

Only one of 111 movements changes, by exactly the number of vehicles that had
been deleted on that approach — the traffic is restored, not redistributed.
`--no-combine-routes` writes `<name>_demand_separate.inpx` for comparison.

The 4 that remain are on a 7.5 m entry stub where vehicles are generated 3.8 m
before the diverge; there is no upstream decision to combine with, so that one
needs a longer entry link rather than better information.

### Step 4 — signal control

```bash
python vissim/scripts/04_write_signals.py     --inpx vissim/work/mynet/chatt_demand.inpx     --matchup vissim/work/mynet/MatchupTable.xlsx     --links vissim/work/mynet/chatt_links.csv     --movements vissim/work/mynet/chatt_movements.csv     --strict
```

Synchro timings become one `.prbc` Ring Barrier Controller per junction, then
signal heads, detectors, conflict areas and right-turn-on-red stop signs are
written over COM. Writes `chatt_demand_signals.inpx`.

`--strict` refuses to write if any lane of a signalised approach would be left
without a signal head. Use it: a lane with no head crosses the junction
unsignalised, and nothing in a phase audit will notice.

### What you should see on Chattanooga

Every number below is checked by the pipeline itself, so a fresh run that
differs means something is wrong:

```
stage 1   441 links (186 links, 255 connectors), 10 junctions
stage 2   104 movements, 10 junctions, 34 approaches
stage 3   16 vehicle inputs, 30 routing decisions, 100 routes, 4,061 vehicles
stage 4   6 controllers, 32 signal groups, 69 signal heads, 68 detectors,
          24 right-turn-on-red stop signs
          Coverage: 69/69 lanes carry exactly one signal head,
                    69/69 of them upstream of the stop line
```

### Tests

```bash
python -m pytest vissim/tests/ -q        # 117 tests, no Vissim licence needed
```

---

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
| `update_matchup_table` fills the derivable columns | same, but codes derived per row rather than positionally |
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
| Turn mix | R28 T30 L24 U22 | R28 T30 L24 U22 |

The tables agree on **all 104 movements, 34 approaches and 10 junctions**, column
for column.  Approach bearings, derived independently on each side, agree to
0.073° on average.

The last two turn labels only agreed once SUMO's own rule was ported from
`NBNode::getDirection`: a 44° straight band **plus** a check for whether another
movement off the same approach is straighter.  A fixed angular threshold cannot
express that second part, and mislabelled two skewed approaches.

---

## Layout

```
vissim/
  rt_vissim/
    com.py          COM session, OpenDRIVE import, collection reads    [done]
    network.py      links -> junctions, bearings, turn movements       [done]
    ir.py           simulator-agnostic scenario IR                     [done]
    matchup.py      generate / read the Vissim MatchupTable            [done]
    demand.py       GridSmart turn counts -> IR                        [done]
    routes.py       vehicle inputs and static routing decisions        [done]
    signal.py       Synchro UTDF -> IR                                 [done]
    rbc.py          IR -> .prbc Ring Barrier Controller files          [done]
    heads.py        per-lane signal heads, detectors, RTOR             [done]
    conflicts.py    conflict-area right of way                         [done]
    writer.py       IR -> Vissim, over COM                             [done]
    pipeline.py     orchestrator                                       [not built]
  scripts/
    01_import_opendrive.py    netconvert + import + link/movement CSVs
    02_matchup_table.py       MatchupTable from the link table
    03_write_demand.py        vehicle inputs + routing decisions
    04_write_signals.py       controllers, heads, detectors, conflicts, RTOR
  tests/                      117 tests, no Vissim licence needed
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

## Artefacts and options

Everything lands in `vissim/work/<scenario>/`:

| file | written by | what it is |
|---|---|---|
| `<name>.xodr` | stage 1 | OpenDRIVE, georeferenced to true UTM |
| `<name>.inpx` | stage 1 | the imported Vissim network |
| `<name>_links.csv` | stage 1 | every link and connector, incl. `FromLanes` and `FromPos` |
| `<name>_movements.csv` | stage 1 | junctions, bearings, turn movements |
| `<name>_junctions.csv` | stage 1 | junction membership and coordinates |
| `MatchupTable.xlsx` | stage 2 | the join between network, counts and signals |
| `<name>_demand.inpx` | stage 3 | network + vehicle inputs + routing decisions |
| `rbc_timings_<INTID>.prbc` | stage 4 | one Ring Barrier Controller per junction |
| `<name>_demand_signals.inpx` | stage 4 | the finished model |

Useful flags: `--progid` pins a Vissim version, `--visible` shows the GUI,
`--skip-netconvert` reuses an existing `.xodr`, `--no-conflicts` / `--no-rtor` /
`--no-heads` skip parts of stage 4.

A Vissim instance started over COM terminates when Python releases it, so
`--open-gui` launches the saved `.inpx` in a standalone GUI instead.

Do not run two Vissim COM sessions at once: `Dispatch` attaches to the existing
instance, and when one script exits it takes the other's instance down.

---

## Status

**Working:** OpenDRIVE conversion with correct georeferencing, COM import into
Vissim 2026, link/connector extraction, junction derivation, approach bearings,
turn classification — validated against the SUMO MatchupTable on Chattanooga.
MatchupTable generation and read-back, laid out so columns A–N match the SUMO
table position for position. GridSmart ingestion into vehicle inputs and static
routing decisions — all six Chattanooga exports parse to 96 quarter-hour bins.

MatchupTable auto-fill. As in the SUMO flow, the only hand input is a
per-junction seed — which GridSmart file and which Synchro `INTID` belong to each
junction, plus the one `File_Synchro`; 13 of Chattanooga's 104 rows. The
intersection name, date, `Need calibration?` and all 80 `Turn_GridSmart` /
`Turn_Synchro` codes are derived. Seeded from the SUMO table's own user input,
102 of 104 codes in each column match it exactly.

**Next:** the COM writer. Nothing reaches a `.inpx` until it exists, so demand
cannot yet be inspected in Vissim. Then Synchro UTDF → `.prbc`, then RTOR and
stop control.

## Open questions

- **Detector position.** Synchro's `DetectPos1` is ignored: two approaches ask
  for a 6 ft detector 224 ft upstream (an advance detector) and both are placed
  at the stop bar instead, raised to a car length.  Fine for presence detection,
  wrong if advance detection matters.
- **Counts.** Junctions 5, 7, 13 and 15 have no GridSmart data, so four entry
  links carry zero volume and six movements are starved — including both of
  junction 4's throughs (1,032 and 958 counted, 0 simulated). That is a data
  gap, not a code gap.
- **Calibration.** `RealTwin.Calibration.Vissim` is still a stub.
- **Right turn on red.** Not a port of the SUMO path. SUMO has no RTOR concept,
  so RealTwin folds Synchro's `Allow RTOR` into the `tlLogic` state string as a
  permissive `s`. Vissim models it structurally instead — a conflict area or
  priority rule on the right-turn connector, with the signal head omitted or set
  to allow red-on-right — so this stage has to be designed against Vissim
  semantics rather than translated. Same for stop/yield control on unsignalised
  approaches.
