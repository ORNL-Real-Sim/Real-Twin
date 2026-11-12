# Bringing a new site into the Vissim pipeline

The pipeline's code is site-agnostic; what changes from one site to the next
is the data and the geometry it describes, and each new site so far has turned
up two or three things the previous ones did not. This is the procedure for
finding them, in the order they show up, with the check that proves each one
and the place its fix belongs. The commands themselves are in the
[README](../README.md#running-it-end-to-end); this document is about judgment.

Chattanooga (6 signals, 10 junctions) and Phoenix (13 signals, 115 junctions)
are the two sites it has been through. Everything below was learned on them.

## 1. Before running anything

Get, and look at, all of it before stage 1:

| what | where it goes | check |
|---|---|---|
| SUMO network `<name>.net.xml` | anywhere; the dataset folder is found from it | has a `netOffset` in `<location>` (stage 1 re-applies it; without one the network lands at the origin) |
| GridSmart counts, one `.xls` per surveyed junction | `<dataset>/Traffic/` | each file is named after its **SUMO junction** (`4.xls`, `J9.xls`); open one and confirm the count period you want is in it |
| Synchro UTDF export, one `.csv` | `<dataset>/Control/` | its `Lanes` table has an `Allow RTOR` row; its `INTID`s are what you will type into the MatchupTable, and one of them may not be a number (Phoenix's `J9`) |
| SUMO's `MatchupTable_SUMO.xlsx` | beside the network | **comparison only**, never read by the pipeline; it tells you which junctions SUMO surveyed |

`Traffic/` and `Control/` must sit at or above the network, within three
folders, because stage 1 records that folder in `work/<site>/dataset.json` and
every later stage reads its data from there. A dataset supplied with a
misspelt or doubled folder name (`Pheonix/Pheonix/`) is kept as supplied: the
README's commands name it, so do not rename it.

Agree with whoever owns the data, before starting:

- the **period** (both sites: 08:00–09:00; the counts are in 15-minute bins and the
  Synchro plan must cover it);
- **which junctions have counts** — the rest have no demand of their own and
  stage 6 estimates it; knowing the list up front tells you what stage 5 can
  and cannot verify;
- whether any **approach is known to be over capacity** in SUMO (Phoenix's 7th
  St southbound), because the Vissim model will show the same shortfall and it
  is not a fault to chase.

## 2. The run, and what must be true after each stage

Run every stage from the repository root, one Vissim session at a time, with
no Vissim window open. Each stage prints its own checks; a stage that passes
them has nothing left for you to verify by hand. Write down the numbers: they
become the site's "what you should see" section in the README.

| stage | must be true before moving on | if not |
|---|---|---|
| 1 import | `Re-applying UTM offset (...)`; link count against drivable OpenDRIVE roads differs only by U-turn paths that do not import; junction and movement counts match SUMO's MatchupTable on every approach you can compare | §3.1 |
| 2 matchup | after the manual step: `N junctions name their data files`, every surveyed junction signalised, no WARNING lines; `Turn_GridSmart`/`Turn_Synchro` identical to SUMO's table except where you can explain the difference | §3.2 |
| 3 demand | one vehicle input per counted entry at 4 × its 15-min count; every non-zero count on the movement SUMO puts it on; entries nobody counted written at 0 | §3.3 |
| 4 signals | one controller per signalised junction; `Coverage: N/N signalised lanes carry exactly one signal head`; right-turn-on-red signs = right turns Synchro permits; every Synchro timing value identical in the `.prbc` files; conflict tally with nothing unranked inside a junction | §3.4 |
| 5 check | removed vehicles explained link by link; entry-fed approaches within GEH 5 (`--compare-counts`); turn shares within 5 points; upstream-fed approaches short *only* for lack of demand | §3.5 |
| 6 calibrate | mean GEH per approach falls; the approaches that still miss are the same ones SUMO misses, or you know why not | §3.6 |

## 3. Triage: symptom → cause → proof → where the fix goes

Three places a fix can go, and the rule for choosing:

- **code** (`rt_vissim/`, with a test) when the site shows a *general* case the
  rules did not cover — a geometry, a Synchro value, a naming scheme any site
  could have;
- **data / MatchupTable** when the site's files are what they are and the model
  should say so — a junction with no counts, an over-capacity entry, a folder
  name;
- **documentation** (README "Open questions", this file) when it is a limit of
  the import or of Vissim that no code of ours changes.

Never fix a site's symptom by special-casing its junction numbers in code.

### 3.1 Stage 1 — import

| symptom | cause | proof | fix |
|---|---|---|---|
| network at the origin, far from its aerial | no `netOffset` applied | the `Re-applying UTM offset` line is missing | data: the network's `<location>`; code only if the offset is there and not read |
| links ≠ drivable roads by more than a few | roads Vissim cannot import | compare the WARNING's count with the U-turn paths in the OpenDRIVE | document; two per site so far, both U-turns |
| a junction SUMO lists is missing | two-leg junctions (a road changing name) carry no turns | `*_junctions.csv` against `MatchupTable_SUMO.xlsx` | expected; note it in the README figures |

### 3.2 Stage 2 — MatchupTable

| symptom | cause | proof | fix |
|---|---|---|---|
| `No junction names its data files yet` | the manual step is not done | — | MatchupTable: `File_GridSmart`, `File_Synchro`, `IntersectionID_Synchro` on the first row of each surveyed junction's block |
| a surveyed junction matches no file | GridSmart file name ≠ SUMO junction name | `Name_OpenDrive` in `*_junctions.csv` | data: name the file after the junction, or the junction row after the file |
| a turn code differs from SUMO's | a movement SUMO has and this network does not (a U-turn counted 0, a lane the import dropped) | stage 2's comparison line | explain it or fix the network; one per site so far, explained |
| `... is open in another program` | the table is open in Excel | — | close Excel |

### 3.3 Stage 3 — demand

| symptom | cause | proof | fix |
|---|---|---|---|
| many entries at volume 0 | no count traces to them | the stage's own line (Phoenix: 52 of 67) | expected; stage 6 estimates them |
| a count on the wrong movement | approach bearing or turn ranking off for an odd geometry | `*_movements.csv` for that junction against the GridSmart sheet | code (`network.py`) with a test on that geometry |
| a route pinned to one lane path | the turn leaves from a lane the diverge does not reach without pinning | the `pinned` line | expected; it is how double turns are kept |

### 3.4 Stage 4 — signals, heads, conflicts, RTOR

| symptom | cause | proof | fix |
|---|---|---|---|
| `INTID ... is not a number` | Synchro names an intersection (`J9`) | the WARNING names the controller number it chose | expected; write the number down |
| `COVERAGE GAP` | Synchro's lane table and the network disagree on lanes | the gap names the lane; compare the Synchro `Lanes` row with the link's `NumLanes` | data if the network is right, document if Synchro is (Phoenix: 3 lanes in Synchro, 2 in the network) |
| a lane shared by two phases | left and through on one lane, different phases | the WARNING; one head per lane shows the through's phase (manual p. 652) | expected; documented in the README |
| RTOR signs ≠ permitted right turns | a right turn with no connector of its own, or `Allow RTOR` 0 | `diagnostics/rtor_signs.py --work ...` lists every right turn with both values | code if a permitted turn has a connector and no sign; data if Synchro says 0 |
| conflict tally reports pairs no rule reached inside one junction | a geometry the rules have not met | `diagnostics/conflict_dryrun.py --work ...` lists them by turn pair with an example; equal rank at a signal, head-on, and passive areas are expected | code (`conflicts.py`) with a test built from that example |
| `SG N ... Veh Extension should be > 0` from Vissim | Synchro's `.prbc` detail | the controller file | document; harmless |

### 3.5 Stage 5 — what the run did

| symptom | cause | proof | fix |
|---|---|---|---|
| vehicles stranded on link N | a routing decision too close to the diverge for the lane change | `diagnostics/describe_link.py --work ... N`: a short entry link with an input and a decision a few metres in | document as a geometry limit (Chattanooga link 43); a longer entry link is a network change, not ours |
| an entry delivers far less than counted | capacity: the entry's lanes cannot pass the count | SUMO leaves about the same number uninserted on the same demand | document; not a fault to calibrate away |
| an entry-fed approach off by GEH > 5 for no capacity reason | a route, a lane, a count on the wrong movement | the `--compare-counts` table, per movement | code or data, by what the movement table shows |
| upstream-fed approaches short | their demand passes unsurveyed junctions written at 0 | expected until stage 6 | — |
| turn shares off on an approach | routing decision placement or a pinned route missing | the shares columns; the approach's decisions in `describe_link.py` | code (`routes.py`) |
| vehicles enter an RTOR sign on red without stopping | usually amber creep at a few km/h, or a vehicle created on a stub link next to the sign | `diagnostics/rtor_watch.py --work ...` prints each one with its speed history | expected if explained by those two; otherwise code (`heads.py`) |

### 3.6 Stage 6 — calibration

| symptom | cause | proof | fix |
|---|---|---|---|
| the 85% per-approach target is not reached | the counts along a stretch do not balance, or an inflow sits at the 200 veh/h ceiling | the same approaches miss in SUMO's run; the summary's inflow values | document; raise the ceiling only with a reason the counts support |
| a split lands at 0 or 1 | a turn no downstream count needs | the summary's values | expected; it is an estimate, say so |

## 4. Done

A site is done when:

1. stages 1–5 pass their own checks, and every WARNING line is one of the
   expected ones listed for that stage in the README;
2. `rtor_signs.py` reports 0 mismatches and `conflict_dryrun.py` reports
   nothing unreached inside a junction beyond the expected kinds;
3. the README has a **What you should see on \<site\>** section with the
   figures from each stage, written so a fresh run that differs is wrong;
4. every new limit found is in the README's **Open questions**, with the
   junction or link it concerns;
5. the tests still pass (`python -m pytest vissim/tests/ -q`) and any new rule
   has one built from the site's own example.

Nothing from the dataset is committed.
