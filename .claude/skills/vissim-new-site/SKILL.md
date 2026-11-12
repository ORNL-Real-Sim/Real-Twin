---
name: vissim-new-site
description: Bring a new site (network + GridSmart counts + Synchro export) through the RealTwin Vissim pipeline, or diagnose a stage that fails on an existing one. Use when the user asks to build, run, check or calibrate a Vissim model for a dataset, names a new city or corridor for Vissim, or asks why a stage's numbers are off.
---

# Vissim: a new site through the pipeline

The procedure and the triage tables live in the repository, not here. Read
them first, every time:

1. `vissim/docs/NEW_SITE.md` — what to get before starting, what must be true
   after each stage, symptom → cause → proof → where the fix belongs.
2. `vissim/README.md`, section "Running it end to end" — the commands, with the
   expected output of every stage on Phoenix and Chattanooga.

This file holds only what those documents cannot: how to work while doing it.

## How to work

- **Stay in `vissim/`.** The SUMO pipeline is not to be changed for a Vissim
  problem. Commits go to the Vissim feature branch, never `main`, and never
  include `datasets/`.
- **One Vissim session at a time**, no Vissim window open, every command from
  the repository root with absolute paths — the shell's working directory
  drifts. Put any ad-hoc Python in a file and run the file; inline `python -c`
  breaks on quoting in PowerShell.
- **Before stage 2's manual step, stop and ask.** Naming each surveyed
  junction's files in the MatchupTable is the one hand input; do it only when
  the user has confirmed which GridSmart file belongs to which junction, or
  the README's table for that site already says.
- **Run the stage, read every line it prints, then decide.** Each stage checks
  itself. A WARNING is either on the README's expected list for that stage or
  it is the next thing to look at — there is no third kind.
- **Prove before fixing.** For any symptom, run the diagnostic the triage
  table names (`vissim/scripts/diagnostics/`) and quote its output. Then
  choose where the fix goes by the rule in NEW_SITE.md §3: code with a test
  when the case is general, data or MatchupTable when the site is simply what
  it is, documentation when it is a limit of the import or of Vissim. Never
  special-case a site's junction numbers in code.
- **Long runs go in the background.** Stage 5 is minutes, stage 6 hours.
  Start them, say what is running and what will be checked when it finishes,
  and do not poll.
- **Write the numbers down as you go.** The finished site gets a "What you
  should see on <site>" section in the README, its new limits go in "Open
  questions", and findings about the dataset itself (as opposed to the code)
  go in memory, not the README.
- **Report faithfully.** If a check fails, say so with the output. If a stage
  was skipped, say that. "Done" means the Done list in NEW_SITE.md §4 is met.
