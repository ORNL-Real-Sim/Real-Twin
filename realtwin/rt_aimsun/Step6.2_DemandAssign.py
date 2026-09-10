import sys

from PyANGBasic import *
from PyANGKernel import *
from PyANGConsole import *

def main( argv ):
    # Start a Console
    console = ANGConsole()
    # Load a network
    if console.open( argv[1] ):
        model = console.getModel()

        import os
        import json

        try:
            # External site-packages providing pandas.
            _SITEPACKAGES = r"C:\Users\xh8\AppData\Local\Programs\Python\Python310\Lib\site-packages"
        except Exception:
            input_config = json.loads(argv[-1])
            _SITEPACKAGES = input_config["AIMSUN"]["site_packages"]

        if _SITEPACKAGES not in sys.path:
            sys.path.append(_SITEPACKAGES)

        import pandas as pd

        # Folder of the loaded .ang; holds the solution csv files
        INPUT_DIR = os.path.dirname(str(model.getDocumentFileName()))

        TURNS_CSV = "calib_turns.csv"       # from_id, to_id, percentage
        INFLOWS_CSV = "calib_inflows.csv"   # section_id, flow_vph


        def objects_of_type(model, type_name):
            gktype = model.getType(type_name)
            if gktype is None:
                return []
            objs = model.getCatalog().getObjectsByType(gktype)
            if not objs:
                return []
            return [o for o in objs.values() if o is not None]


        def find_section(model, entry):
            try:
                section = model.getCatalog().find(int(float(entry)))
            except (TypeError, ValueError):
                return None
            if section is None or not section.isA("GKSection"):
                return None
            return section


        turns_path = os.path.join(INPUT_DIR, TURNS_CSV)
        inflows_path = os.path.join(INPUT_DIR, INFLOWS_CSV)
        turns = (pd.read_csv(turns_path, header=None, names=["from_id", "to_id", "pct"])
                 if os.path.exists(turns_path) else pd.DataFrame(columns=["from_id", "to_id", "pct"]))
        inflows = (pd.read_csv(inflows_path, header=None, names=["section_id", "flow"])
                   if os.path.exists(inflows_path) else pd.DataFrame(columns=["section_id", "flow"]))
        # print(f"  :{len(turns)} turning row(s), {len(inflows)} inflow row(s) to assign.")

        # Resolve sections once
        turn_rows = []
        for _, row in turns.iterrows():
            f = find_section(model, row["from_id"])
            t = find_section(model, row["to_id"])
            if f is None or t is None:
                print(f"  :turning {row['from_id']} -> {row['to_id']} not resolvable - skipped.")
                continue
            turn_rows.append((f, t, float(row["pct"])))
        inflow_rows = []
        for _, row in inflows.iterrows():
            s = find_section(model, row["section_id"])
            if s is None:
                print(f"  :section {row['section_id']} not resolvable - skipped.")
                continue
            inflow_rows.append((s, float(row["flow"])))

        # Apply the same values to every traffic state
        states = objects_of_type(model, "GKTrafficState")
        n_turn, n_flow = 0, 0
        for state in states:
            for f, t, pct in turn_rows:
                try:
                    state.setTurningPercentage(f, t, None, pct)
                    n_turn += 1
                except Exception as exc:
                    print(f"  :setTurningPercentage failed on '{state.getName()}' ({exc}).")
            for s, flow in inflow_rows:
                try:
                    state.setEntranceFlow(s, None, flow)
                    n_flow += 1
                except Exception as exc:
                    print(f"  :setEntranceFlow failed on '{state.getName()}' ({exc}).")

        model.getCommander().addCommand(None)
        # print(f"  :Assigned {n_turn} turning value(s) and {n_flow} inflow"
        #       f" value(s) across {len(states)} state(s).")

        console.save( argv[1])
        console.close()
    else:
        console.getLog().addError( "Cannot load the network" )
        print ("cannot load network")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
