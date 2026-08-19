import sys
import json

from PyANGBasic import *
from PyANGKernel import *
from PyANGConsole import *

def main( argv ):

    print("Input argv: ", argv)

    # Start a Console
    console = ANGConsole()
    # Load a network
    if console.open( argv[1] ):
        model = console.getModel()

        input_config = json.loads(argv[-1])
        IMPORT_FILE = input_config["AIMSUN"]["model_xdor"]

        # Network file to import
        # IMPORT_FILE = r"D:/RealTwin/RealTwin tool development/Aimsun_automation/test/Aimsun automation code/Chatt_test/Model/chatt.xodr"
        # IMPORT_FILE = r"C:\Users\xh8\ORNL_Work\github_workspace\Aimsun-integration-to-RealTwin\chatt_roy\Model\chatt.xodr"

        # Preferred layer name to import into
        LAYER_NAME = "Network"

        # Insertion point offset
        INSERT_X = 0.0
        INSERT_Y = 0.0
        INSERT_Z = 0.0

        def resolve_layer(model):
            """Find the layer to import into: by name, else active, else any."""
            geo = model.getGeoModel()
            layer = geo.findLayer(LAYER_NAME)
            if layer is not None:
                print(f"  :importing into layer '{LAYER_NAME}' (by name).")
                return layer

            layer = geo.getActiveLayer(True)
            if layer is not None:
                print(f"  :layer '{LAYER_NAME}' not found; using active/any layer '{layer.getName()}'.")
                return layer

            layer = next(iter(geo.getLayers()), None)
            if layer is not None:
                print(f"  :using first available layer '{layer.getName()}'.")
            return layer



        import os
        if not os.path.exists(IMPORT_FILE):
            print(f"  :ERROR - file not found: {IMPORT_FILE}")
            return

        layer = resolve_layer(model)
        if layer is None:
            print("  :ERROR - no layer found to import into.  Create a layer first.")
            return

        print(f"  :Importing {os.path.basename(IMPORT_FILE)} ...")
        ok = model.importFile(IMPORT_FILE, layer,
                              GKPoint(INSERT_X, INSERT_Y, INSERT_Z), GKBBox())
        if ok:
            print(f"  :Import complete into layer '{layer.getName()}'.")
        else:
            print("  :Import returned False - check the Aimsun log for details.")

        try:
            model.getCommander().addCommand(None)
        except Exception:  # noqa: BLE001, S110
            pass
        try:
            GKGUISystem.getGUISystem().getActiveGui().invalidateViews()
        except Exception:  # noqa: BLE001, S110
            pass

        console.save( argv[1])
        console.close()
    else:
        console.getLog().addError( "Cannot load the network" )
        print("Cannot load the network")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
