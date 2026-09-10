import sys

from PyANGBasic import *
from PyANGKernel import *
from PyANGConsole import *
from PyQt5.QtCore import QVariant

def main( argv ):
    # Start a Console
    console = ANGConsole()
    # Load a network
    if console.open( argv[1] ):
        model = console.getModel()

        import os

        # Folder of the loaded .ang; holds the parameter csv
        INPUT_DIR = os.path.dirname(str(model.getDocumentFileName()))

        # One value per line: MinDist, MaxAcc, NormalDec, MaxDec, MinHeadway,
        # SensitivityFactor
        PARAMETER_CSV = "DrivingBehaviorParameter.csv"

        # Vehicle type to modify: by name, or set VEHICLE_TYPE_ID to override
        VEHICLE_NAME = "Car"
        VEHICLE_TYPE_ID = None


        def resolve_vehicle(model):
            catalog = model.getCatalog()
            if VEHICLE_TYPE_ID is not None:
                veh = catalog.find(int(VEHICLE_TYPE_ID))
                if veh is not None and veh.isA("GKVehicle"):
                    return veh
                print(f"  :VEHICLE_TYPE_ID {VEHICLE_TYPE_ID} not found; falling back to '{VEHICLE_NAME}'.")
            veh = catalog.findByName(VEHICLE_NAME)
            if veh is not None and veh.isA("GKVehicle"):
                return veh
            gktype = model.getType("GKVehicle")
            for obj in (catalog.getObjectsByType(gktype) or {}).values():
                if obj is not None and str(obj.getName()) == VEHICLE_NAME:
                    return obj
            return None

        csv_path = os.path.join(INPUT_DIR, PARAMETER_CSV)
        if not os.path.exists(csv_path):
            # print(f"  :ERROR - parameter file not found: {csv_path}")
            # console.close()
            raise Exception("Parameter file not found: " + csv_path)
            # return

        data = []
        with open(csv_path, "r") as file:
            for line in file.readlines():
                row = line.strip().split(",")
                if row and row[0] != "":
                    data.append([float(value) for value in row])

        MinDist = data[0][0]
        MaxAcc = data[1][0]
        NormalDec = data[2][0]
        MaxDec = data[3][0]
        MinHeadway = data[4][0]
        SensitivityFactor = data[5][0]

        vehType = resolve_vehicle(model)
        if vehType is None:
            print(f"  :ERROR - vehicle type '{VEHICLE_NAME}' not found.")
            console.close()
            return

        # min dist/spacing
        vehType.setDataValueByID(GKVehicle.minDistMean, QVariant( MinDist ))
        vehType.setDataValueByID(GKVehicle.minDistMin, QVariant( MinDist*0.8 ))
        vehType.setDataValueByID(GKVehicle.minDistMax, QVariant( MinDist*1.2 ))
        # max acceleration
        vehType.setDataValueByID(GKVehicle.maxAccelMean, QVariant( MaxAcc ))
        vehType.setDataValueByID(GKVehicle.maxAccelMin, QVariant( MaxAcc*0.8 ))
        vehType.setDataValueByID(GKVehicle.maxAccelMax, QVariant( MaxAcc*1.2 ))
        # normal deceleration
        vehType.setDataValueByID(GKVehicle.normalDecelMean, QVariant( NormalDec ))
        vehType.setDataValueByID(GKVehicle.normalDecelMin, QVariant( NormalDec*0.8 ))
        vehType.setDataValueByID(GKVehicle.normalDecelMax, QVariant( NormalDec*1.2 ))
        # max deceleration
        vehType.setDataValueByID(GKVehicle.maxDecelMean, QVariant( MaxDec ))
        vehType.setDataValueByID(GKVehicle.maxDecelMin, QVariant( MaxDec*0.8 ))
        vehType.setDataValueByID(GKVehicle.maxDecelMax, QVariant( MaxDec*1.2 ))
        # min headway
        vehType.setDataValueByID(GKVehicle.minimunHeadwayMean, QVariant( MinHeadway ))
        vehType.setDataValueByID(GKVehicle.minimunHeadwayMin, QVariant( MinHeadway*0.8 ))
        vehType.setDataValueByID(GKVehicle.minimunHeadwayMax, QVariant( MinHeadway*1.2 ))
        # sensitivity factor
        vehType.setDataValueByID(GKVehicle.sensitivityFactorMean, QVariant( SensitivityFactor ))

        model.getCommander().addCommand(None)
        # print(f"  :Applied driving behavior to '{vehType.getName()}' (id {vehType.getId()}): MinDist={MinDist:g} MaxAcc={MaxAcc:g} "
        #       f"NormalDec={NormalDec:g} MaxDec={MaxDec:g} MinHeadway={MinHeadway:g} SensitivityFactor={SensitivityFactor:g}")
        # print(f"  :Applied driving behavior to '{vehType.getName()}' (id {vehType.getId()}): MinDist={MinDist:g} MaxAcc={MaxAcc:g} "
        #       f"NormalDec={NormalDec:g} MaxDec={MaxDec:g} MinHeadway={MinHeadway:g} SensitivityFactor={SensitivityFactor:g}")

        console.save( argv[1])
        console.close()
    else:
        console.getLog().addError( "Cannot load the network" )
        print ("cannot load network")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
