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

        DEMAND_NAME = ""
        DEMAND_NAME_CONTAINS = "Traffic Demand"

        # Master control plan to assign.
        MASTER_NAME = "RealTwin Master Control"

        # Names for the created objects.
        SCENARIO_NAME = "RealTwin Dynamic Scenario"
        EXPERIMENT_NAME = "RealTwin Experiment"
        REPLICATION_NAME = "RealTwin Replication"

        # Replication random seed.
        RANDOM_SEED = 100

        # Simulation engine: "micro", "meso" or "hybrid".
        SIMULATION_ENGINE = "micro"

        # Simulation window (HH:MM:SS); SIM_DATE "" -> today.
        SET_SIMULATION_TIME = True
        SIM_START = "08:00:00"
        SIM_END = "09:00:00"
        SIM_DATE = ""            # "" -> today

        # Delete an existing scenario of the same name before creating.
        REPLACE_EXISTING_SCENARIO = True

        def resolve_name(*names):
            """Return ``(object, name)`` for the first console-global name that resolves."""
            for name in names:
                try:
                    value = eval(name)
                except NameError:
                    continue
                if value is not None:
                    return value, name
            return None, None


        def call_first(obj, setters, *args):
            """Call the first method in ``setters`` that exists and does not raise; return its name."""
            for setter in setters:
                fn = getattr(obj, setter, None)
                if fn is None:
                    continue
                try:
                    fn(*args)
                    return setter
                except Exception:
                    continue
            return None


        def dump_methods(obj, *needles):
            """Print the object's methods matching any needle."""
            hits = sorted(m for m in dir(obj)
                          if any(n.lower() in m.lower() for n in needles))
            print("    :available methods (%s): %s"
                  % (getattr(type(obj), "__name__", "?"), hits))


        def objects_of_type(model, type_name):
            """Every catalog object of the named GK type, as a list (handles the {id:obj} map)."""
            gktype = model.getType(type_name)
            if gktype is None:
                return []
            objs = model.getCatalog().getObjectsByType(gktype)
            if not objs:
                return []
            return [o for o in objs.values() if o is not None]


        def find_by_name(model, type_name, name):
            """First catalog object of ``type_name`` whose name matches, or None."""
            for obj in objects_of_type(model, type_name):
                if str(obj.getName()) == name:
                    return obj
            return None


        def delete_object(model, obj):
            """Delete an object through the commander."""
            try:
                model.getCommander().addCommand(obj.getDelCmd())
                return True
            except Exception as exc:
                print("  :could not delete '%s': %s" % (obj.getName(), exc))
                return False


        def pick_demand(model):
            """Choose the GKTrafficDemand to assign."""
            demands = objects_of_type(model, "GKTrafficDemand")
            if not demands:
                return None
            if DEMAND_NAME:
                for d in demands:
                    if str(d.getName()) == DEMAND_NAME:
                        return d
                print("  :demand named '%s' not found; falling back to auto-pick." % DEMAND_NAME)
            if len(demands) == 1:
                return demands[0]
            for d in demands:
                if DEMAND_NAME_CONTAINS and DEMAND_NAME_CONTAINS in str(d.getName()):
                    return d
            print("  :multiple demands and none match '%s'; using the first: %s"
                  % (DEMAND_NAME_CONTAINS, demands[0].getName()))
            return demands[0]


        def get_scenario_folder(model):
            """Folder to file the new scenario in, creating a Scenarios folder if needed."""
            for scen in objects_of_type(model, "GKScenario"):
                for folder in (scen.getParentFolders() or []):
                    if folder is not None:
                        return folder

            tag = "GKModel::scenarios"
            root = model.getCreateRootFolder()
            folder = root.findFolder(tag)
            if folder is None:
                try:
                    folder = root.createFolder("Scenarios", tag)
                except Exception:
                    folder = GKSystem.getSystem().createFolder(root, tag)
                    try:
                        folder.setName("Scenarios")
                    except Exception:
                        pass
            return folder


        def make_qdatetime(hhmmss, date_str):
            """Build a QDateTime from an HH:MM:SS time and an optional ISO date."""
            qdate_cls, _ = resolve_name("QDate")
            qtime_cls, _ = resolve_name("QTime")
            qdt_cls, _ = resolve_name("QDateTime")
            qt_ns, _ = resolve_name("Qt")
            if qdate_cls is None or qtime_cls is None or qdt_cls is None:
                return None
            try:
                iso = qt_ns.ISODate if qt_ns is not None else 1
                if date_str:
                    qdate = qdate_cls.fromString(date_str, iso)
                else:
                    qdate = qdate_cls.currentDate()
                qtime = qtime_cls.fromString(hhmmss, iso)
                return qdt_cls(qdate, qtime)
            except Exception as exc:
                print("  :could not build QDateTime for %s (%s)." % (hhmmss, exc))
                return None



        def create_scenario(model, demand, master):
            """Create the Dynamic Scenario and assign its demand + master control plan."""
            folder = get_scenario_folder(model)

            scenario = GKSystem.getSystem().newObject("GKScenario", model)
            scenario.setName(SCENARIO_NAME)
            folder.append(scenario)

            if call_first(scenario, ("setDemand",), demand) is None:
                print("  :WARNING - setDemand failed."); dump_methods(scenario, "demand")
            else:
                print("  :assigned demand '%s'." % demand.getName())

            if call_first(scenario, ("setMasterControlPlan",), master) is None:
                print("  :WARNING - setMasterControlPlan failed."); dump_methods(scenario, "master", "control")
            else:
                print("  :assigned master control plan '%s'." % master.getName())

            return scenario


        def create_experiment(model, scenario):
            """Create the experiment, set the micro engine, and add it to the scenario."""
            experiment = GKSystem.getSystem().newObject("GKExperiment", model)
            experiment.setName(EXPERIMENT_NAME)

            engine_attr = {"micro": "eMicro", "meso": "eMeso", "hybrid": "eHybrid"}.get(
                SIMULATION_ENGINE.lower(), "eMicro")
            engine = getattr(GKExperiment, engine_attr, None)
            if engine is not None:
                call_first(experiment, ("setSimulatorEngine",), engine)
            one_shot = getattr(GKExperiment, "eOneShot", None)
            if one_shot is not None:
                call_first(experiment, ("setEngineMode",), one_shot)

            if call_first(scenario, ("addExperiment",), experiment) is None:
                print("  :WARNING - scenario.addExperiment failed."); dump_methods(scenario, "experiment")
            try:
                if getattr(experiment, "getScenario", lambda: None)() is None:
                    call_first(experiment, ("setScenario",), scenario)
            except Exception:
                pass

            print("  :created experiment '%s' (engine %s, one-shot)."
                  % (EXPERIMENT_NAME, engine_attr))
            return experiment


        def create_replication(model, experiment):
            """Create the replication, set seed + window, and add it to the experiment."""
            replication = GKSystem.getSystem().newObject("GKReplication", model)
            replication.setName(REPLICATION_NAME)

            if call_first(replication, ("setRandomSeed",), int(RANDOM_SEED)) is None:
                print("  :WARNING - setRandomSeed failed."); dump_methods(replication, "seed")

            if SET_SIMULATION_TIME:
                start_dt = make_qdatetime(SIM_START, SIM_DATE)
                end_dt = make_qdatetime(SIM_END, SIM_DATE)
                if start_dt is not None and end_dt is not None:
                    call_first(replication, ("setInitSimulationTime",), start_dt)
                    call_first(replication, ("setEndSimulationTime",), end_dt)
                    print("  :simulation window %s..%s." % (SIM_START, SIM_END))
                else:
                    print("  :simulation window left at Aimsun defaults (QDateTime "
                          "unavailable) - set it on the replication in the GUI if needed.")

            if call_first(experiment, ("addReplication",), replication) is None:
                print("  :WARNING - experiment.addReplication failed."); dump_methods(experiment, "replicat")
            try:
                if getattr(replication, "getExperiment", lambda: None)() is None:
                    call_first(replication, ("setExperiment",), experiment)
            except Exception:
                pass

            print("  :created replication '%s' (seed %d)." % (REPLICATION_NAME, RANDOM_SEED))
            print("REPLICATION_ID=%s" % replication.getId())
            return replication



        demand = pick_demand(model)
        if demand is None:
            print("  :ERROR - no GKTrafficDemand found.  Run DemandImport_Aimsun.py first.")
            return
        print("  :Using demand '%s' (id %s)." % (demand.getName(), demand.getId()))

        master = find_by_name(model, "GKMasterControlPlan", MASTER_NAME)
        if master is None:
            print("  :ERROR - master control plan '%s' not found.  Run "
                  "ConfigMasterControl.py first (or fix MASTER_NAME)." % MASTER_NAME)
            return
        print("  :Using master control plan '%s' (id %s)." % (master.getName(), master.getId()))

        if REPLACE_EXISTING_SCENARIO:
            existing = find_by_name(model, "GKScenario", SCENARIO_NAME)
            while existing is not None:
                print("  :Removing existing scenario '%s' (id %s)."
                      % (SCENARIO_NAME, existing.getId()))
                if not delete_object(model, existing):
                    break
                nxt = find_by_name(model, "GKScenario", SCENARIO_NAME)
                existing = None if nxt is existing else nxt

        scenario = create_scenario(model, demand, master)
        experiment = create_experiment(model, scenario)
        replication = create_replication(model, experiment)

        for obj in (scenario, experiment, replication):
            try:
                obj.setStatus(GKObject.eModified)
            except Exception:
                pass
        try:
            GKGUISystem.getGUISystem().getActiveGui().invalidateViews()
        except Exception:
            pass
        model.getCommander().addCommand(None)

        print("  :Scenario generation complete -> Scenario '%s' / Experiment '%s' / "
              "Replication '%s'. Run the replication to simulate."
              % (SCENARIO_NAME, EXPERIMENT_NAME, REPLICATION_NAME))

        console.save( argv[1])
        console.close()
    else:
        console.getLog().addError( "Cannot load the network" )
        print ("cannot load network")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
