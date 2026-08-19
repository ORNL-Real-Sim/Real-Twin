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

        SCENARIO_NAME = "RealTwin Dynamic Scenario"
        APPLY_TO_ALL_SCENARIOS = False       # apply to every scenario instead

        # --- Statistics ---
        STAT_GENERATE_TIME_SERIES = True
        STAT_STORE_IN_DATABASE = True
        STAT_INTERVAL = "01:00:00"

        # --- Path Assignment ---
        PATH_KEEP_IN_MEMORY = False

        # --- Detection ---
        DET_GENERATE_TIME_SERIES = False
        DET_STORE_IN_DATABASE = False
        DET_INTERVAL = "00:00:01"

        # --- Trajectories ---
        TRAJECTORIES_STORE_IN_DATABASE = False

        # --- Micro / Detection Cycle ---
        DETECTION_CYCLE_SAME_AS_STEP = False
        DETECTION_CYCLE_SECONDS = 1.0

        # --- Store XML Animation ---
        STORE_XML_ANIMATION = False

        # --- Database (Store Locations tab): "sqlite" | "access" | "project" | "" ---
        DATABASE_MODE = "sqlite"



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
            """Every catalog object of the named GK type, as a list."""
            gktype = model.getType(type_name)
            if gktype is None:
                return []
            objs = model.getCatalog().getObjectsByType(gktype)
            if not objs:
                return []
            return [o for o in objs.values() if o is not None]


        def make_timeduration(hhmmss):
            """Build a GKTimeDuration from an 'HH:MM:SS' string, or None."""
            cls, _ = resolve_name("GKTimeDuration")
            if cls is None:
                return None
            try:
                return cls.fromString(hhmmss)
            except Exception as exc:
                print("  :could not build GKTimeDuration('%s') (%s)." % (hhmmss, exc))
                return None


        def set_database(scenario):
            """Set the scenario's output database per DATABASE_MODE."""
            mode = (DATABASE_MODE or "").strip().lower()
            if not mode:
                return

            db = None
            getter = getattr(scenario, "getDB", None)
            if getter is not None:
                for args in ((False,), ()):
                    try:
                        db = getter(*args)
                        break
                    except Exception:
                        continue
            if db is None:
                print("    :WARNING - could not read scenario DB info; database unchanged.")
                return

            if mode == "project":
                call_first(db, ("setUseProjectDB",), True)
                label = "Use Project Outputs Database"
            elif mode in ("sqlite", "access"):
                call_first(db, ("setUseProjectDB",), False)
                call_first(db, ("setAutomatic",), True)
                driver = "QSQLITE" if mode == "sqlite" else "ACCESS"
                call_first(db, ("setDriverName",), driver)
                label = "Automatic Using %s" % ("SQLite" if mode == "sqlite" else "Access")
            else:
                print("    :WARNING - DATABASE_MODE '%s' not recognised; database unchanged."
                      % DATABASE_MODE)
                return

            try:
                scenario.setDB(db)
                print("      %-34s <- %s" % ("Database", label))
            except Exception as exc:
                print("    :WARNING - setDB failed (%s)." % exc)


        def apply_setting(data, setters, value, label, applied):
            """Call the first working setter; record the outcome for the summary."""
            used = call_first(data, setters, value)
            if used is None:
                applied.append((label, "FAILED", value))
                print("    :WARNING - could not set %s; dumping candidates." % label)
                dump_methods(data, setters[0][3:] if setters and setters[0].startswith("set")
                             else setters[0])
            else:
                applied.append((label, used, value))



        def adjust_scenario(scenario):
            """Apply all Outputs-to-Generate settings to one scenario's input data."""
            data = scenario.getInputData()
            if data is None:
                print("  :scenario '%s' has no input data - skipped." % scenario.getName())
                return

            applied = []

            # --- Statistics ---
            apply_setting(data, ("setKeepHistoryStat",), STAT_GENERATE_TIME_SERIES,
                          "Statistics/GenerateTimeSeries", applied)
            apply_setting(data, ("enableStoreStatistics",), STAT_STORE_IN_DATABASE,
                          "Statistics/StoreInDatabase", applied)
            stat_int = make_timeduration(STAT_INTERVAL)
            if stat_int is not None:
                apply_setting(data, ("setStatisticalInterval",), stat_int,
                              "Statistics/Interval=%s" % STAT_INTERVAL, applied)

            # --- Path Assignment ---
            apply_setting(data, ("setKeepPathsInMemory",), PATH_KEEP_IN_MEMORY,
                          "PathAssignment/KeepInMemory", applied)

            # --- Detection ---
            apply_setting(data, ("setKeepHistoryDet",), DET_GENERATE_TIME_SERIES,
                          "Detection/GenerateTimeSeries", applied)
            apply_setting(data, ("enableStoreDetection",), DET_STORE_IN_DATABASE,
                          "Detection/StoreInDatabase", applied)
            det_int = make_timeduration(DET_INTERVAL)
            if det_int is not None:
                apply_setting(data, ("setDetectionInterval",), det_int,
                              "Detection/Interval=%s" % DET_INTERVAL, applied)

            # --- Trajectories ---
            apply_setting(data, ("setTrajectoriesStatistics",), TRAJECTORIES_STORE_IN_DATABASE,
                          "Trajectories/StoreInDatabase", applied)

            # --- Micro / Detection Cycle ---
            apply_setting(data, ("setCycleDetectionAsSimulationStep",),
                          DETECTION_CYCLE_SAME_AS_STEP, "Micro/DetectionCycleSameAsStep", applied)
            if not DETECTION_CYCLE_SAME_AS_STEP:
                apply_setting(data, ("setCycleDetection",), float(DETECTION_CYCLE_SECONDS),
                              "Micro/DetectionCycle=%gs" % DETECTION_CYCLE_SECONDS, applied)

            # --- Store XML Animation ---
            apply_setting(data, ("setSaveAnimation",), STORE_XML_ANIMATION,
                          "StoreXMLAnimation", applied)

            try:
                scenario.setInputData(data)
            except Exception:
                pass

            ok = sum(1 for _, used, _ in applied if used != "FAILED")
            print("  :scenario '%s' - applied %d/%d settings."
                  % (scenario.getName(), ok, len(applied)))
            for label, used, value in applied:
                print("      %-34s <- %s%s"
                      % (label, value, "" if used != "FAILED" else "   (FAILED)"))

            # Output database (Store Locations tab).
            set_database(scenario)

            def _get(name):
                fn = getattr(data, name, None)
                try:
                    return fn() if fn is not None else "?"
                except Exception:
                    return "?"
            print("      verify: activateStatistics=%s storeStatisticsEnabled=%s "
                  "keepHistoryStat=%s keepHistoryDet=%s storeDetectionEnabled=%s "
                  "keepPathsInMemory=%s"
                  % (_get("getActivateStatistics"), _get("storeStatisticsEnabled"),
                     _get("getKeepHistoryStat"), _get("getKeepHistoryDet"),
                     _get("storeDetectionEnabled"), _get("getKeepPathsInMemory")))



        scenarios = objects_of_type(model, "GKScenario")
        if not scenarios:
            print("  :ERROR - no GKScenario found.  Run ScenarioGeneration.py first.")
            return

        if APPLY_TO_ALL_SCENARIOS:
            targets = scenarios
        else:
            targets = [s for s in scenarios if str(s.getName()) == SCENARIO_NAME]
            if not targets:
                print("  :ERROR - scenario '%s' not found.  Available: %s"
                      % (SCENARIO_NAME, [s.getName() for s in scenarios]))
                return

        for scenario in targets:
            adjust_scenario(scenario)

        model.getCommander().addCommand(None)
        try:
            GKGUISystem.getGUISystem().getActiveGui().invalidateViews()
        except Exception:
            pass
        print("  :Scenario adjustment complete - %d scenario(s)." % len(targets))

        console.save( argv[1])
        console.close()
    else:
        console.getLog().addError( "Cannot load the network" )
        print ("cannot load network")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
