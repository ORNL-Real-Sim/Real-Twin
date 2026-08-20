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

        CONTROL_PLAN_NAME = "RealTwin Synchro Actuated"

        # Name of the created master control plan.
        MASTER_NAME = "RealTwin Master Control"

        # Schedule start and duration, in seconds from midnight.
        SCHEDULE_FROM = 0            # 00:00:00
        SCHEDULE_DURATION = 86400    # 24 h

        # Delete an existing master of the same name before creating.
        REPLACE_EXISTING_MASTER = True

        # Attach the master to every dynamic scenario automatically.
        ATTACH_TO_SCENARIOS = True


        def resolve_name(*names):
            """Return (object, name) for the first name that resolves, else (None, None)."""
            for name in names:
                try:
                    value = eval(name)
                except NameError:
                    continue
                if value is not None:
                    return value, name
            return None, None


        def call_first(obj, setters, *args):
            """Call the first method in setters that exists and does not raise; return its name or None."""
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
            """Return a list of every catalog object of the named GK type."""
            gktype = model.getType(type_name)
            if gktype is None:
                return []
            objs = model.getCatalog().getObjectsByType(gktype)
            if not objs:
                return []
            return [o for o in objs.values() if o is not None]


        def find_by_name(model, type_name, name):
            """First catalog object of type_name whose name matches, or None."""
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


        def get_master_folder(model):
            """Return (creating if needed) the folder that holds master control plans."""
            folder_name = "GKModel::masterControlPlans"
            folder = model.getCreateRootFolder().findFolder(folder_name)
            if folder is None:
                folder = GKSystem.getSystem().createFolder(model.getCreateRootFolder(),
                                                           folder_name)
            return folder


        # Candidate master control plan type names.
        MASTER_TYPE_CANDIDATES = ("GKMasterControlPlan", "GKControlPlanMaster")


        def resolve_master_type(model):
            """Name of the available master control plan type, or None."""
            for cand in MASTER_TYPE_CANDIDATES:
                if model.getType(cand) is not None:
                    return cand
            return None


        def create_master(model, name):
            """Create the master control plan object and file it in its folder."""
            type_name = resolve_master_type(model)
            if type_name is None:
                print("  :ERROR - no master control plan type found (tried %s)."
                      % ", ".join(MASTER_TYPE_CANDIDATES))
                return None

            master = GKSystem.getSystem().newObject(type_name, model)
            master.setName(name)
            get_master_folder(model).append(master)
            print("  :Created master control plan (type %s)." % type_name)
            return master


        def add_plan_to_master(master, plan, from_s, duration_s):
            """Schedule plan inside master from from_s for duration_s."""
            item_cls, item_name = resolve_name("GKScheduleMasterControlPlanItem",
                                                "GKScheduleControlPlanItem")
            if item_cls is not None and hasattr(master, "addToSchedule"):
                try:
                    item = item_cls()
                    if call_first(item, ("setControlPlan", "setControlPlanItem",
                                         "setItem"), plan) is None:
                        print("  :WARNING - no setter linked the control plan to the "
                              "schedule item; dumping item methods.")
                        dump_methods(item, "control", "plan", "set")
                    if call_first(item, ("setFrom",), int(from_s)) is None:
                        print("  :note - schedule item has no setFrom.")
                    call_first(item, ("setDuration",), int(duration_s))
                    master.addToSchedule(item)
                    print("  :Scheduled '%s' via %s + addToSchedule()."
                          % (plan.getName(), item_name))
                    return True
                except Exception as exc:
                    print("  :schedule-item path failed (%s); trying a direct add." % exc)

            # Fallback: a direct add method on the master.
            worked = call_first(master, ("addControlPlan", "addToControlPlans",
                                         "setControlPlan"), plan)
            if worked:
                print("  :Scheduled '%s' via master.%s()." % (plan.getName(), worked))
                return True

            print("  :ERROR - could not add the control plan to the master.  Its "
                  "scheduling API is one of the methods below:")
            dump_methods(master, "schedul", "control", "plan", "add")
            return False


        def attach_to_scenarios(model, master):
            """Set master as the master control plan on every dynamic scenario."""
            scenarios = objects_of_type(model, "GKScenario")
            if not scenarios:
                print("  :no GKScenario found - select the master by hand in the "
                      "scenario's 'Master Control Plan' field.")
                return 0

            n_set = 0
            for scen in scenarios:
                done = False
                for setter in ("setMasterControlPlan", "addMasterControlPlan",
                               "setControlPlan"):
                    fn = getattr(scen, setter, None)
                    if fn is None:
                        continue
                    try:
                        fn(master)
                        done = True
                        break
                    except Exception:
                        continue
                if done:
                    n_set += 1
                    print("    :scenario '%s' -> master '%s'"
                          % (scen.getName(), master.getName()))
                else:
                    print("    :scenario '%s' - could not set master automatically; "
                          "select '%s' in its 'Master Control Plan' field."
                          % (scen.getName(), master.getName()))
                    dump_methods(scen, "master", "control", "plan")
            return n_set



        plan = find_by_name(model, "GKControlPlan", CONTROL_PLAN_NAME)
        if plan is None:
            print("  :ERROR - control plan '%s' not found.  Run "
                  "SignalImport_Aimsun.py first (or fix CONTROL_PLAN_NAME)."
                  % CONTROL_PLAN_NAME)
            return
        print("  :Found control plan '%s' (id %s)." % (CONTROL_PLAN_NAME, plan.getId()))

        master_type = resolve_master_type(model)
        if REPLACE_EXISTING_MASTER and master_type is not None:
            existing = find_by_name(model, master_type, MASTER_NAME)
            while existing is not None:
                print("  :Removing existing master '%s' (id %s)."
                      % (MASTER_NAME, existing.getId()))
                if not delete_object(model, existing):
                    break
                nxt = find_by_name(model, master_type, MASTER_NAME)
                existing = None if nxt is existing else nxt

        master = create_master(model, MASTER_NAME)
        if master is None:
            return

        if add_plan_to_master(master, plan, SCHEDULE_FROM, SCHEDULE_DURATION):
            print("  :Master '%s' now schedules '%s' for %ds..%ds."
                  % (MASTER_NAME, CONTROL_PLAN_NAME, SCHEDULE_FROM,
                     SCHEDULE_FROM + SCHEDULE_DURATION))

        if ATTACH_TO_SCENARIOS:
            n_set = attach_to_scenarios(model, master)
            print("  :Attached master to %d scenario(s)." % n_set)

        try:
            master.setStatus(GKObject.eModified)
        except Exception:
            pass
        try:
            GKGUISystem.getGUISystem().getActiveGui().invalidateViews()
        except Exception:
            pass
        model.getCommander().addCommand(None)
        print("  :Master control plan configuration complete.")

        console.save( argv[1])
        console.close()
    else:
        console.getLog().addError( "Cannot load the network" )
        print ("cannot load network")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
