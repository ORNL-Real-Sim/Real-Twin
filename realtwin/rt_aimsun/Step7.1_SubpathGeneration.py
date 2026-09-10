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
        import heapq

        # Subpath definitions are written by Run_DrivingBehaviorCalibration.ipynb:
        # one 'name,start_section_id,end_section_id,real_travel_time_s' per line
        SUBPATHS_CSV = "subpaths.csv"

        # Delete same-named subpaths before re-creating them
        REPLACE_EXISTING_SUBPATHS = True

        INPUT_DIR = os.path.dirname(str(model.getDocumentFileName()))
        subpaths_path = os.path.join(INPUT_DIR, SUBPATHS_CSV)
        SUBPATHS = []
        if os.path.exists(subpaths_path):
            with open(subpaths_path, "r") as fh:
                for line in fh.readlines():
                    row = [x.strip() for x in line.strip().split(",")]
                    if len(row) >= 4 and row[0]:
                        SUBPATHS.append((row[0], int(float(row[1])),
                                         int(float(row[2])), float(row[3])))
        if not SUBPATHS:
            print(f"  :ERROR - no subpath definitions found in {subpaths_path}")
            console.close()
            return


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


        def _node_turnings(node):
            try:
                ts = node.getTurnings()
            except AttributeError:
                return []
            if ts is None:
                return []
            try:
                return list(ts)
            except TypeError:
                return [ts[i] for i in range(len(ts))]


        def build_downstream_map(model):
            catalog = model.getCatalog()
            node_type = model.getType("GKNode")
            downstream = {}
            for node in (catalog.getObjectsByType(node_type) or {}).values():
                if node is None:
                    continue
                for turning in _node_turnings(node):
                    try:
                        origin = turning.getOrigin()
                        destination = turning.getDestination()
                    except Exception:
                        continue
                    if origin is None or destination is None:
                        continue
                    downstream.setdefault(origin.getId(), set()).add(destination.getId())
            return downstream


        def section_length(model, section_id, cache={}):
            if section_id not in cache:
                section = find_section(model, section_id)
                try:
                    cache[section_id] = float(section.length2D())
                except Exception:
                    cache[section_id] = 1.0
            return cache[section_id]


        def shortest_section_chain(model, downstream, start_id, end_id):
            """Dijkstra over the turning graph; returns [start..end] section ids."""
            if start_id == end_id:
                return [start_id]
            dist = {start_id: 0.0}
            prev = {}
            heap = [(0.0, start_id)]
            visited = set()
            while heap:
                d, u = heapq.heappop(heap)
                if u in visited:
                    continue
                visited.add(u)
                if u == end_id:
                    break
                for v in downstream.get(u, ()):
                    nd = d + section_length(model, v)
                    if nd < dist.get(v, float("inf")):
                        dist[v] = nd
                        prev[v] = u
                        heapq.heappush(heap, (nd, v))
            if end_id not in prev and end_id != start_id:
                return None
            chain = [end_id]
            while chain[-1] != start_id:
                chain.append(prev[chain[-1]])
            chain.reverse()
            return chain


        def get_subpath_folder(model):
            tag = "GKModel::subPaths"
            root = model.getCreateRootFolder()
            folder = root.findFolder(tag)
            if folder is None:
                try:
                    folder = root.createFolder("Subpaths", tag)
                except Exception:
                    folder = GKSystem.getSystem().createFolder(root, tag)
                    try:
                        folder.setName("Subpaths")
                    except Exception:
                        pass
            return folder


        # Remove same-named subpaths from previous runs.  Each one is first
        # DE-REGISTERED from every scenario's input data: deleting a subpath a
        # scenario still references leaves dangling pointers that crash Aimsun
        # while saving the file (the .ang is truncated mid-write).
        if REPLACE_EXISTING_SUBPATHS:
            wanted = {name for name, _, _, _ in SUBPATHS}
            commander = model.getCommander()
            scenarios = objects_of_type(model, "GKScenario")
            for sp in objects_of_type(model, "GKSubPath"):
                if str(sp.getName()) in wanted:
                    print(f"  :removing existing subpath '{sp.getName()}' (id {sp.getId()}).")
                    for scenario in scenarios:
                        try:
                            data = scenario.getInputData()
                            data.removeSubPath(sp)
                            scenario.setInputData(data)
                        except Exception as exc:
                            print(f"  :could not de-register '{sp.getName()}' from scenario '{scenario.getName()}': {exc}")
                    try:
                        commander.addCommand(sp.getDelCmd())
                    except Exception as exc:
                        print(f"  :could not delete '{sp.getName()}': {exc}")

        print("  :Building the section graph ...")
        downstream = build_downstream_map(model)
        folder = get_subpath_folder(model)

        # Layer for the new subpaths: a geo-object saved with a NULL layer is
        # invalid (Aimsun has to repair it on the next load)
        geo = model.getGeoModel()
        layer = geo.getActiveLayer(True)
        if layer is None:
            layer = next(iter(geo.getLayers()), None)

        n_ok = 0
        created_subpaths = []
        for name, start_id, end_id, real_tt in SUBPATHS:
            start = find_section(model, start_id)
            end = find_section(model, end_id)
            if start is None or end is None:
                print(f"  :{name} FAILED - section {start_id} or {end_id} not found (or not a GKSection).")
                continue
            chain = shortest_section_chain(model, downstream, int(start_id), int(end_id))
            if chain is None:
                print(f"  :{name} FAILED - no route from {start_id} to {end_id}.")
                continue

            subpath = GKSystem.getSystem().newObject("GKSubPath", model)
            subpath.setName(name)
            folder.append(subpath)
            if layer is not None:
                try:
                    geo.add(layer, subpath)
                except Exception:
                    try:
                        subpath.setLayer(layer)
                    except Exception:
                        pass
            for sec_id in chain:
                subpath.add(find_section(model, sec_id))

            try:
                correct = subpath.isCorrect()
            except Exception:
                correct = "?"
            try:
                length_m = float(subpath.length3D())
            except Exception:
                length_m = sum(section_length(model, s) for s in chain)
            print(f"  :{name} - {len(chain)} section(s) {chain}, length {length_m:.1f} m, isCorrect={correct}")
            print(f"  :SUBPATH_ID {name}={subpath.getId()} real_travel_time={real_tt}")
            created_subpaths.append(subpath)
            n_ok += 1

        # Register the subpaths for statistics collection in every scenario, so
        # their travel times are written to the MISUBPATH output table
        if created_subpaths:
            for scenario in objects_of_type(model, "GKScenario"):
                try:
                    data = scenario.getInputData()
                    data.setSubPathsStatistics(True)
                    n_added = 0
                    for sp in created_subpaths:
                        try:
                            already = data.isSubPathInStatistics(sp)
                        except Exception:
                            already = False
                        if not already:
                            data.addSubPath(sp)
                            n_added += 1
                    scenario.setInputData(data)
                    print(f"  :scenario '{scenario.getName()}' - subpath statistics enabled ({n_added} of {len(created_subpaths)} "
                          f"newly registered).")
                except Exception as exc:
                    print(f"  :could not enable subpath statistics on '{scenario.getName()}': {exc}")

        model.getCommander().addCommand(None)
        print(f"  :Subpath generation complete - {n_ok}/{len(SUBPATHS)} created.")


        console.save( argv[1])
        console.close()
    else:
        console.getLog().addError( "Cannot load the network" )
        print ("cannot load network")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
