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

        import math
        import os
        import zipfile

        DEBUG = 0

        def calculate_bearing(lat1, lon1, lat2, lon2):
            """Compass bearing (deg from true north, CW) from point 1 to point 2."""
            delta_lon = math.radians(lon2 - lon1)
            lat1r = math.radians(lat1)
            lat2r = math.radians(lat2)
            x = math.sin(delta_lon) * math.cos(lat2r)
            y = (math.cos(lat1r) * math.sin(lat2r)
                 - math.sin(lat1r) * math.cos(lat2r) * math.cos(delta_lon))
            initial = math.degrees(math.atan2(x, y))
            return (initial + 360.0) % 360.0


        def _points_list(section):
            """Return section.getPoints() as a list of GKPoint."""
            pts = section.getPoints()
            if pts is None:
                return []
            try:
                n = len(pts)
                return [pts[i] for i in range(n)]
            except (TypeError, AttributeError):
                pass
            try:
                n = pts.size()
                try:
                    return [pts.atIndex(i) for i in range(n)]
                except AttributeError:
                    return [pts.at(i) for i in range(n)]
            except AttributeError:
                return []


        def _to_lonlat(coord_translator, p):
            """Convert a GKPoint in the model's coordinate system to (lat, lon)."""
            deg = coord_translator.toDegrees(GKPoint(p.x, p.y, 0))
            return (deg.y, deg.x)


        def section_approach_bearing(section, ct):
            """Bearing of the last segment of `section` (entering the downstream node)."""
            pts = _points_list(section)
            if len(pts) < 2:
                return None
            lat1, lon1 = _to_lonlat(ct, pts[-2])
            lat2, lon2 = _to_lonlat(ct, pts[-1])
            return calculate_bearing(lat1, lon1, lat2, lon2)


        def section_exit_bearing(section, ct):
            """Bearing of the first segment of `section` (leaving the upstream node)."""
            pts = _points_list(section)
            if len(pts) < 2:
                return None
            lat1, lon1 = _to_lonlat(ct, pts[0])
            lat2, lon2 = _to_lonlat(ct, pts[1])
            return calculate_bearing(lat1, lon1, lat2, lon2)


        # Turn vocabulary, ordered by decreasing turn angle.
        TURN_ORDER = ["right", "thru", "left", "Uturn"]
        THRU_INDEX = TURN_ORDER.index("thru")


        def turn_angle(in_bearing, out_bearing):
            """Signed turn angle from the approach and exit bearings, in (-210, 150]."""
            if in_bearing is None or out_bearing is None:
                return None
            delta = (out_bearing - in_bearing + 540.0) % 360.0 - 180.0
            return delta - 360.0 if delta > 150.0 else delta


        def classify_angle(angle):
            """Bucket a turn angle into the coarse right/thru/left/Uturn vocabulary."""
            if angle is None:
                return None
            if angle >= 150.0 or angle <= -150.0:
                return "Uturn"
            if angle > 30.0:
                return "right"
            if angle > -30.0:
                return "thru"
            return "left"


        def classify_turn(in_bearing, out_bearing):
            """Classify a turn from the approach + exit bearings."""
            return classify_angle(turn_angle(in_bearing, out_bearing))



        def _iter_objects(catalog, gktype):
            """Yield every object of the given GKType (handles subtypes)."""
            try:
                for sub in catalog.getUsedSubTypesFromType(gktype):
                    if sub is None:
                        continue
                    for obj in sub.values():
                        if obj is not None:
                            yield obj
            except Exception:
                d = catalog.getObjectsByType(gktype)
                if d:
                    for obj in d.values():
                        if obj is not None:
                            yield obj


        def _node_turnings(node):
            """Return a list of GKTurning belonging to `node`."""
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


        def _safe_sections(call):
            """Convert a sections accessor's result to a Python list, tolerating None."""
            try:
                secs = call()
            except Exception:
                return []
            if secs is None:
                return []
            try:
                return list(secs)
            except TypeError:
                try:
                    return [secs[i] for i in range(len(secs))]
                except (TypeError, AttributeError):
                    return []


        def collect_matchup_rows():
            """Walk the network and return a list of dict rows for the matchup table."""
            geo = model.getGeoModel()
            ct = geo.getCoordinateTranslator()
            cat = model.getCatalog()

            rows = []
            seen_keys = set()
            n_total = 0
            n_kept = 0

            nodeType = model.getType("GKNode")
            for node in _iter_objects(cat, nodeType):
                n_total += 1

                entrance_sections = _safe_sections(node.getEntranceSections)
                exit_sections = _safe_sections(node.getExitSections)

                # Keep only real intersections (at least 2 exits).
                if len(exit_sections) < 2:
                    continue
                n_kept += 1

                approach_brg = {}
                for sec in entrance_sections:
                    if sec is None:
                        continue
                    try:
                        brg = section_approach_bearing(sec, ct)
                    except Exception as e:
                        if DEBUG:
                            print("  approach bearing failed for section %s: %s"
                                  % (sec.getId(), e))
                        brg = None
                    if brg is not None:
                        approach_brg[sec.getId()] = brg

                for turning in _node_turnings(node):
                    try:
                        origin = turning.getOrigin()
                        destination = turning.getDestination()
                    except Exception:
                        continue
                    if origin is None or destination is None:
                        continue

                    in_brg = approach_brg.get(origin.getId())
                    if in_brg is None:
                        in_brg = section_approach_bearing(origin, ct)
                    out_brg = section_exit_bearing(destination, ct)
                    angle = turn_angle(in_brg, out_brg)
                    turn_dir = classify_angle(angle)

                    key = (node.getId(), origin.getId(), destination.getId(), turn_dir)
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)

                    rows.append({
                        "JunctionID_Aimsun": node.getId(),
                        "Bearing": round(in_brg, 2) if in_brg is not None else None,
                        "Numbering": int(round(in_brg / 10.0)) if in_brg is not None else None,
                        "FromRoadID_Aimsun": origin.getId(),
                        "ToRoadID_Aimsun": destination.getId(),
                        "Turn": turn_dir,
                        "TurnAngle": angle,
                    })

            print("Scanned %d nodes; %d are intersections (>=2 exits); produced %d turn rows."
                  % (n_total, n_kept, len(rows)))
            return rows


        def sort_rows(rows):
            """Sort rows by junction id, bearing (shifted by 337.5), then turn angle descending."""
            def key_fn(r):
                b = r["Bearing"] if r["Bearing"] is not None else 0.0
                shifted = (b - 337.5) % 360.0
                angle = r.get("TurnAngle")
                rank = -angle if angle is not None else 999.0
                return (str(r["JunctionID_Aimsun"]), shifted, rank)

            rows.sort(key=key_fn)
            return rows


        def _label_options(n):
            """All strictly increasing length-``n`` label-index tuples over TURN_ORDER."""
            if n == 0:
                return []
            options = [(i,) for i in range(len(TURN_ORDER))]
            for _ in range(n - 1):
                options = [opt + (j,)
                           for opt in options
                           for j in range(opt[-1] + 1, len(TURN_ORDER))]
            return options


        def _is_toward_thru(assigned, geometric):
            """True if moving from ``geometric`` to ``assigned`` steps toward 'thru'."""
            if assigned == geometric or assigned == THRU_INDEX:
                return True
            same_side = (assigned - THRU_INDEX) * (geometric - THRU_INDEX) > 0
            return same_side and abs(assigned - THRU_INDEX) < abs(geometric - THRU_INDEX)


        def _assignment_cost(option, geometric, angles):
            """Cost of one candidate label assignment."""
            centres = {0: 90.0, 1: 0.0, 2: -90.0, 3: -180.0}
            cost = 0.0
            drift = 0.0
            for assigned, geo, angle in zip(option, geometric, angles):
                step = abs(assigned - geo)
                cost += step if _is_toward_thru(assigned, geo) else step * 100.0
                if angle is not None:
                    drift += abs(angle - centres[assigned])
            return (cost, drift, option)


        def enforce_unique_turns(rows):
            """Relabel movements so no approach carries the same ``Turn`` twice; returns the count relabelled."""
            groups = {}
            for r in rows:
                groups.setdefault((r["JunctionID_Aimsun"], r["FromRoadID_Aimsun"]), []).append(r)

            relabelled = 0
            failures = []
            for (junction_id, from_id), group in sorted(groups.items(), key=lambda kv: str(kv[0])):
                if len(group) < 2:
                    continue

                group.sort(key=lambda r: -(r["TurnAngle"] if r["TurnAngle"] is not None else -999.0))
                geometric = [TURN_ORDER.index(r["Turn"]) if r["Turn"] in TURN_ORDER else THRU_INDEX
                             for r in group]
                if len(set(geometric)) == len(geometric):
                    continue

                options = _label_options(len(group))
                if not options:
                    failures.append((junction_id, from_id, group,
                                     "%d movements, only %d turn labels available"
                                     % (len(group), len(TURN_ORDER))))
                    continue

                angles = [r["TurnAngle"] for r in group]
                best = min(options, key=lambda o: _assignment_cost(o, geometric, angles))

                for row, geo, assigned in zip(group, geometric, best):
                    if assigned == geo:
                        continue
                    angle_txt = "n/a" if row["TurnAngle"] is None else "%+.1f deg" % row["TurnAngle"]
                    print("  Relabelled junction %s approach %s -> %s (%s): %s -> %s"
                          % (junction_id, from_id, row["ToRoadID_Aimsun"], angle_txt,
                             TURN_ORDER[geo], TURN_ORDER[assigned]))
                    row["Turn"] = TURN_ORDER[assigned]
                    relabelled += 1

            for junction_id, from_id, group, why in failures:
                print("ERROR: junction %s approach %s cannot be given unique turn labels "
                      "(%s).  Movements:" % (junction_id, from_id, why))
                for row in group:
                    angle_txt = "n/a" if row["TurnAngle"] is None else "%+.1f deg" % row["TurnAngle"]
                    print("         -> %s  %s  %s" % (row["ToRoadID_Aimsun"], angle_txt, row["Turn"]))
                print("       Duplicate turn codes remain; the demand import will keep only "
                      "the first of each and silently drop the rest.")

            return relabelled



        NETWORK_COLS = ["JunctionID_Aimsun", "Bearing", "Numbering",
                        "FromRoadID_Aimsun", "ToRoadID_Aimsun", "Turn"]
        DEMAND_COLS = ["File_GridSmart", "Date_GridSmart",
                       "IntersectionName_GridSmart", "Turn_GridSmart"]
        SIGNAL_COLS = ["File_Synchro", "IntersectionID_Synchro", "Turn_Synchro"]
        OTHER_COLS = ["Need calibration?"]

        ALL_HEADERS = NETWORK_COLS + DEMAND_COLS + SIGNAL_COLS + OTHER_COLS
        COL_WIDTHS = [20, 15, 15, 25, 25, 15, 20, 20, 30, 20, 20, 25, 20, 20]


        def _col_letter(n):
            """1-based column index -> 'A', 'B', ..., 'AA', ..."""
            result = ""
            while n > 0:
                n, r = divmod(n - 1, 26)
                result = chr(65 + r) + result
            return result


        def _xml_escape(s):
            return (s.replace("&", "&amp;")
                     .replace("<", "&lt;")
                     .replace(">", "&gt;")
                     .replace('"', "&quot;"))


        def _cell_xml(row, col, value, style_id=1):
            """XML for one <c> cell."""
            if value is None or value == "":
                return ""
            ref = "%s%d" % (_col_letter(col), row)
            if isinstance(value, bool):
                return ('<c r="%s" s="%d" t="inlineStr"><is><t>%s</t></is></c>'
                        % (ref, style_id, _xml_escape(str(value))))
            if isinstance(value, (int, float)):
                if isinstance(value, float) and value.is_integer():
                    v = str(int(value))
                else:
                    v = repr(value) if isinstance(value, float) else str(value)
                return '<c r="%s" s="%d"><v>%s</v></c>' % (ref, style_id, v)
            return ('<c r="%s" s="%d" t="inlineStr"><is><t>%s</t></is></c>'
                    % (ref, style_id, _xml_escape(str(value))))


        def _build_sheet_xml(rows):
            parts = []

            # Column widths
            parts.append("<cols>")
            for i, w in enumerate(COL_WIDTHS):
                parts.append('<col min="%d" max="%d" width="%d" customWidth="1"/>'
                             % (i + 1, i + 1, w))
            parts.append("</cols>")

            parts.append("<sheetData>")

            # Row 1: group headers (Network / Demand / Signal / "")
            row1_vals = (["Network"] * len(NETWORK_COLS)
                         + ["Demand"] * len(DEMAND_COLS)
                         + ["Signal"] * len(SIGNAL_COLS)
                         + [""] * len(OTHER_COLS))
            parts.append('<row r="1">')
            for c, v in enumerate(row1_vals):
                parts.append(_cell_xml(1, c + 1, v))
            parts.append("</row>")

            # Row 2: column headers
            parts.append('<row r="2">')
            for c, v in enumerate(ALL_HEADERS):
                parts.append(_cell_xml(2, c + 1, v))
            parts.append("</row>")

            # Row 3+: data
            for i, r in enumerate(rows):
                row_idx = i + 3
                values = [r["JunctionID_Aimsun"], r["Bearing"], r["Numbering"],
                          r["FromRoadID_Aimsun"], r["ToRoadID_Aimsun"], r["Turn"],
                          None, None, None, None,
                          None, None, None,
                          None]
                parts.append('<row r="%d">' % row_idx)
                for c, v in enumerate(values):
                    parts.append(_cell_xml(row_idx, c + 1, v))
                parts.append("</row>")

            parts.append("</sheetData>")

            # Merged cells
            merges = []
            merges.append("A1:F1")  # Network group header
            merges.append("G1:J1")  # Demand group header
            merges.append("K1:M1")  # Signal group header

            # Per-junction merges in cols A, G, H, I, L, N
            n = len(rows)
            if n > 0:
                current_start = 3
                for i in range(n):
                    data_row = i + 3
                    is_last = (i == n - 1)
                    same_next = (not is_last
                                 and rows[i]["JunctionID_Aimsun"]
                                     == rows[i + 1]["JunctionID_Aimsun"])
                    if not same_next:
                        if current_start < data_row:
                            for col in ("A", "G", "H", "I", "L", "N"):
                                merges.append("%s%d:%s%d"
                                              % (col, current_start, col, data_row))
                        current_start = data_row + 1
                # File_Synchro (col K) spans all data rows
                merges.append("K3:K%d" % (n + 2))

            if merges:
                parts.append('<mergeCells count="%d">' % len(merges))
                for m in merges:
                    parts.append('<mergeCell ref="%s"/>' % m)
                parts.append("</mergeCells>")

            body = "".join(parts)
            return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                    '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
                    + body +
                    '</worksheet>')


        _CONTENT_TYPES_XML = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            '<Override PartName="/xl/worksheets/sheet1.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            '<Override PartName="/xl/styles.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
            '</Types>'
        )

        _ROOT_RELS_XML = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
            'Target="xl/workbook.xml"/>'
            '</Relationships>'
        )

        _WORKBOOK_XML = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            '<sheets><sheet name="Sheet" sheetId="1" r:id="rId1"/></sheets>'
            '</workbook>'
        )

        _WORKBOOK_RELS_XML = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            'Target="worksheets/sheet1.xml"/>'
            '<Relationship Id="rId2" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
            'Target="styles.xml"/>'
            '</Relationships>'
        )

        # Cell styles: 0 = plain, 1 = centered
        _STYLES_XML = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            '<fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>'
            '<fills count="2">'
            '<fill><patternFill patternType="none"/></fill>'
            '<fill><patternFill patternType="gray125"/></fill>'
            '</fills>'
            '<borders count="1"><border/></borders>'
            '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
            '<cellXfs count="2">'
            '<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
            '<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0" applyAlignment="1">'
            '<alignment horizontal="center" vertical="center"/>'
            '</xf>'
            '</cellXfs>'
            '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
            '</styleSheet>'
        )


        def write_xlsx(rows, path):
            """Write the matchup table as a valid .xlsx using only stdlib (zipfile)."""
            sheet_xml = _build_sheet_xml(rows)
            with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
                z.writestr("[Content_Types].xml", _CONTENT_TYPES_XML)
                z.writestr("_rels/.rels", _ROOT_RELS_XML)
                z.writestr("xl/workbook.xml", _WORKBOOK_XML)
                z.writestr("xl/_rels/workbook.xml.rels", _WORKBOOK_RELS_XML)
                z.writestr("xl/styles.xml", _STYLES_XML)
                z.writestr("xl/worksheets/sheet1.xml", sheet_xml)

        def _default_output_dir():
            try:
                doc = str(model.getDocumentFileName())
            except Exception:
                doc = ""
            folder = os.path.dirname(doc) if doc else ""
            if not folder or not os.path.isdir(folder):
                folder = os.path.expanduser("~")
            return folder


        print("Collecting matchup data from the Aimsun network ...")
        rows = collect_matchup_rows()

        relabelled = enforce_unique_turns(rows)
        if relabelled:
            print("Relabelled %d movement(s) so no approach repeats a turn." % relabelled)

        sort_rows(rows)

        folder = _default_output_dir()
        out_path = os.path.join(folder, "MatchupTable.xlsx")
        write_xlsx(rows, out_path)
        print("Matchup table saved to: %s" % out_path)

        console.save( argv[1])
        console.close()
    else:
        console.getLog().addError( "Cannot load the network" )
        print ("cannot load network")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
