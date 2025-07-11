import dash
from dash import html, dcc
import dash_leaflet as dl
from dash.dependencies import Input, Output, State
from shapely.geometry import Point, Polygon

# --- your existing data ---
existing_points = [
    (-74.0059, 40.7128),
    (-74.0020, 40.7150),
    (-74.0100, 40.7100),
    # …add as many as you like
]
point_features = [{
    "type": "Feature",
    "geometry": {"type": "Point", "coordinates": pt},
    "properties": {}
} for pt in existing_points]

basemap_options = [
    {"label": "OpenStreetMap", "value": "osm"},
    {"label": "Stamen Toner",   "value": "stamen_toner"},
    {"label": "CartoDB Positron", "value": "cartodb_positron"},
]
basemap_urls = {
    "osm":               "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    "stamen_toner":      "https://stamen-tiles-{s}.a.ssl.fastly.net/toner/{z}/{x}/{y}.png",
    "cartodb_positron":  "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
}

app = dash.Dash(__name__)

app.layout = html.Div([
    # dropdown to switch basemap
    dcc.Dropdown(
        id="basemap_dropdown",
        options=basemap_options,
        value="osm",
        clearable=False,
        style={"width": "300px", "marginBottom": "10px"}
    ),
    # the map itself
    dl.Map(
        id="map",
        center=[40.7128, -74.0060],
        zoom=12,
        style={"width": "100%", "height": "500px"},
        children=[
            dl.TileLayer(id="base_layer", url=basemap_urls["osm"]),
            # all your original points
            dl.GeoJSON(data={"type": "FeatureCollection", "features": point_features},
                       id="points_layer"),
            # layer for the user to draw on
            dl.FeatureGroup(
                id="editable_layer",
                children=[
                    dl.EditControl(
                        id="edit_control",
                        draw={
                            "polyline": False,
                            "circle": True,
                            "circlemarker": False,
                            "rectangle": True,
                            "marker": False,
                            "polygon": True
                        },
                        edit={"featureGroup": "editable_layer"},
                    )
                ]
            ),
            # NEW: layer where we'll add red circle-markers
            dl.LayerGroup(id="selected_points_layer"),
        ]
    ),
    html.Div(id="output_div", style={"marginTop": "1em"})
])

# switch basemap URL


@app.callback(
    Output("base_layer", "url"),
    Input("basemap_dropdown", "value")
)
def update_basemap(value):
    return basemap_urls.get(value, basemap_urls["osm"])


# when the user draws something, compute which existing points lie inside it
@app.callback(
    [
        Output("output_div", "children"),
        Output("selected_points_layer", "children")
    ],
    Input("edit_control", "geojson"),
    prevent_initial_call=True
)
def select_points_within(geojson):
    if not geojson or "features" not in geojson:
        return "No shape drawn yet.", []

    # build one (or more) Shapely polygons from what was drawn
    regions = []
    for feat in geojson["features"]:
        geom_type = feat["geometry"]["type"]
        coords = feat["geometry"]["coordinates"]
        if geom_type == "Polygon":
            # a rectangle, polygon, or drawn‐circle all come back as a Polygon
            regions.append(Polygon(coords[0]))
        # you could add elifs here if you want to support other geo‐types

    if not regions:
        return "No polygonal region detected.", []

    # test each existing point
    inside = []
    for lon, lat in existing_points:
        pt = Point(lon, lat)
        if any(region.contains(pt) for region in regions):
            inside.append((lon, lat))

    # format a human‐readable list
    if inside:
        msg = "Points inside region:\n" + \
            "\n".join(f"• ({lon:.6f}, {lat:.6f})" for lon, lat in inside)
    else:
        msg = "No existing points fall inside the drawn region.  \n\n"

    # build red CircleMarker for each selected point
    markers = [
        dl.CircleMarker(center=[lat, lon], radius=8,
                        color="red", fillOpacity=0.6)
        for lon, lat in inside
    ]

    print("GeoJSON data:", geojson)
    # coords = [feature["geometry"]["coordinates"][0] for feature in geojson["features"]]

    # if not coords:
    #     return "No coordinates found in the polygon."

    for each_feature in geojson["features"]:
        property_type = each_feature["properties"].get("type", "Polygon")
        property_coords = each_feature["geometry"]["coordinates"]

        try:
            msg += f"{property_type} vertices: {', '.join(f'({lon:.6f}, {lat:.6f})' for lon,
                                                          lat in property_coords[0])} \n\n"
        except TypeError:
            msg += f"{property_type} vertices: {', '.join(str(coord) for coord in property_coords)} \n\n"

    return msg, markers


if __name__ == "__main__":
    app.run(debug=True)
