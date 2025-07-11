import dash
from dash import html, dcc
import dash_leaflet as dl
from dash.dependencies import Input, Output, State

# Example list of existing points (lon, lat)
existing_points = [
    (-74.0059, 40.7128),
    (-74.0020, 40.7150),
    (-74.0100, 40.7100),
    # …add as many as you like
]

# Convert to a GeoJSON FeatureCollection
point_features = [{
    "type": "Feature",
    "geometry": {"type": "Point", "coordinates": pt},
    "properties": {}
} for pt in existing_points]

# Basemap options and their URLs
basemap_options = [
    {"label": "OpenStreetMap", "value": "osm"},
    {"label": "Stamen Toner", "value": "stamen_toner"},
    {"label": "CartoDB Positron", "value": "cartodb_positron"},
]

basemap_urls = {
    "osm": "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    "stamen_toner": "https://stamen-tiles-{s}.a.ssl.fastly.net/toner/{z}/{x}/{y}.png",
    "cartodb_positron": "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
}

app = dash.Dash(__name__)

app.layout = html.Div([
    # Dropdown to select basemap
    dcc.Dropdown(
        id="basemap_dropdown",
        options=basemap_options,
        value="osm",
        clearable=False,
        style={"width": "300px", "marginBottom": "10px"}
    ),
    # Map with dynamic TileLayer
    dl.Map(
        id="map",
        center=[40.7128, -74.0060],
        zoom=12,
        style={"width": "100%", "height": "500px"},
        children=[
            dl.TileLayer(id="base_layer", url=basemap_urls["osm"]),
            dl.GeoJSON(data={"type": "FeatureCollection",
                       "features": point_features}, id="points_layer"),
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
        ]
    ),
    html.Div(id="output_div", style={"marginTop": "1em"})
])

# Callback to update the TileLayer URL based on dropdown selection


@app.callback(
    Output("base_layer", "url"),
    Input("basemap_dropdown", "value")
)
def update_basemap(value):
    return basemap_urls.get(value, basemap_urls["osm"])

# Callback to display polygon coordinates


@app.callback(
    Output("output_div", "children"),
    Input("edit_control", "geojson"),
    prevent_initial_call=True
)
def display_polygon(geojson):
    if not geojson or "features" not in geojson:
        return "No polygon drawn yet."

    print("GeoJSON data:", geojson)
    formatted = ""
    coords = [feature["geometry"]["coordinates"][0] for feature in geojson["features"]]

    if not coords:
        return "No coordinates found in the polygon."

    for each_feature in geojson["features"]:
        property_type = each_feature["properties"].get("type", "Polygon")
        property_coords = each_feature["geometry"]["coordinates"]

        try:
            formatted += f"{property_type} vertices: {', '.join(f'({lon:.6f}, {lat:.6f})' for lon, lat in property_coords[0])} \n"
        except TypeError:
            formatted += f"{property_type} vertices: {', '.join(str(coord) for coord in property_coords)} \n"

    return formatted or "No coordinates found in the polygon."


if __name__ == "__main__":
    app.run(debug=True)
