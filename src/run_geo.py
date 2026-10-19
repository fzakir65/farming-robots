"""Launch the farm on a real map (Mesa-Geo + Leaflet) at http://127.0.0.1:8522.

Crop areas are coloured by how ripe they are (red: mostly ripe, pink: some ripe,
green: none). Click a robot or an area for details. The base map comes from
OpenStreetMap, so it needs an internet connection; the simulation runs without one.

    python run_geo.py                    # the browser map
    python run_geo.py --export farm.geojson --steps 200   # GeoJSON for QGIS etc.
"""
import argparse

from mesa_geo.visualization import MapModule
from mesa_geo.visualization.ModularVisualization import ModularServer

from farm.geo import GeoFarmModel, export_geojson, map_centre, map_portrayal
from farm.server import StatusPanel, harvest_chart, model_params


def make_server(port=8522):
    centre = map_centre(GeoFarmModel(seed=1))
    server = ModularServer(
        GeoFarmModel,
        [MapModule(map_portrayal, view=list(centre), zoom=19, map_width=640, map_height=600),
         StatusPanel(), harvest_chart],
        "Robot-Assisted Farming on a map",
        model_params,
    )
    server.port = port
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export", help="run headless and write a GeoJSON snapshot to this file")
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()
    if args.export:
        model = GeoFarmModel(seed=1, mode="Extended")
        for _ in range(args.steps):
            model.step()
        n = export_geojson(model, args.export)
        print(f"wrote {n} features to {args.export}")
    else:
        make_server().launch()


if __name__ == "__main__":
    main()
