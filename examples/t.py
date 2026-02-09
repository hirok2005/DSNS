"""
Routing Visualizer Demo

This script demonstrates using the RoutingVisualizer to visualize a source-routed path
over the Starlink constellation between two ground stations.
"""

import argparse
from dsns.presets import StarlinkMultiConstellation, EARTH_RADIUS, EARTH_ROTATION_PERIOD
from dsns.visualizer import RoutingVisualizer
from dsns.message_actors import SourceRoutingDataProvider
from dsns.solvers import BmsspSolver, DijkstraSolver


def get_parser():
    parser = argparse.ArgumentParser(
        description="Demo script for the routing visualizer."
    )
    parser.add_argument(
        "--update-interval",
        help="Update interval for the routing provider (s)",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--time-scale", help="The time scale to use (s/s)", type=float, default=100.0
    )
    parser.add_argument(
        "--space-scale", help="The space scale to use (m/m)", type=float, default=1e-6
    )
    parser.add_argument(
        "--white-bg", help="Use a white background", action="store_true"
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    # 1. Create the Multiconstellation
    print("Initializing Starlink Constellation...")
    mc = StarlinkMultiConstellation()

    # 2. Select Source and Destination Ground Stations
    # The StarlinkMultiConstellation preset includes AWS ground stations by default.
    ground_ids = [s.sat_id for s in mc.satellites if s.constellation_name == "ground"]

    if len(ground_ids) < 2:
        print("Error: Not enough ground stations found in the constellation preset.")
        return

    # Pick two distant stations for an interesting path (e.g., first and last in the list)
    source_id = ground_ids[2]
    dest_id = ground_ids[7]

    print(f"Source: {source_id}")
    print(f"Destination: {dest_id}")

    # 3. Setup the Routing Provider
    # logic mimics strict source routing with a potentially stale view (update_interval)
    provider = SourceRoutingDataProvider(
        solver=DijkstraSolver, update_interval=args.update_interval
    )

    # Pre-initialize provider logic?
    # The visualizer calls provider.update(), so we just need to ensure it's ready.
    # We might want to run a quick update first to ensure the graph is built before the first frame.
    provider.update(mc, 0.0)

    # 4. Initialize Visualizer
    print("Starting Visualizer...")
    visualizer = RoutingVisualizer(
        constellation=mc,
        source_id=source_id,
        dest_id=dest_id,
        provider=provider,
        show_links=False,  # Link clutter can hide the path
        time_scale=args.time_scale,
        space_scale=args.space_scale,
        bg_color=(1.0, 1.0, 1.0) if args.white_bg else (0.0, 0.0, 0.0),
    )

    # 5. Add Earth manually (since RoutingVisualizer doesn't inherit from EarthVisualizer)
    earth_materials = (
        "assets/2k_earth_daymap.jpg",
        "assets/2k_earth_normal_map.jpg",
        "assets/2k_earth_metallic_roughness_map.jpg",
    )
    earth_color = (107 / 255, 147 / 255, 214 / 255)

    visualizer.add_planet(
        radius=EARTH_RADIUS,
        rotation_period=EARTH_ROTATION_PERIOD,
        materials=earth_materials,
        color=earth_color,
        center=mc.constellations[
            0
        ].orbital_center,  # Use the center associated with Earth
    )

    # 6. Run
    visualizer.run_simulation()
    visualizer.run_viewer()


if __name__ == "__main__":
    main()
