"""
EchoLoop Embodied Reaction MVP — simulation runner.

Usage (from inside experiments/embodied_reaction_mvp/):

    python run_simulation.py --scenario speech_then_silence
    python run_simulation.py --scenario all
    python run_simulation.py --scenario all --verbose
"""

import argparse
import csv
import json
import os
import time

from echoloop import run_simulation
from scenarios import SCENARIOS, get_scenario
from plotting import plot_inputs, plot_internals, plot_event_raster, plot_iei

LOGS_DIR = os.path.join("outputs", "logs")
PLOTS_DIR = os.path.join("outputs", "plots")
DT = 1.0 / 60.0


def run_and_save(scenario_name: str, verbose: bool = False) -> None:
    print(f"\n[{scenario_name}]")

    signals = get_scenario(scenario_name, dt=DT)
    records, event_log = run_simulation(signals, dt=DT)

    duration = records[-1]["time"] if records else 0.0
    print(f"  steps: {len(records)},  duration: {duration:.1f} s")
    print(f"  events fired: {len(event_log)}")

    if event_log:
        counts: dict = {}
        for ev in event_log:
            counts[ev["name"]] = counts.get(ev["name"], 0) + 1
        for name in sorted(counts):
            print(f"    {name}: {counts[name]}")

    # CSV — per-timestep values
    os.makedirs(LOGS_DIR, exist_ok=True)
    csv_path = os.path.join(LOGS_DIR, f"{scenario_name}.csv")
    if records:
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
        print(f"  log  → {csv_path}")

    # JSON — event log
    json_path = os.path.join(LOGS_DIR, f"{scenario_name}_events.json")
    with open(json_path, "w") as fh:
        json.dump(event_log, fh, indent=2)
    if verbose:
        print(f"  events → {json_path}")

    # Plots
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_inputs(records, scenario_name, PLOTS_DIR)
    plot_internals(records, scenario_name, PLOTS_DIR)
    plot_event_raster(records, event_log, scenario_name, PLOTS_DIR)
    plot_iei(event_log, scenario_name, PLOTS_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EchoLoop Embodied Reaction MVP — run a simulation scenario."
    )
    parser.add_argument(
        "--scenario",
        required=True,
        choices=list(SCENARIOS) + ["all"],
        metavar="SCENARIO",
        help=(
            "Scenario name, or 'all' to run every scenario.  "
            f"Available: {', '.join(SCENARIOS)}"
        ),
    )
    parser.add_argument("--verbose", action="store_true", help="Print extra paths.")
    args = parser.parse_args()

    t0 = time.time()
    if args.scenario == "all":
        for name in SCENARIOS:
            run_and_save(name, verbose=args.verbose)
    else:
        run_and_save(args.scenario, verbose=args.verbose)

    print(f"\ndone in {time.time() - t0:.2f} s")


if __name__ == "__main__":
    main()
