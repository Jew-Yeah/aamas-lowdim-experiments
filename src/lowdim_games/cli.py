"""Command-line entry points for recorded, reproducible comparisons."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.distance import pdist

from .benchmarks import (affine_dimension, layered_paraboloid, make_allocation,
                         make_synthetic, regime_path)
from .data import build_nyc_benchmark
from .experiment import METHODS, run_comparison, software_versions, write_json
from .geometry import project_convex_hull
from .plotting import plot_comparison, plot_geometry


def geometry_run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for q in args.q:
        for horizon in args.horizons:
            points = layered_paraboloid(q, horizon)
            increments, gaps = [], []
            for t in range(1, horizon):
                result = project_convex_hull(points[t], points[:t])
                if not result.success:
                    raise RuntimeError(f"Geometry projection failed at q={q}, T={horizon}, t={t+1}")
                increments.append(result.distance)
                gaps.append(result.gap)
            diameter = float(pdist(points).max())
            exponent = (q + 1) / 2
            moment = float(np.sum(np.asarray(increments) ** exponent))
            bound = float(2 * (2**q - 1) * diameter**exponent)
            record = {"q_cap": q, "realized_dimension": affine_dimension(points),
                      "horizon": horizon, "diameter": diameter, "V_T": float(sum(increments)),
                      "moment_exponent": exponent, "moment_sum": moment,
                      "moment_bound": bound, "bound_holds_numerically": bool(moment <= bound + 1e-7),
                      "max_projection_gap": max(gaps), "increments": increments}
            if not record["bound_holds_numerically"]:
                raise RuntimeError("The numerical critical-moment check failed.")
            records.append(record)
            print(f"geometry q={q} T={horizon}: V={record['V_T']:.6g}, moment={moment:.6g}/{bound:.6g}", flush=True)
    write_json(output / "geometry.json", {"software": software_versions(), "records": records,
                                          "interpretation": "Numerical finite-horizon checks, not an empirical proof of asymptotic rates."})
    plot_geometry(records, output)


def synthetic_run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for seed in args.seeds:
        instance = make_synthetic(seed=seed)
        for q in args.q:
            path, _ = regime_path(q, args.horizon, seed=seed + 100)
            name = f"synthetic_q{q}_seed{seed}_T{args.horizon}"
            print(f"Starting {name}", flush=True)
            manifest = run_comparison(instance, path, name=name, output_dir=output,
                                      config={"seed": seed, "path_seed": seed + 100, "q": q,
                                              "path": "sequentially introduced regime mixtures"})
            if args.plots:
                plot_comparison(manifest, output)
            records.extend(manifest["summaries"])
            print("  " + ", ".join(f"{x['method']}={x['delta']:.5g}" for x in manifest["summaries"]), flush=True)
    write_json(output / "synthetic_summary.json", {"software": software_versions(), "summaries": records})


def nyc_run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    records, quantization = [], []
    for profiles in args.profiles:
        data = build_nyc_benchmark(args.data, n_profiles=profiles, seed=args.seed)
        stats = data.quantized_test.summary()
        sources = {str(year): json.loads((Path(args.data) / f"nyc_forestry_hazard_daily_{year}.provenance.json").read_text(encoding="utf-8"))
                   for year in (2019, 2021, 2022)}
        np.savez_compressed(output / f"nyc_preprocessing_M{profiles}.npz",
                            profiles=data.profiles, train_counts=data.train.counts,
                            test_counts=data.test.counts, dates=np.asarray(data.test.dates),
                            projected_counts=data.quantized_test.projected,
                            profile_indices=data.quantized_test.indices,
                            errors_l1=data.quantized_test.errors_l1,
                            errors_l2=data.quantized_test.errors_l2)
        write_json(output / f"nyc_preprocessing_M{profiles}.json",
                   {"profiles": profiles, "seed": args.seed, "quantization": stats,
                    "source_csv_sha256": {year: value["csv_sha256"] for year, value in sources.items()}})
        quantization.append({"profiles": profiles, **stats})
        for ratio in args.capacity_ratios:
            instance = make_allocation(data.profiles, data.train.counts, ratio)
            name = f"nyc_M{profiles}_capacity{ratio:.2f}"
            print(f"Starting {name}, {len(data.test.counts)} held-out days", flush=True)
            manifest = run_comparison(instance, data.quantized_test.mixtures,
                                      name=name, output_dir=output, raw_demands=data.test.counts,
                                      config={"seed": args.seed, "train_years": [2019],
                                              "test_years": [2021, 2022], "profiles": profiles,
                                              "capacity_ratio": ratio, "quantization": stats,
                                              "source_csv_sha256": {year: value["csv_sha256"] for year, value in sources.items()},
                                              "metrics": "target distance in the quantized game; service metrics on original held-out counts"})
            manifest["boroughs"] = list(data.test.boroughs)
            manifest["dates"] = list(data.test.dates)
            write_json(output / f"{name}.json", manifest)
            plot_comparison(manifest, output)
            records.extend(manifest["summaries"])
            print("  " + ", ".join(f"{x['method']}={x['mean_unmet_requests']:.3f} unmet/day" for x in manifest["summaries"]), flush=True)
    write_json(output / "nyc_summary.json", {"software": software_versions(),
                                            "quantization": quantization, "summaries": records})


def cage_calibrate_run(args):
    from .cage import calibrate_cage

    if args.steps < 1 or args.train_episodes < 2 or args.test_episodes < 2:
        raise ValueError("Use positive episode steps and at least two seeds per split.")
    training = list(range(args.seed, args.seed + args.train_episodes))
    held_out = list(range(args.seed + 1_000_000,
                          args.seed + 1_000_000 + args.test_episodes))
    print(f"Calibrating CAGE 2: {args.steps} steps, "
          f"{len(training)} training and {len(held_out)} held-out episode seeds per cell", flush=True)
    calibration = calibrate_cage(args.source, episode_steps=args.steps,
                                 train_seeds=training, test_seeds=held_out,
                                 output_dir=args.output)
    print(f"Saved {len(calibration.blue_names)} defensive policies x "
          f"{len(calibration.red_names)} attacking policies to {args.output}", flush=True)


def cage_run(args):
    from .cage import load_cage_calibration
    from .policy_experiment import run_policy_comparison

    calibration = load_cage_calibration(args.calibration)
    if args.initial_red_index < 0 or args.initial_red_index >= len(calibration.red_names):
        raise ValueError("initial-red-index must identify a calibrated attacking policy")
    initial = np.eye(len(calibration.red_names))[args.initial_red_index]
    summaries = []
    for scenario in args.scenarios:
        for seed in args.seeds:
            name = f"cage2_{scenario}_seed{seed}_T{args.horizon}"
            print(f"Starting {name}", flush=True)
            manifest = run_policy_comparison(calibration, scenario=scenario,
                                             horizon=args.horizon, seed=seed,
                                             output_dir=args.output, name=name,
                                             initial_distribution=initial)
            if args.plots:
                from .policy_plotting import plot_policy_comparison
                plot_policy_comparison(manifest, args.output)
            summaries.extend(manifest["summaries"])
            print(f"Completed {name}", flush=True)
    write_json(Path(args.output) / "cage_summary.json",
               {"software": software_versions(), "summaries": summaries,
                "interpretation": "CAGE 2 simulator-calibrated policy game with separate held-out episode seeds."})


def parser():
    root = argparse.ArgumentParser(description="Reproducible finite vector-payoff game experiments")
    commands = root.add_subparsers(dest="command", required=True)
    selected = commands.add_parser("cage-selected", help="Selected scalar-aware one-switch, tuned window and Hedge")
    selected.add_argument("--calibration", default="data/cage2")
    selected.add_argument("--bank", default="data/cage2_adaptation/test50")
    selected.add_argument("--selection", default="results/cage_adaptation/selection.json")
    selected.add_argument("--protocol", default="docs/cage_adaptation_protocol.json")
    selected.add_argument("--horizon", type=int, default=512)
    selected.add_argument("--seeds", nargs="+", type=int, default=list(range(41000000, 41000050)))
    selected.add_argument("--output", default="results/runs/cage_selected")
    selected.set_defaults(run=cage_selected_run)
    synthetic = commands.add_parser("synthetic", help="Unknown-regime adaptation comparison")
    synthetic.add_argument("--q", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    synthetic.add_argument("--horizon", type=int, default=128)
    synthetic.add_argument("--seeds", nargs="+", type=int, default=[20261003, 20261004, 20261005])
    synthetic.add_argument("--output", default="results/runs/synthetic")
    synthetic.add_argument("--plots", action="store_true")
    synthetic.set_defaults(run=synthetic_run)
    geometry = commands.add_parser("geometry", help="Layered-paraboloid moment checks")
    geometry.add_argument("--q", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    geometry.add_argument("--horizons", nargs="+", type=int, default=[32, 64, 128])
    geometry.add_argument("--output", default="results/runs/geometry")
    geometry.set_defaults(run=geometry_run)
    nyc = commands.add_parser("nyc", help="Chronological NYC Forestry allocation comparison")
    nyc.add_argument("--data", default="data")
    nyc.add_argument("--profiles", nargs="+", type=int, default=[3, 6, 12])
    nyc.add_argument("--capacity-ratios", nargs="+", type=float, default=[0.6, 0.8, 1.0])
    nyc.add_argument("--seed", type=int, default=20261003)
    nyc.add_argument("--output", default="results/runs/nyc")
    nyc.set_defaults(run=nyc_run)
    cage_calibration = commands.add_parser("cage-calibrate", help="Estimate a policy game from real CAGE 2 simulator episodes")
    cage_calibration.add_argument("--source", default=".external/cage-challenge-2")
    cage_calibration.add_argument("--steps", type=int, default=50)
    cage_calibration.add_argument("--train-episodes", type=int, default=40)
    cage_calibration.add_argument("--test-episodes", type=int, default=40)
    cage_calibration.add_argument("--seed", type=int, default=20261003)
    cage_calibration.add_argument("--output", default="data/cage2")
    cage_calibration.set_defaults(run=cage_calibrate_run)
    cage = commands.add_parser("cage", help="Policy selection against changing CAGE 2 attackers")
    cage.add_argument("--calibration", default="data/cage2")
    cage.add_argument("--scenarios", nargs="+", choices=["fixed", "curriculum", "interactive"], default=["fixed", "curriculum", "interactive"])
    cage.add_argument("--horizon", type=int, default=512)
    cage.add_argument("--initial-red-index", type=int, default=1,
                      help="Initial attacking policy: 0=Sleep, 1=Meander, 2=B_line")
    cage.add_argument("--seeds", nargs="+", type=int, default=[20261003, 20261004, 20261005])
    cage.add_argument("--output", default="results/runs/cage2")
    cage.add_argument("--plots", action="store_true")
    cage.set_defaults(run=cage_run)
    return root


def cage_selected_run(args):
    from .recommended import run_selected_cage

    summary = run_selected_cage(calibration=args.calibration, bank=args.bank,
                                selection=args.selection, protocol=args.protocol,
                                horizon=args.horizon, seeds=args.seeds, output=args.output)
    print(json.dumps(summary["native_loss_means"], indent=2), flush=True)


def main(argv=None):
    args = parser().parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()
