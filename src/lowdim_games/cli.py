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


def parser():
    root = argparse.ArgumentParser(description="Reproducible finite vector-payoff game experiments")
    commands = root.add_subparsers(dest="command", required=True)
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
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()
