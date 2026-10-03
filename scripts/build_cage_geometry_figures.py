"""Full-target calibrated error of the three locked CAGE policy methods.

Prespecified prefix checkpoints retain every primary path. A separate horizon
sweep uses new common paths and fixed selected settings, without simulator
collection or parameter selection. Numerical distances are model diagnostics.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
from pathlib import Path
import platform
import sys
import tempfile
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from lowdim_games.cage import load_cage_calibration
from lowdim_games.experiment import tensor_hash
from lowdim_games.game import FiniteGame
from lowdim_games.policy_experiment import curriculum_path, phase_boundaries
from scripts.collect_cage_bank import _atomic_npz
from scripts.run_cage_adaptation_study import _new_learner

METHODS = ("selected_scalar_one_switch", "selected_window", "selected_hedge")
CONFIGURATIONS = (
    {"name": METHODS[0], "kind": "scalar", "window": 16, "rho": 0.25},
    {"name": METHODS[1], "kind": "window", "window": 16},
    {"name": METHODS[2], "kind": "hedge", "eta_multiplier": 4},
)
MODULES = (
    "src/lowdim_games/cage_adaptations.py", "src/lowdim_games/learners.py",
    "src/lowdim_games/policy_experiment.py", "src/lowdim_games/game.py",
    "src/lowdim_games/geometry.py", "scripts/run_cage_adaptation_study.py",
)
FIELDS = ("lower_distance", "distance", "gap", "alpha", "success",
          "feasibility_error", "iterations", "affine_dimension")
COLORS = ("#2673b8", "#d68820", "#3a946c")
LABELS = {
    "en": ("Our one-switch", "Window (W=16)", "Hedge (eta x4)"),
    "ru": ("Наш one-switch", "Окно (W=16)", "Hedge (eta x4)"),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".geometry-", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def projection_row(game, average, path, *, tol=1e-11):
    """Project over all response cells of the whole hull, including mixtures."""
    hull = np.unique(np.asarray(path, dtype=float), axis=0)
    result = game.target_projection(average, hull, tol=tol)
    if not result.success:
        raise RuntimeError("Full-target projection failed its numerical certificate.")
    if not (np.isfinite(result.distance) and np.isfinite(result.gap)
            and result.lower_distance <= result.distance + 1e-14):
        raise RuntimeError("Invalid numerical distance interval.")
    return np.array([result.lower_distance, result.distance, result.gap,
                     result.alpha, float(result.success), result.feasibility_error,
                     result.iterations, np.linalg.matrix_rank(hull-hull[0], tol=1e-12)])


def checkpoints(path, every=16):
    """Powers of two, regular grid, phase changes and first exact mode appearances."""
    horizon = len(path)
    values = {1, horizon}
    values.update(range(every, horizon+1, every))
    value = 1
    while value <= horizon:
        values.add(value)
        value *= 2
    for boundary in phase_boundaries(horizon)[1:-1]:
        values.update((boundary, boundary+1))
    seen = set()
    for index, point in enumerate(path, 1):
        key = tuple(point)
        if key not in seen:
            values.add(index)
            seen.add(key)
    return np.asarray(sorted(values), dtype=np.int64)


def freeze_protocol(output, calibration, primary_dir):
    """Record all conditions and source hashes before any new trajectory."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/"protocol.json").exists():
        return verify_protocol(output)
    original = load_cage_calibration(calibration)
    meta = read(Path(primary_dir)/"meta.json")
    expected = list(range(41000000, 41000050))
    if meta["status"] != "complete" or meta["path_seeds"] != expected or meta["horizon"] != 512:
        raise ValueError("Every locked primary path is required.")
    selection = read(REPO/"results/cage_adaptation/selection.json")
    chosen = selection["selected"]
    if (chosen["scalar"]["window"] != 16 or chosen["scalar"]["rho"] != .25
            or chosen["window"]["window"] != 16 or chosen["hedge"]["eta_multiplier"] != 4):
        raise ValueError("Locked selection changed.")
    immutable = {
        "schema_version": 1,
        "scope": "Post-study descriptive calibrated geometry, without retuning",
        "global_figure_protocol_sha256": sha(REPO/"docs/cage_figures_protocol.json"),
        "calibration_npz_sha256": original.provenance["calibration_npz_sha256"],
        "tensor_sha256": tensor_hash(original.tensor),
        "selection_sha256": sha(REPO/"results/cage_adaptation/selection.json"),
        "primary_meta_sha256": sha(Path(primary_dir)/"meta.json"),
        "primary_checkpoint_sha256": meta["path_sha256"],
        "primary_inputs_sha256": sha(output/"primary_inputs.npz"),
        "primary_sources_sha256": sha(output/"primary_sources.json"),
        "implementation_sha256": {name:sha(REPO/name) for name in MODULES},
        "figure_runner_sha256": sha(__file__),
        "configurations": list(CONFIGURATIONS), "methods": list(METHODS),
        "initial_red_index": 1, "phase_fractions": [.25,.5,.25], "attacker_exploration": .1,
        "prefix": {"horizon":512, "seeds":expected, "checkpoint_every":16,
                   "extra_checkpoints":"powers of two, phase boundaries and successors, first occurrence of every exact mode",
                   "targets":["moving S(Q_t)", "fixed final S(Q_T), retrospective only"]},
        "terminal_sweep": {"horizons":[64,128,256,512,1024,2048],
                           "seeds":list(range(45000000,45000020)),
                           "new_path_count":120, "method_trajectory_count":360,
                           "scenario":"horizon-scaled exogenous curriculum learned against uniform reference",
                           "hedge_eta":"4 sqrt(8 log(K)/T) / global calibrated scalar loss range",
                           "attacker_eta":"original sqrt(8 log(M)/T) / global calibrated scalar loss range",
                           "not_nested_prefixes":True},
        "geometry":{"target":"full closed convex response target over all realized-hull mixtures",
                    "payoffs":"frozen normalized training tensor, not held-out simulator costs",
                    "projection_tol":1e-11, "fields":list(FIELDS),
                    "numerical_check":"floating-point primal/full-target-support gap, not exact arithmetic",
                    "near_floor_threshold":1e-10,
                    "zero_handling":"No epsilon added; floor cases counted in separate panel",
                    "slope":"OLS of all horizon medians only if every lower and upper median exceeds numerical floor"},
        "inference":{"retuning":False, "native_simulator_episodes_added":0,
                     "bands":"10-90% empirical path quantiles conditional on calibration",
                     "theory_reference":"T^(-1/2) display-scaled guide for dimension<=2"},
    }
    path = output/"protocol.json"
    if path.exists():
        previous = read(path)
        for key,value in immutable.items():
            if previous.get(key) != value:
                raise ValueError(f"Frozen geometry protocol changed: {key}")
        return previous
    immutable["created_utc"] = datetime.now(timezone.utc).isoformat()
    write(path,immutable)
    return immutable


def verify_protocol(output):
    protocol = read(Path(output)/"protocol.json")
    if protocol["figure_runner_sha256"] != sha(__file__):
        # A documented post-run packaging repair can preserve scientific input
        # identities while fixing byte-preserving public replay. No algorithm,
        # seed, parameter, projection or trajectory computation is amended.
        repair_path = Path(output)/"serialization_repair.json"
        if not repair_path.exists():
            raise ValueError("Runner changed after protocol freeze.")
        repair = read(repair_path)
        if (repair["frozen_runner_sha256"] != protocol["figure_runner_sha256"]
                or repair["repaired_runner_sha256"] != sha(__file__)
                or repair["scope"] != "preserve frozen source JSON bytes during repeated replay"):
            raise ValueError("Runner differs from documented serialization repair.")
    for name,digest in protocol["implementation_sha256"].items():
        if sha(REPO/name) != digest:
            raise ValueError(f"Frozen core changed: {name}")
    if protocol["global_figure_protocol_sha256"] != sha(REPO/"docs/cage_figures_protocol.json"):
        raise ValueError("Global figure protocol changed.")
    if protocol["selection_sha256"] != sha(REPO/"results/cage_adaptation/selection.json"):
        raise ValueError("Locked selection changed.")
    return protocol


def load_primary_inputs(output, primary_dir, protocol, original):
    """Replay public traces if ignored original checkpoints are unavailable."""
    output,primary_dir = Path(output),Path(primary_dir)
    if not all((primary_dir/"paths"/f"seed_{seed}.npz").exists()
               for seed in protocol["prefix"]["seeds"]):
        if sha(output/"primary_inputs.npz") != protocol["primary_inputs_sha256"]:
            raise ValueError("Public primary input checksum mismatch.")
        if sha(output/"primary_sources.json") != protocol["primary_sources_sha256"]:
            raise ValueError("Public primary sources checksum mismatch.")
        sources = read(output/"primary_sources.json")
        if sources["source_checkpoint_sha256"] != protocol["primary_checkpoint_sha256"]:
            raise ValueError("Public primary source provenance changed.")
        with np.load(output/"primary_inputs.npz",allow_pickle=False) as data:
            if (data["path_seeds"].tolist() != protocol["prefix"]["seeds"]
                    or data["methods"].tolist() != list(METHODS)
                    or not np.array_equal(data["tensor"],original.tensor)
                    or not np.array_equal(data["weights"],original.weights)):
                raise ValueError("Public primary trace menu or calibration changed.")
            return data["opponent_actions"].copy(),data["actions"].copy(),sources["source_checkpoint_sha256"]
    paths,actions,hashes = [],[],{}
    for seed in protocol["prefix"]["seeds"]:
        source = primary_dir/"paths"/f"seed_{seed}.npz"
        digest = sha(source)
        if digest != protocol["primary_checkpoint_sha256"][source.name]:
            raise ValueError(f"Primary checkpoint changed: {source.name}")
        hashes[source.name] = digest
        with np.load(source,allow_pickle=False) as data:
            names = data["methods"].tolist()
            opponent = data["opponent_actions"].copy()
            selected = np.stack([data[f"method_{names.index(name)}_actions"] for name in METHODS])
            for index,name in enumerate(METHODS):
                replay = np.einsum("tk,kmd,tm->td",selected[index],original.tensor,opponent)
                if not np.allclose(replay,data[f"method_{names.index(name)}_train_payoffs"],atol=1e-13,rtol=0):
                    raise ValueError("Saved payoff does not replay.")
        paths.append(opponent)
        actions.append(selected)
    return np.stack(paths),np.stack(actions),hashes


def prefix_geometry(output, primary_dir, calibration):
    output = Path(output)
    protocol = verify_protocol(output)
    original = load_cage_calibration(calibration)
    if tensor_hash(original.tensor) != protocol["tensor_sha256"]:
        raise ValueError("Frozen tensor changed.")
    paths,actions,hashes = load_primary_inputs(output,primary_dir,protocol,original)
    grid = np.unique(np.concatenate([checkpoints(path) for path in paths]))
    game = FiniteGame(original.tensor,original.weights)
    moving = np.empty((len(paths),3,len(grid),8))
    fixed = np.empty_like(moving)
    for index,(path,selected) in enumerate(zip(paths,actions)):
        for method,action in enumerate(selected):
            payoffs = np.einsum("tk,kmd,tm->td",action,game.tensor,path)
            sums = payoffs.cumsum(axis=0)
            for count,t in enumerate(grid):
                query = sums[t-1]/t
                moving[index,method,count] = projection_row(game,query,path[:t],tol=1e-11)
                fixed[index,method,count] = projection_row(game,query,path,tol=1e-11)
        print(f"Full-target primary geometry {index+1}/{len(paths)}",flush=True)
    # Identical public shared traces are retained for the dynamic-figure builder.
    shared = output/"primary_inputs.npz"
    if shared.exists():
        with np.load(shared,allow_pickle=False) as data:
            if not (np.array_equal(data["actions"],actions) and np.array_equal(data["opponent_actions"],paths)
                    and np.array_equal(data["tensor"],original.tensor)):
                raise ValueError("Previously published shared traces differ.")
    else:
        _atomic_npz(shared,tensor=original.tensor,weights=original.weights,methods=np.asarray(METHODS),
                    path_seeds=np.asarray(protocol["prefix"]["seeds"]),actions=actions,opponent_actions=paths,
                    normalization_scale=original.normalization_scale)
    _atomic_npz(output/"prefix_geometry.npz",rows=moving,fixed_target_rows=fixed,checkpoints=grid,
                fields=np.asarray(FIELDS),methods=np.asarray(METHODS),seeds=np.asarray(protocol["prefix"]["seeds"]),
                protocol_sha256=np.asarray(sha(output/"protocol.json")))
    sources_path = output/"primary_sources.json"
    if sources_path.exists():
        if sha(sources_path) != protocol["primary_sources_sha256"]:
            raise ValueError("Frozen source JSON bytes changed.")
        if read(sources_path)["source_checkpoint_sha256"] != hashes:
            raise ValueError("Primary source provenance changed.")
    else:
        raise ValueError("Frozen public source JSON is missing.")


def sweep_job(job):
    tensor,weights,horizon,seed,fixed,target = job
    game = FiniteGame(tensor,weights)
    initial = np.eye(game.M)[fixed["initial_red_index"]]
    path = curriculum_path(tensor,weights,horizon,seed,initial_distribution=initial,
                           phase_fractions=fixed["phase_fractions"],exploration=fixed["attacker_exploration"])
    selected,summaries,rows = [],[],[]
    for config in CONFIGURATIONS:
        learner = _new_learner(config,game,horizon,initial)
        actions = []
        beta,fallbacks = 0.,0
        for ell in path:
            actions.append(learner.choose())
            record = learner.observe(ell)
            beta = max(beta,record.get("beta_t",0) or 0)
            fallbacks += int(record.get("scalar_oracle_fallback",False))
        actions = np.asarray(actions)
        selected.append(actions)
        average = np.einsum("tk,kmd,tm->d",actions,tensor,path)/horizon
        rows.append(projection_row(game,average,path,tol=1e-11))
        summaries.append({"method":config["name"],"action_sha256":tensor_hash(actions),
                          "switch_round":getattr(learner,"switch_round",None),
                          "max_beta":float(beta),"scalar_fallback_count":fallbacks,
                          "learning_rate":getattr(learner,"learning_rate",None)})
    _atomic_npz(target,tensor_sha256=np.asarray(tensor_hash(tensor)),
                protocol_sha256=np.asarray(fixed["protocol_sha256"]),
                seed=np.asarray(seed),horizon=np.asarray(horizon),methods=np.asarray(METHODS),
                opponent_actions=path,actions=np.stack(selected),geometry=np.stack(rows),
                summaries_json=np.asarray(json.dumps(summaries,allow_nan=False)))
    return horizon,seed


def verify_sweep_checkpoint(path,tensor,protocol_digest,horizon,seed):
    with np.load(path,allow_pickle=False) as data:
        if (int(data["seed"]) != seed or int(data["horizon"]) != horizon
                or str(data["protocol_sha256"]) != protocol_digest
                or str(data["tensor_sha256"]) != tensor_hash(tensor)
                or data["methods"].tolist() != list(METHODS)):
            raise ValueError("Sweep checkpoint changed frozen conditions.")
        actions,opponent = data["actions"],data["opponent_actions"]
        if (actions.shape != (3,horizon,tensor.shape[0]) or opponent.shape != (horizon,tensor.shape[1])
                or not np.isfinite(actions).all() or np.min(actions)<-1e-12
                or not np.allclose(actions.sum(axis=-1),1,atol=1e-10,rtol=0)
                or not np.allclose(opponent.sum(axis=-1),1,atol=1e-12,rtol=0)
                or not np.isfinite(data["geometry"]).all() or not np.all(data["geometry"][:,4] == 1)):
            raise ValueError("Invalid sweep arrays.")
        summaries = json.loads(str(data["summaries_json"]))
        for config,action,summary in zip(CONFIGURATIONS,actions,summaries):
            if summary["method"] != config["name"] or summary["action_sha256"] != tensor_hash(action):
                raise ValueError("Sweep action checksum mismatch.")
        return {key:data[key].copy() for key in data.files}


def terminal_sweep(output,calibration,run_dir,workers):
    output,run_dir = Path(output),Path(run_dir)
    protocol = verify_protocol(output)
    original = load_cage_calibration(calibration)
    if tensor_hash(original.tensor) != protocol["tensor_sha256"]:
        raise ValueError("Frozen calibration changed.")
    run_dir.mkdir(parents=True,exist_ok=True)
    fixed = dict(protocol,protocol_sha256=sha(output/"protocol.json"))
    jobs = []
    for horizon in protocol["terminal_sweep"]["horizons"]:
        for seed in protocol["terminal_sweep"]["seeds"]:
            target = run_dir/f"horizon_{horizon}_seed_{seed}.npz"
            if target.exists():
                verify_sweep_checkpoint(target,original.tensor,fixed["protocol_sha256"],horizon,seed)
            else:
                jobs.append((original.tensor,original.weights,horizon,seed,fixed,str(target)))
    total = protocol["terminal_sweep"]["new_path_count"]
    completed = total-len(jobs)
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(sweep_job,job) for job in jobs]
        for future in as_completed(futures):
            horizon,seed = future.result()
            completed += 1
            print(f"Terminal geometry {completed}/{total}: T={horizon}, seed={seed}, elapsed={time.perf_counter()-started:.1f}s",flush=True)
    combine_sweep(output,run_dir,original,protocol)


def combine_sweep(output,run_dir,original,protocol):
    horizons = protocol["terminal_sweep"]["horizons"]
    seeds = protocol["terminal_sweep"]["seeds"]
    digest = sha(Path(output)/"protocol.json")
    inputs = {"tensor":original.tensor,"weights":original.weights,"methods":np.asarray(METHODS),
              "horizons":np.asarray(horizons),"seeds":np.asarray(seeds),"protocol_sha256":np.asarray(digest)}
    all_rows,summaries,hashes = [],{},{}
    for horizon in horizons:
        opponents,actions,rows = [],[],[]
        summaries[str(horizon)] = []
        for seed in seeds:
            path = Path(run_dir)/f"horizon_{horizon}_seed_{seed}.npz"
            data = verify_sweep_checkpoint(path,original.tensor,digest,horizon,seed)
            hashes[path.name] = sha(path)
            opponents.append(data["opponent_actions"])
            actions.append(data["actions"])
            rows.append(data["geometry"])
            summaries[str(horizon)].append(json.loads(str(data["summaries_json"])))
        inputs[f"T{horizon}_opponent_actions"] = np.stack(opponents)
        inputs[f"T{horizon}_actions"] = np.stack(actions)
        all_rows.append(np.stack(rows))
    _atomic_npz(Path(output)/"horizon_inputs.npz",**inputs)
    _atomic_npz(Path(output)/"horizon_geometry.npz",rows=np.stack(all_rows),horizons=np.asarray(horizons),
                seeds=np.asarray(seeds),methods=np.asarray(METHODS),fields=np.asarray(FIELDS),
                protocol_sha256=np.asarray(digest))
    write(Path(output)/"horizon_sources.json",{"checkpoint_sha256":hashes,"summaries_by_horizon":summaries,
                                            "actual_path_count":len(horizons)*len(seeds),
                                            "actual_method_trajectory_count":3*len(horizons)*len(seeds),
                                            "native_simulator_episodes_added":0})


def slope_summary(horizons,lower,upper,floor):
    med_lower,med_upper = np.median(lower,axis=1),np.median(upper,axis=1)
    if np.any(med_lower <= floor) or np.any(med_upper <= floor):
        return {"estimated_slope":None,"reason":"At least one horizon median is zero or numerically unresolved."}
    slope,intercept = np.polyfit(np.log(horizons),np.log(med_upper),1)
    return {"estimated_slope":float(slope),"intercept":float(intercept),
            "interpretation":"Descriptive OLS of all horizon medians under changing scaled curricula and learning rates."}


def save_figure(fig,output,name,created):
    for suffix in ("png","pdf"):
        metadata = ({"Software":"Matplotlib"} if suffix == "png" else
                    {"Creator":"CAGE full-target geometry","CreationDate":created,"ModDate":created})
        fig.savefig(Path(output)/(name+"."+suffix),dpi=190,bbox_inches="tight",metadata=metadata)
    plt.close(fig)


def plot_curves(x,rows,language,prefix,output,created,floor,fixed=False):
    if not prefix:
        rows = rows.transpose(1,2,0,3)
    title = (("Distance to the full realized-hull target" if prefix else "Terminal error at independently run horizons")
             if language == "en" else
             ("Расстояние до полного целевого множества" if prefix else "Конечная ошибка при отдельных запусках"))
    if fixed:
        title = ("Fixed final target: retrospective diagnostic" if language == "en" else
                 "Фиксированная конечная цель: ретроспективная диагностика")
    fig,(ax,floor_ax) = plt.subplots(2,1,figsize=(9.4,6.4),sharex=True,
                                    gridspec_kw={"height_ratios":[3.2,1.2]},constrained_layout=True)
    reference_scale = .1
    for method,(label,color) in enumerate(zip(LABELS[language],COLORS)):
        distances = rows[:,method,:,1]
        med = np.median(distances,axis=0)
        low,high = np.quantile(distances,[.1,.9],axis=0)
        resolved = med>floor
        ax.plot(x,np.where(resolved,med,np.nan),label=label,color=color,
                marker="o" if not prefix else None,markersize=4)
        ax.fill_between(x,low,high,where=(low>floor)&resolved,color=color,alpha=.15)
        if method == 2 and np.any(resolved):
            first = np.flatnonzero(resolved)[0]
            reference_scale = float(med[first]*np.sqrt(x[first]))
        floor_ax.plot(x,100*np.mean(distances<=floor,axis=0),color=color,
                      marker="o" if not prefix else None,markersize=3)
    ax.plot(x,reference_scale/np.sqrt(x),"--",color="#686868",linewidth=1.2,
            label=(r"$T^{-1/2}$ guide (display scale)" if language == "en" else
                   r"Ориентир $T^{-1/2}$ (масштаб для показа)"))
    ax.set_xscale("log",base=2)
    ax.set_yscale("log")
    ax.set_title(title,fontweight="bold")
    ax.set_ylabel(r"$\delta_t$" if prefix else r"$\delta_T$")
    ax.grid(alpha=.22,which="both")
    ax.legend(fontsize=8.4,loc="best")
    floor_ax.set_ylim(-4,104)
    floor_ax.set_yticks([0,50,100])
    floor_ax.set_ylabel(("At numerical\nfloor, %" if language == "en" else "У численного\nпредела, %"),fontsize=9)
    floor_ax.set_xlabel((("Meta-round t" if prefix else "Announced horizon T") if language == "en" else
                        ("Раунд выбора политики t" if prefix else "Объявленный горизонт T")))
    floor_ax.grid(alpha=.22)
    if prefix:
        for boundary in (128,384):
            ax.axvline(boundary,color="#888888",alpha=.45,linewidth=.8)
            floor_ax.axvline(boundary,color="#888888",alpha=.45,linewidth=.8)
    fig.suptitle(("Frozen calibrated vector payoffs; median and 10-90% path range" if language == "en" else
                  "Векторные исходы калиброванной игры; медиана и диапазон 10–90% путей"),fontsize=10)
    floor_ax.text(.01,-.48,("Values <= 1e-10 are counted below, not shifted onto the log axis. The guide is not a fitted rate."
                            if language == "en" else
                            "Значения ≤ 1e-10 учтены отдельно; сдвиг для логарифма не применяется. Ориентир не является оценкой порядка."),
                  transform=floor_ax.transAxes,fontsize=8)
    floor_ax.text(.01,-.67,("Bands whose lower limit reaches the display floor are omitted on the log panel; their floor share is shown separately."
                           if language == "en" else
                           "Диапазоны с нижней границей у порога на log-панели не показаны; доля значений ≤ 1e-10 дана отдельно."),
                  transform=floor_ax.transAxes,fontsize=7.8)
    base = "prefix_fixed_target" if fixed else ("prefix_error" if prefix else "horizon_error")
    save_figure(fig,output,base+(".ru" if language == "ru" else ""),created)


def build_report(output):
    output = Path(output)
    protocol = verify_protocol(output)
    with np.load(output/"prefix_geometry.npz",allow_pickle=False) as data:
        prefix,fixed,grid = data["rows"],data["fixed_target_rows"],data["checkpoints"]
    with np.load(output/"horizon_geometry.npz",allow_pickle=False) as data:
        terminal,horizons = data["rows"],data["horizons"]
    floor = protocol["geometry"]["near_floor_threshold"]
    metrics = {
        "protocol_sha256":sha(output/"protocol.json"),
        "software":{"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,"matplotlib":matplotlib.__version__},
        "prefix":{"path_count":len(prefix),"checkpoint_count":len(grid),
                  "moving_projection_count":int(np.prod(prefix.shape[:-1])),
                  "fixed_projection_count":int(np.prod(fixed.shape[:-1])),
                  "max_gap":float(max(prefix[...,2].max(),fixed[...,2].max())),
                  "max_alpha":float(max(prefix[...,3].max(),fixed[...,3].max())),
                  "all_success":bool(np.all(prefix[...,4] == 1) and np.all(fixed[...,4] == 1)),
                  "maximum_affine_dimension":int(prefix[...,7].max()),"methods":{}},
        "terminal_sweep":{"horizons":horizons.tolist(),"path_count_per_horizon":terminal.shape[1],
                          "projection_count":int(np.prod(terminal.shape[:-1])),
                          "max_gap":float(terminal[...,2].max()),"max_alpha":float(terminal[...,3].max()),
                          "all_success":bool(np.all(terminal[...,4] == 1)),"methods":{}},
        "numerical_floor":floor,"native_simulator_episodes_added":0,
    }
    for index,name in enumerate(METHODS):
        metrics["prefix"]["methods"][name] = {
            "terminal_median_distance":float(np.median(prefix[:,index,-1,1])),
            "terminal_max_distance":float(prefix[:,index,-1,1].max()),
            "terminal_at_or_below_floor_count":int(np.sum(prefix[:,index,-1,1]<=floor)),
            "total_at_or_below_floor_count":int(np.sum(prefix[:,index,:,1]<=floor))}
        metrics["terminal_sweep"]["methods"][name] = {
            "median_distances":np.median(terminal[:,:,index,1],axis=1).tolist(),
            "median_lower_distances":np.median(terminal[:,:,index,0],axis=1).tolist(),
            "maximum_distances":terminal[:,:,index,1].max(axis=1).tolist(),
            "at_or_below_floor_counts":(terminal[:,:,index,1]<=floor).sum(axis=1).tolist(),
            **slope_summary(horizons,terminal[:,:,index,0],terminal[:,:,index,1],floor)}
    write(output/"metrics.json",metrics)
    created = datetime.fromisoformat(protocol["created_utc"])
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    for language in ("en","ru"):
        plot_curves(grid,prefix,language,True,output,created,floor)
        plot_curves(grid,fixed,language,True,output,created,floor,fixed=True)
        plot_curves(horizons,terminal,language,False,output,created,floor)
    write_readmes(output,protocol,metrics)
    names = sorted(path.name for path in output.iterdir() if path.is_file() and path.name!="SHA256SUMS.json")
    write(output/"SHA256SUMS.json",{"algorithm":"sha256","files":{name:sha(output/name) for name in names}})
    return metrics


def write_readmes(output,protocol,metrics):
    # Generated factual captions keep both language versions tied to the same data.
    english = """[Русский](README.ru.md) · [Main report](../README.md)

Full-target vector error of the selected implementation
======================================================

The error is delta_t = dist(mean_(s<=t) u(p_s, ell_s), S(Q_t)) in the frozen normalized TRAINING game. It is distinct from held-out native simulator cost. The target includes the full realized opponent hull, its mixtures and all actual response cells.

![Prefix error](prefix_error.png)

All 50 locked primary paths and all three selected methods are retained. The target expands upon first appearances of modes: a reduction can reflect both average-payoff dynamics and target expansion. The vertical lines mark curriculum phases. Early decay of our method also dilutes the one uniform initial action; it does not establish an asymptotic order.

![Fixed target](prefix_fixed_target.png)

The same payoff prefixes are compared with the fixed final S(Q_T). This is a retrospective diagnostic, using the final realized hull only for evaluation. It separates changes of the target from changes of the mean payoff and does not change any learner's information.

![Separate horizon sweep](horizon_error.png)

The independent terminal sweep runs 120 new common paths / 360 method trajectories: 20 seeds 45000000–45000019 at each prespecified horizon 64, 128, 256, 512, 1024 and 2048. Curriculum phase lengths and the original Hedge/opponent learning-rate formulas scale with the announced horizon. These are separate runs, not truncated longest-run prefixes. Settings W=16, rho=0.25 and Hedge multiplier 4 are unchanged. No parameter is retuned, no seed omitted and no new simulator episode collected.

Median lines and 10–90% empirical path bands are conditional on the frozen training calibration. Values at or below the declared numerical display floor 1e-10 are counted in the lower panel, never shifted by an epsilon for logarithmic plotting. Every numerical lower/upper distance, gap, alpha=sqrt(gap), success flag, feasibility residual and affine dimension is published in the NPZ arrays. Numerical checks are not exact-arithmetic proofs. The dashed T^(-1/2) line is a display-scaled guide for dimension at most two, not a fitted theorem constant.

MEDIAN_TABLE

Bands whose lower limit reaches the display floor are omitted on the logarithmic panel; the share of values at or below 1e-10 is shown separately.

The terminal sweep does not identify a decay order for our method or window when their errors are at numerical precision. These figures can show small full-target error relative to Hedge; they do not establish an advantage over the window strategy. See metrics.json for all values and any descriptive slope estimate; any such estimate concerns the changing horizon-scaled scenario, not a proof of an asymptotic rate.

Reproduce figures with the committed arrays:

    python scripts/build_cage_geometry_figures.py --stage report

To rerun original-path projections and new trajectories:

    python scripts/build_cage_geometry_figures.py --stage all --workers 4

The full stage replays published primary_inputs.npz if original ignored checkpoints are absent. It contains actions [50,3,512,6], opponent_actions [50,512,3], tensor, weights, methods and path_seeds. horizon_inputs.npz publishes every new path/action array keyed by horizon. prefix_geometry.npz and horizon_geometry.npz store all diagnostics. Protocols record settings, seeds and frozen source hashes before the new sweep; source files preserve checkpoint hashes; metrics.json records software; SHA256SUMS.json covers all artifacts.

Calibration uses the pinned [official CAGE Challenge 2 simulator](https://github.com/cage-challenge/cage-challenge-2/tree/26ce1c1253fa9e2e73f25e6a7f2da32860c11257). See [methodology](../../../docs/cage_adaptation_en.md). The original article, calibration, selected settings and algorithm core are unchanged.
"""
    russian = """[English](README.md) · [Основной отчёт](../README.ru.md)

Векторная ошибка выбранной реализации относительно полной цели
============================================================

Измеряется delta_t = dist(mean_(s<=t) u(p_s, ell_s), S(Q_t)) в фиксированной нормированной ОБУЧАЮЩЕЙ игре. Это другой показатель, чем стоимость на тестовых эпизодах симулятора. Учитывается полная оболочка реализованных действий противника, её смеси и все области выбора ответа.

![Ошибка по префиксам](prefix_error.ru.png)

Сохранены все 50 исходных тестовых путей и все три выбранных метода. При первом появлении новых режимов цель расширяется: снижение расстояния может отражать и динамику среднего исхода, и расширение цели. Вертикальные линии отмечают фазы противника. Раннее убывание ошибки нашего метода также уменьшает вклад первого равномерного хода; оно не устанавливает асимптотический порядок.

![Фиксированная цель](prefix_fixed_target.ru.png)

Те же префиксы сравниваются с фиксированной конечной целью S(Q_T). Это ретроспективная диагностика: конечная оболочка используется только при оценке. Она отделяет изменение цели от изменения среднего исхода и не меняет информацию, доступную алгоритмам.

![Отдельные запуски на разных горизонтах](horizon_error.ru.png)

Отдельно выполнено 120 новых общих путей / 360 траекторий методов: 20 начальных состояний генератора 45000000–45000019 на каждом заранее заданном горизонте 64, 128, 256, 512, 1024 и 2048. Длины фаз и исходные формулы скорости обучения Hedge и противника масштабируются с объявленным горизонтом. Это отдельные запуски, а не усечённые префиксы самого длинного пути. W=16, rho=0,25 и множитель Hedge 4 сохранены. Параметры не подбирались повторно, пути не исключались, новые эпизоды симулятора не собирались.

Показаны медиана и диапазон 10–90% путей при фиксированной обучающей калибровке. Значения не выше заранее объявленного порога отображения 1e-10 учитываются в нижней панели; положительная константа для логарифма не прибавляется. В NPZ опубликованы все нижние/верхние оценки расстояния, разрыв, alpha=sqrt(gap), успешность численной проверки, нарушение допустимости и аффинная размерность. Численные проверки не являются доказательством в точной арифметике. Пунктир T^(-1/2) — ориентир с масштабом для показа при размерности не выше двух, а не оценка константы теоремы.

MEDIAN_TABLE

Диапазоны, нижняя граница которых достигает порога отображения, на логарифмической панели не показаны; доля значений ≤ 1e-10 дана отдельно.

Конечные ошибки у численного предела не позволяют установить порядок убывания нашего метода или окна на этих данных. Графики показывают малую ошибку относительно полной цели в сравнении с Hedge, но не устанавливают преимущества над оконной стратегией. Все значения и возможная описательная оценка наклона находятся в metrics.json; такая оценка относится к меняющимся постановкам с масштабированными фазами, а не доказывает асимптотический порядок.

Повтор графиков из опубликованных массивов:

    python scripts/build_cage_geometry_figures.py --stage report

Повтор исходных проекций и новых траекторий:

    python scripts/build_cage_geometry_figures.py --stage all --workers 4

Если исходные игнорируемые контрольные файлы отсутствуют, полный этап читает опубликованный primary_inputs.npz. В нём actions [50,3,512,6], opponent_actions [50,512,3], tensor, weights, methods и path_seeds. horizon_inputs.npz содержит каждый новый путь и действия по горизонтам. prefix_geometry.npz и horizon_geometry.npz содержат диагностики. Протоколы фиксируют условия, начальные состояния генератора и хеши исходников до новых запусков; source-файлы сохраняют хеши контрольных файлов; metrics.json — версии ПО; SHA256SUMS.json покрывает артефакты.

Калибровка использует зафиксированную версию [официального CAGE Challenge 2](https://github.com/cage-challenge/cage-challenge-2/tree/26ce1c1253fa9e2e73f25e6a7f2da32860c11257). См. [методику](../../../docs/cage_adaptation_ru.md). Статья, калибровка, выбранные настройки и основные алгоритмы не изменены.
"""
    table = "| T | Our one-switch | Window | Hedge |\n|---:|---:|---:|---:|\n"
    for index,horizon in enumerate(protocol["terminal_sweep"]["horizons"]):
        values = [metrics["terminal_sweep"]["methods"][name]["median_distances"][index] for name in METHODS]
        labels = ["≤ 1e−10" if value<=1e-10 else f"{value:.6g}" for value in values]
        table += f"| {horizon} | "+" | ".join(labels)+" |\n"
    english = english.replace("MEDIAN_TABLE","Median terminal numerical upper distances:\n\n"+table)
    russian = russian.replace("MEDIAN_TABLE","Медианы численных верхних оценок конечного расстояния:\n\n"+
                                table.replace("Our one-switch","Наш one-switch").replace("Window","Окно"))
    (Path(output)/"README.md").write_text(english,encoding="utf-8",newline="\n")
    (Path(output)/"README.ru.md").write_text(russian,encoding="utf-8",newline="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=("prepare","prefix","sweep","report","all"),default="all")
    parser.add_argument("--output",type=Path,default=REPO/"results/cage_adaptation/geometry")
    parser.add_argument("--primary-dir",type=Path,default=REPO/"results/runs/cage_adaptation/selection/test")
    parser.add_argument("--calibration",type=Path,default=REPO/"data/cage2")
    parser.add_argument("--run-dir",type=Path,default=REPO/"results/runs/cage_geometry_horizons")
    parser.add_argument("--workers",type=int,default=4)
    args = parser.parse_args()
    if args.stage in ("prepare","all"):
        freeze_protocol(args.output,args.calibration,args.primary_dir)
    if args.stage in ("prefix","all"):
        prefix_geometry(args.output,args.primary_dir,args.calibration)
    if args.stage in ("sweep","all"):
        terminal_sweep(args.output,args.calibration,args.run_dir,args.workers)
    if args.stage in ("report","all"):
        print(json.dumps(build_report(args.output),indent=2),flush=True)


if __name__ == "__main__":
    main()
