"""Empirical finite games calibrated with the official CAGE Challenge 2 simulator.

This module does not contain a substitute simulator.  CybORG is imported from
an explicitly supplied, revision-pinned upstream checkout.  A meta-game action
selects a frozen policy for a complete episode, and every episode starts from a
fresh network state.  Calibration and held-out episodes remain separate.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np


CAGE_SOURCE_URL = "https://github.com/cage-challenge/cage-challenge-2"
CAGE_REVISION = "26ce1c1253fa9e2e73f25e6a7f2da32860c11257"
METRIC_NAMES = ("host_compromise", "server_compromise",
                "operational_disruption", "restore")
BLUE_NAMES = ("monitor", "react_remove", "react_restore", "decoy_cycle",
              "critical_restore", "decoy_react")
RED_NAMES = ("sleep", "meander", "b_line")
POLICY_DESCRIPTIONS = {
    "monitor": "Official BlueMonitorAgent; automatic monitoring only.",
    "react_remove": "Official BlueReactRemoveAgent, unchanged.",
    "react_restore": "Official BlueReactRestoreAgent, unchanged.",
    "decoy_cycle": "Cycle eight official decoy types over four critical hosts; no restores.",
    "critical_restore": "Restore Op_Server0 and Enterprise2 alternately every third step; monitor otherwise.",
    "decoy_react": "Restore hosts with a PID in the latest Blue observation, with a three-step host cooldown; deploy cyclic decoys otherwise.",
}


@dataclass(frozen=True)
class CageCalibration:
    tensor: np.ndarray
    train_mean: np.ndarray
    test_mean: np.ndarray
    train_episode_losses: np.ndarray
    test_episode_losses: np.ndarray
    train_seeds: np.ndarray
    test_seeds: np.ndarray
    weights: np.ndarray
    blue_names: tuple
    red_names: tuple
    metric_names: tuple
    normalization_scale: float
    provenance: dict


@dataclass(frozen=True)
class CageEpisode:
    mean_loss: np.ndarray
    step_losses: np.ndarray
    scalar_rewards: np.ndarray
    actions: tuple
    seed: int
    episode_steps: int
    max_reward_reconstruction_error: float


def _scenario_path(source_dir):
    return Path(source_dir).resolve() / "CybORG/CybORG/Shared/Scenarios/Scenario2.yaml"


def _import_simulator(source_dir):
    source_dir = Path(source_dir).resolve()
    package_root = source_dir / "CybORG"
    if not _scenario_path(source_dir).is_file():
        raise FileNotFoundError(f"Official CAGE Scenario2.yaml missing under {source_dir}.")
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    try:
        import CybORG
        from CybORG.Agents import (B_lineAgent, RedMeanderAgent, SleepAgent,
                                  BlueMonitorAgent, BlueReactRemoveAgent,
                                  BlueReactRestoreAgent)
        from CybORG.Shared import Actions
    except ImportError as exc:
        raise ImportError("CAGE simulator requires its public source checkout and "
                          "optional dependencies PyYAML, paramiko and prettytable. "
                          "Install the cage extra or use the documented isolated environment.") from exc
    imported = Path(CybORG.__file__).resolve()
    if not imported.is_relative_to(package_root):
        raise RuntimeError(f"Another CybORG installation is already imported: {imported}.")
    reds = {"sleep": SleepAgent, "meander": RedMeanderAgent, "b_line": B_lineAgent}
    blues = {"monitor": BlueMonitorAgent, "react_remove": BlueReactRemoveAgent,
             "react_restore": BlueReactRestoreAgent}
    return CybORG.CybORG, Actions, reds, blues


def scenario_component_bounds(source_dir):
    """A priori per-step loss bounds from Scenario2, independent of observations.

    User0 has zero confidentiality value.  Four other User hosts, three
    Op_Host hosts, and Defender have value 0.1, giving 0.8 in component one.
    The four server hosts contribute at most 4.0.  OTService disruption is
    penalized by 10.0 on every step while it is absent.  Restore costs 1.0.
    """
    import yaml
    scenario = yaml.safe_load(_scenario_path(source_dir).read_text(encoding="utf-8"))
    mapping = {"None": 0.0, "Low": 0.1, "Medium": 1.0, "High": 10.0}
    bounds = np.zeros(4)
    for name, host in scenario["Hosts"].items():
        group = 1 if name.startswith("Enterprise") or name == "Op_Server0" else 0
        bounds[group] += mapping[host.get("ConfidentialityValue", "Low")]
    # The pinned scenario has one OTService, on Op_Server0.  The native
    # disruption reward only counts missing OTService processes.
    bounds[2] = mapping[scenario["Hosts"]["Op_Server0"].get("AvailabilityValue", "Low")]
    bounds[3] = 1.0
    return bounds


def reconstruct_loss_vector(host_scores, action_cost, scalar_reward):
    """Recover native reward categories and check their sum on every step.

    get_reward_breakdown('Blue') excludes action cost.  In the official code,
    cost is -1 for Restore even when it fails.  InvalidAction has cost -0.1;
    it is rejected here rather than silently relabelled as a restore.
    """
    if not np.isclose(action_cost, 0.0) and not np.isclose(action_cost, -1.0):
        raise RuntimeError(f"Unexpected Blue action cost {action_cost}; possibly InvalidAction.")
    loss = np.zeros(4)
    for name, scores in host_scores.items():
        group = 1 if name.startswith("Enterprise") or name == "Op_Server0" else 0
        loss[group] -= float(scores.confidentiality)
        loss[2] -= float(scores.availability)
    loss[3] = -float(action_cost)
    if np.any(loss < -1e-12) or not np.all(np.isfinite(loss)):
        raise RuntimeError(f"Unexpected signed/non-finite native loss: {loss}.")
    error = abs(float(loss.sum()) + float(scalar_reward))
    if error > 1e-8:
        raise RuntimeError(f"Native reward decomposition mismatch: vector {loss}, reward {scalar_reward}.")
    return loss, error


class _FrozenBluePolicy:
    """Predeclared heuristics using only the latest partial Blue observation."""

    def __init__(self, name, actions, native_agents):
        if name not in BLUE_NAMES:
            raise ValueError(f"Unknown frozen Blue policy {name}.")
        self.name = name
        self.actions = actions
        self.native = native_agents[name]() if name in native_agents else None
        self.step = 0
        self.decoy_index = 0
        self.last_restore = {}
        self.hosts = ("Op_Server0", "Enterprise2", "Enterprise1", "Enterprise0")
        self.decoys = tuple(getattr(actions, name) for name in (
            "DecoySSHD", "DecoyHarakaSMPT", "DecoySmss", "DecoyApache",
            "DecoyFemitter", "DecoySvchost", "DecoyTomcat", "DecoyVsftpd"))

    def _decoy(self, session):
        index = self.decoy_index
        self.decoy_index += 1
        host = self.hosts[index % len(self.hosts)]
        decoy = self.decoys[(index // len(self.hosts)) % len(self.decoys)]
        return decoy(agent="Blue", session=session, hostname=host)

    def get_action(self, observation, action_space):
        step = self.step
        self.step += 1
        if self.native is not None:
            return self.native.get_action(observation, action_space)
        known_sessions = [key for key, available in action_space["session"].items() if available]
        if not known_sessions:
            raise RuntimeError("No available Blue session.")
        session = min(known_sessions)
        if self.name == "critical_restore":
            if step % 3 == 0:
                host = ("Op_Server0", "Enterprise2")[(step // 3) % 2]
                return self.actions.Restore(agent="Blue", session=session, hostname=host)
            return self.actions.Monitor(agent="Blue", session=session)
        if self.name == "decoy_react":
            # Native monitoring includes legitimate activity.  This is an
            # explicitly fixed heuristic, not a true-state compromise oracle.
            suspicious = []
            for value in observation.values():
                if not isinstance(value, dict) or "System info" not in value:
                    continue
                host = value["System info"].get("Hostname")
                if host is None or host == "User0":
                    continue
                if any("PID" in process for process in value.get("Processes", [])):
                    if step - self.last_restore.get(host, -1000) >= 3:
                        suspicious.append(host)
            if suspicious:
                priority = {host: i for i, host in enumerate(self.hosts)}
                host = min(suspicious, key=lambda name: (priority.get(name, 10), name))
                self.last_restore[host] = step
                return self.actions.Restore(agent="Blue", session=session, hostname=host)
        return self._decoy(session)


def run_cage_episode(source_dir, blue_name, red_name, seed, episode_steps=50):
    """Run a real fresh-state simulator episode with frozen policies.

    Seeding occurs before construction because CybORG randomizes its initial
    network during construction.  Global RNG states are restored afterwards;
    the simulator uses Python's random module, and NumPy is seeded defensively.
    """
    if episode_steps < 1:
        raise ValueError("episode_steps must be positive.")
    simulator, actions, red_agents, blue_agents = _import_simulator(source_dir)
    if red_name not in RED_NAMES:
        raise ValueError(f"Unknown official Red policy {red_name}.")
    random_state = random.getstate()
    numpy_state = np.random.get_state()
    try:
        random.seed(int(seed))
        np.random.seed(int(seed) % (2 ** 32))
        env = simulator(str(_scenario_path(source_dir)), "sim",
                        agents={"Red": red_agents[red_name]})
        result = env.reset(agent="Blue")
        policy = _FrozenBluePolicy(blue_name, actions, blue_agents)
        bounds = scenario_component_bounds(source_dir)
        losses, rewards, executed = [], [], []
        errors = []
        for _ in range(episode_steps):
            action = policy.get_action(result.observation, result.action_space)
            result = env.step(agent="Blue", action=action)
            loss, error = reconstruct_loss_vector(
                env.get_reward_breakdown("Blue"), result.action.cost, result.reward)
            if np.any(loss > bounds + 1e-10):
                raise RuntimeError(f"Native loss exceeds scenario bounds: {loss} > {bounds}.")
            losses.append(loss)
            rewards.append(float(result.reward))
            executed.append(str(result.action))
            errors.append(error)
        values = np.asarray(losses)
        return CageEpisode(values.mean(axis=0), values, np.asarray(rewards),
                           tuple(executed), int(seed), int(episode_steps), max(errors))
    finally:
        random.setstate(random_state)
        np.random.set_state(numpy_state)


def _source_provenance(source_dir):
    source_dir = Path(source_dir).resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(source_dir), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(source_dir), "status", "--porcelain", "--untracked-files=no"],
        text=True).strip()
    if revision != CAGE_REVISION or dirty:
        raise RuntimeError(f"Expected clean official source at {CAGE_REVISION}; "
                           f"got {revision}, tracked changes={bool(dirty)}.")
    paths = ("CybORG/CybORG/Shared/Scenarios/Scenario2.yaml",
             "CybORG/CybORG/Shared/BlueRewardCalculator.py",
             "CybORG/CybORG/Shared/RedRewardCalculator.py",
             "CybORG/CybORG/Shared/EnvironmentController.py",
             "CybORG/CybORG/Shared/Actions/AbstractActions/Restore.py")
    return {"url": CAGE_SOURCE_URL, "revision": revision, "tracked_source_clean": True,
            "source_sha256": {path: hashlib.sha256((source_dir / path).read_bytes()).hexdigest()
                              for path in paths}, "upstream_modifications": []}


def _validate_seeds(train_seeds, test_seeds):
    train = np.asarray(tuple(train_seeds), dtype=np.int64)
    test = np.asarray(tuple(test_seeds), dtype=np.int64)
    for name, values in (("train", train), ("test", test)):
        if len(values) < 2 or len(set(values)) != len(values):
            raise ValueError(f"Require at least two distinct {name} seeds for uncertainty estimates.")
    if set(train) & set(test):
        raise ValueError("Calibration and held-out seeds must be disjoint.")
    return train, test


def calibrate_cage(source_dir, episode_steps=50, train_seeds=range(100, 110),
                   test_seeds=range(200, 210), output_dir=None):
    """Calibrate all frozen Blue/Red cells, then evaluate disjoint held-out seeds.

    Each seed is reused across cells for paired comparisons.  Policy changes
    alter RNG consumption, so this pairing does not create identical simulator
    trajectories.  All episode observations are real simulator outputs.
    """
    train_seeds, test_seeds = _validate_seeds(train_seeds, test_seeds)
    source = _source_provenance(source_dir)
    bounds = scenario_component_bounds(source_dir)
    scale = float(np.linalg.norm(bounds))
    started = time.perf_counter()
    banks = {}
    max_error = 0.0
    for partition, seeds in (("train", train_seeds), ("test", test_seeds)):
        bank = np.empty((len(seeds), len(BLUE_NAMES), len(RED_NAMES), len(METRIC_NAMES)))
        for seed_index, seed in enumerate(seeds):
            for blue_index, blue in enumerate(BLUE_NAMES):
                for red_index, red in enumerate(RED_NAMES):
                    result = run_cage_episode(source_dir, blue, red, seed, episode_steps)
                    bank[seed_index, blue_index, red_index] = result.mean_loss
                    max_error = max(max_error, result.max_reward_reconstruction_error)
            print(f"CAGE {partition}: {seed_index + 1}/{len(seeds)} seeds, "
                  f"{len(BLUE_NAMES) * len(RED_NAMES) * (seed_index + 1)} episodes", flush=True)
        banks[partition] = bank
    train_mean = banks["train"].mean(axis=0)
    test_mean = banks["test"].mean(axis=0)
    packages = {}
    for package in ("numpy", "scipy", "PyYAML", "paramiko", "prettytable"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    provenance = {
        "schema_version": 1, "benchmark": "CAGE Challenge 2 / CybORG simulator",
        "created_utc": datetime.now(timezone.utc).isoformat(), "source": source,
        "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scenario": "Scenario2.yaml", "simulation_only": True,
        "episode_steps": int(episode_steps), "fresh_environment_every_episode": True,
        "seeded_before_environment_construction": True,
        "train_seeds": train_seeds.tolist(), "test_seeds": test_seeds.tolist(),
        "episode_loss_layout": ["seed", "blue_policy", "red_policy", "component"],
        "blue_names": list(BLUE_NAMES), "blue_policy_descriptions": POLICY_DESCRIPTIONS,
        "red_names": list(RED_NAMES), "metric_names": list(METRIC_NAMES),
        "loss_unit": "native negative reward converted to positive loss, averaged per simulator step",
        "component_bounds": bounds.tolist(), "normalization_scale": scale,
        "normalization": "Euclidean norm of a priori scenario component bounds; independent of train/test observations",
        "weights": [1.0] * len(METRIC_NAMES),
        "reward_reconstruction_max_abs_error": max_error,
        "impact_definition": "Penalty on each step while OTService is absent; it persists until restoration.",
        "restore_definition": "Negative executed Restore action cost, including failed Restore; no other Blue action costs accepted.",
        "tensor_estimation": "Only calibration means define the finite game. Held-out episodes never change the tensor, response map, menu, weights, or normalization.",
        "uncertainty": "Independent seed episodes; sample standard deviation and standard error per cell; common seed indices support paired downstream differences.",
        "elapsed_seconds": time.perf_counter() - started,
        "python": sys.version.split()[0], "packages": packages,
        "episodes_total": int((len(train_seeds) + len(test_seeds)) * len(BLUE_NAMES) * len(RED_NAMES)),
    }
    calibration = CageCalibration(train_mean / scale, train_mean, test_mean,
                                  banks["train"], banks["test"], train_seeds,
                                  test_seeds, np.ones(4), BLUE_NAMES, RED_NAMES,
                                  METRIC_NAMES, scale, provenance)
    if output_dir is not None:
        save_cage_calibration(calibration, output_dir)
    return calibration


def save_cage_calibration(calibration, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "calibration.npz"
    np.savez_compressed(path, tensor=calibration.tensor, train_mean=calibration.train_mean,
                        test_mean=calibration.test_mean,
                        train_episode_losses=calibration.train_episode_losses,
                        test_episode_losses=calibration.test_episode_losses,
                        train_seeds=calibration.train_seeds, test_seeds=calibration.test_seeds,
                        weights=calibration.weights,
                        blue_names=np.asarray(calibration.blue_names),
                        red_names=np.asarray(calibration.red_names),
                        metric_names=np.asarray(calibration.metric_names),
                        normalization_scale=calibration.normalization_scale)
    provenance = dict(calibration.provenance)
    provenance["calibration_npz_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    statistics = {}
    for partition, bank in (("train", calibration.train_episode_losses),
                            ("test", calibration.test_episode_losses)):
        std = bank.std(axis=0, ddof=1)
        statistics[partition] = {"episodes_per_cell": len(bank), "mean": bank.mean(axis=0).tolist(),
                                 "standard_deviation": std.tolist(),
                                 "standard_error": (std / np.sqrt(len(bank))).tolist()}
    (output_dir / "cell_statistics.json").write_text(
        json.dumps(statistics, indent=2) + "\n", encoding="utf-8", newline="\n")
    return path


def load_cage_calibration(path):
    path = Path(path)
    if path.is_dir():
        path = path / "calibration.npz"
    provenance = json.loads((path.parent / "provenance.json").read_text(encoding="utf-8"))
    expected = provenance["calibration_npz_sha256"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError("CAGE calibration checksum mismatch.")
    with np.load(path, allow_pickle=False) as values:
        calibration = CageCalibration(
            values["tensor"].copy(), values["train_mean"].copy(), values["test_mean"].copy(),
            values["train_episode_losses"].copy(), values["test_episode_losses"].copy(),
            values["train_seeds"].copy(), values["test_seeds"].copy(), values["weights"].copy(),
            tuple(values["blue_names"].tolist()), tuple(values["red_names"].tolist()),
            tuple(values["metric_names"].tolist()), float(values["normalization_scale"]), provenance)
    if not np.allclose(calibration.tensor * calibration.normalization_scale,
                       calibration.train_mean, atol=1e-13, rtol=1e-13):
        raise ValueError("CAGE tensor does not match calibration-only means.")
    if not np.allclose(calibration.train_episode_losses.mean(axis=0), calibration.train_mean):
        raise ValueError("CAGE calibration episode means do not match stored table.")
    if not np.allclose(calibration.test_episode_losses.mean(axis=0), calibration.test_mean):
        raise ValueError("CAGE held-out episode means do not match stored table.")
    _validate_seeds(calibration.train_seeds, calibration.test_seeds)
    return calibration


load_calibration = load_cage_calibration
