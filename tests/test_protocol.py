import numpy as np
from lowdim_games.benchmarks import make_synthetic, regime_path
from lowdim_games.experiment import run_comparison, LastWeekResponse
from lowdim_games.game import FiniteGame


def test_complete_protocol_records_full_target_and_real_actions(tmp_path):
    instance = make_synthetic(seed=4)
    path, _ = regime_path(2, 12, seed=8)
    results = run_comparison(instance, path, name="test", output_dir=tmp_path)
    assert {x["method"] for x in results["summaries"]} == {
        "one_switch", "shared_past_hull", "block_safe", "uniform",
        "historical_share", "reserve", "last_week"}
    assert results["realized_affine_dimension"] == 2
    assert results["max_target_projection_gap"] <= 1e-8
    first, second = results["summaries"][:2]
    assert abs(first["delta"] - second["delta"]) < 1e-10
    assert first["switch_round"] is None
    for item in results["summaries"]:
        arrays = np.load(tmp_path / f'test__{item["method"]}.npz')
        assert np.allclose(arrays["actions"].sum(axis=1), 1)
        assert np.all(arrays["actions"] >= 0)


def test_last_week_cannot_see_current_observation():
    instance = make_synthetic(seed=6)
    game = FiniteGame(instance.tensor, instance.weights)
    learner = LastWeekResponse(game)
    first = learner.choose()
    assert np.array_equal(first, np.eye(8)[0])
    past = np.eye(8)[3]
    learner.observe(past)
    assert np.array_equal(learner.choose(), game.response(past))
