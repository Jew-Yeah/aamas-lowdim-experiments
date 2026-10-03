"""Run real CAGE Challenge 2 episodes and save a reproducible calibration bank."""

import argparse
from pathlib import Path

from lowdim_games.cage import calibrate_cage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(".external/cage-challenge-2"))
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--train-episodes", type=int, default=40)
    parser.add_argument("--test-episodes", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--output", type=Path, default=Path("data/cage2"))
    args = parser.parse_args()
    train = range(args.seed, args.seed + args.train_episodes)
    test = range(args.seed + 1_000_000, args.seed + 1_000_000 + args.test_episodes)
    result = calibrate_cage(args.source, args.steps, train, test, args.output)
    print(f"Saved {result.provenance['episodes_total']} simulator episodes to {args.output}")


if __name__ == "__main__":
    main()
