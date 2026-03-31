"""Command-line interface for electoral_sim."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from electoral_sim.ballots import BallotProfile
from electoral_sim.fractional import (
    FractionalBallotContinuous,
    FractionalBallotDiscrete,
    fractional_ballot_systems,
)
from electoral_sim.metrics import compute_metrics
from electoral_sim.scenario import built_in_scenario_paths, load_scenario
from electoral_sim.systems import SYSTEM_REGISTRY, ElectoralSystem, get_all_systems


DEFAULT_FRACTIONAL_SIGMAS = [0.1, 0.3, 1.0]
METRIC_FIELDS = [
    "distance_to_median",
    "distance_to_mean",
    "majority_satisfaction",
    "worst_case_distance",
    "mean_voter_distance",
    "gini_distance",
    "centroid_distance_to_median",
    "centroid_distance_to_mean",
]
SORTABLE_FIELDS = ["system_name", "winner_labels"] + METRIC_FIELDS


def _parse_system_keys(raw_values: Iterable[str] | None) -> list[str]:
    keys: list[str] = []
    for raw in raw_values or []:
        for part in raw.split(","):
            key = part.strip()
            if key:
                keys.append(key)
    return keys


def _parse_sigmas(raw: str | None) -> list[float]:
    if not raw:
        return DEFAULT_FRACTIONAL_SIGMAS
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def _instantiate_system(key: str) -> ElectoralSystem:
    normalized = key.strip().lower()
    if normalized in SYSTEM_REGISTRY:
        return SYSTEM_REGISTRY[normalized]()
    if normalized == "fractional_discrete":
        return FractionalBallotDiscrete()
    if normalized == "fractional_continuous":
        return FractionalBallotContinuous()
    raise KeyError(normalized)


def _build_systems(args: argparse.Namespace, rng: np.random.Generator) -> list[ElectoralSystem]:
    selected_keys = _parse_system_keys(args.system)

    if selected_keys:
        systems = [_instantiate_system(key) for key in selected_keys]
    else:
        systems = get_all_systems(rng=rng)

    if args.include_fractional:
        systems.extend(
            fractional_ballot_systems(
                sigmas=_parse_sigmas(args.fractional_sigmas),
                variant=args.fractional_variant,
            )
        )

    return systems


def _winner_labels(result, candidates) -> str:
    return ", ".join(candidates.labels[idx] for idx in result.winner_indices)


def _format_float(value: float) -> str:
    if np.isnan(value):
        return "nan"
    return f"{value:.4f}"


def _tabular_line(values: list[str], widths: list[int]) -> str:
    return "  ".join(value.ljust(width) for value, width in zip(values, widths))


def _print_table(rows: list[dict]) -> None:
    columns = [
        ("system_name", "System"),
        ("winner_labels", "Winner(s)"),
        ("distance_to_median", "d_median"),
        ("majority_satisfaction", "majority_sat"),
        ("distance_to_mean", "d_mean"),
        ("mean_voter_distance", "mean_voter_dist"),
    ]
    formatted_rows = []
    for row in rows:
        formatted_rows.append(
            [
                str(row["system_name"]),
                str(row["winner_labels"]),
                _format_float(row["distance_to_median"]),
                _format_float(row["majority_satisfaction"]),
                _format_float(row["distance_to_mean"]),
                _format_float(row["mean_voter_distance"]),
            ]
        )

    header = [label for _, label in columns]
    widths = [
        max(len(header[idx]), *(len(row[idx]) for row in formatted_rows))
        for idx in range(len(columns))
    ]

    print(_tabular_line(header, widths))
    print(_tabular_line(["-" * width for width in widths], widths))
    for row in formatted_rows:
        print(_tabular_line(row, widths))


def _emit_rows(rows: list[dict], output_format: str) -> None:
    if output_format == "json":
        print(json.dumps(rows, indent=2, sort_keys=True))
        return

    if output_format == "csv":
        writer = csv.DictWriter(
            __import__("sys").stdout,
            fieldnames=list(rows[0].keys()) if rows else ["system_name"],
        )
        writer.writeheader()
        writer.writerows(rows)
        return

    _print_table(rows)


def _collect_rows(
    scenario_path: str,
    systems: list[ElectoralSystem],
    seed: int | None,
    approval_threshold: float | None,
) -> tuple[dict, list[dict]]:
    rng = np.random.default_rng(seed)
    config, electorate, candidates = load_scenario(scenario_path, rng=rng)
    ballots = BallotProfile.from_preferences(electorate, candidates, approval_threshold)

    rows: list[dict] = []
    for system in systems:
        result = system.run(ballots, candidates)
        metrics = compute_metrics(result, electorate, candidates)
        row = {
            "system_name": metrics.system_name,
            "winner_labels": _winner_labels(result, candidates),
            "winner_indices": result.winner_indices,
            "is_pr": metrics.is_pr,
            "outcome_position": np.asarray(result.outcome_position).round(6).tolist(),
            "seat_shares": {candidates.labels[idx]: share for idx, share in result.seat_shares.items()},
        }
        for field in METRIC_FIELDS:
            row[field] = float(getattr(metrics, field))
        rows.append(row)

    return config, rows


def _sort_rows(rows: list[dict], sort_by: str, descending: bool) -> list[dict]:
    if sort_by not in SORTABLE_FIELDS:
        raise ValueError(f"Unknown sort field: {sort_by}")
    return sorted(rows, key=lambda row: row[sort_by], reverse=descending)


def _run_command(args: argparse.Namespace) -> int:
    rng = np.random.default_rng(args.seed)
    systems = _build_systems(args, rng=rng)
    _, rows = _collect_rows(
        scenario_path=args.scenario,
        systems=systems,
        seed=args.seed,
        approval_threshold=args.approval_threshold,
    )
    rows = _sort_rows(rows, sort_by=args.sort_by, descending=args.descending)
    _emit_rows(rows, args.format)
    return 0


def _list_systems_command(args: argparse.Namespace) -> int:
    rows = []
    for key in sorted(SYSTEM_REGISTRY):
        rows.append(
            {
                "key": key,
                "name": SYSTEM_REGISTRY[key]().name,
            }
        )

    rows.extend(
        [
            {"key": "fractional_discrete", "name": FractionalBallotDiscrete().name},
            {"key": "fractional_continuous", "name": FractionalBallotContinuous().name},
        ]
    )

    if args.format == "json":
        print(json.dumps(rows, indent=2, sort_keys=True))
    else:
        key_width = max(len("Key"), *(len(row["key"]) for row in rows))
        name_width = max(len("Name"), *(len(row["name"]) for row in rows))
        print(_tabular_line(["Key", "Name"], [key_width, name_width]))
        print(_tabular_line(["-" * key_width, "-" * name_width], [key_width, name_width]))
        for row in rows:
            print(_tabular_line([row["key"], row["name"]], [key_width, name_width]))
    return 0


def _list_scenarios_command(args: argparse.Namespace) -> int:
    if args.scenarios_dir:
        scenarios_dir = Path(args.scenarios_dir)
        paths = sorted(scenarios_dir.glob("*.yaml"))
        rows = [{"path": str(path), "name": path.stem} for path in paths]
    else:
        paths = built_in_scenario_paths()
        rows = [{"path": path.name, "name": path.stem} for path in paths]

    if args.format == "json":
        print(json.dumps(rows, indent=2, sort_keys=True))
    else:
        for row in rows:
            print(f"{row['name']}: {row['path']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="electoral-sim",
        description="Run electoral system simulations from the command line.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a scenario and print metrics.")
    run_parser.add_argument("scenario", help="Path to a scenario YAML file.")
    run_parser.add_argument(
        "--system",
        action="append",
        help="System key to run. May be repeated or comma-separated. Defaults to all standard systems.",
    )
    run_parser.add_argument(
        "--include-fractional",
        action="store_true",
        help="Also include the standard fractional ballot benchmark systems.",
    )
    run_parser.add_argument(
        "--fractional-variant",
        choices=["discrete", "continuous", "both"],
        default="both",
        help="Fractional ballot variant(s) to include when --include-fractional is set.",
    )
    run_parser.add_argument(
        "--fractional-sigmas",
        help="Comma-separated sigma values for fractional ballot systems. Defaults to 0.1,0.3,1.0.",
    )
    run_parser.add_argument(
        "--approval-threshold",
        type=float,
        help="Override the approval-voting distance threshold.",
    )
    run_parser.add_argument("--seed", type=int, help="Random seed for scenario generation.")
    run_parser.add_argument(
        "--format",
        choices=["table", "json", "csv"],
        default="table",
        help="Output format.",
    )
    run_parser.add_argument(
        "--sort-by",
        choices=SORTABLE_FIELDS,
        default="distance_to_median",
        help="Field to sort the results by.",
    )
    run_parser.add_argument(
        "--descending",
        action="store_true",
        help="Sort in descending order instead of ascending order.",
    )
    run_parser.set_defaults(func=_run_command)

    list_systems_parser = subparsers.add_parser("list-systems", help="List available system keys.")
    list_systems_parser.add_argument(
        "--format",
        choices=["table", "json"],
        default="table",
        help="Output format.",
    )
    list_systems_parser.set_defaults(func=_list_systems_command)

    list_scenarios_parser = subparsers.add_parser(
        "list-scenarios",
        help="List scenario YAML files in a directory.",
    )
    list_scenarios_parser.add_argument(
        "scenarios_dir",
        nargs="?",
        help=(
            "Directory containing scenario YAML files. "
            "If omitted, lists the built-in packaged scenarios."
        ),
    )
    list_scenarios_parser.add_argument(
        "--format",
        choices=["table", "json"],
        default="table",
        help="Output format.",
    )
    list_scenarios_parser.set_defaults(func=_list_scenarios_command)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
