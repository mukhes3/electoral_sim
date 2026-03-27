import json

from electoral_sim.cli import main


SCENARIO_PATH = "configs/scenarios/02_polarized_bimodal.yaml"


def test_list_systems_json_includes_registry_entries(capsys):
    exit_code = main(["list-systems", "--format", "json"])

    assert exit_code == 0
    output = capsys.readouterr().out
    payload = json.loads(output)
    keys = {item["key"] for item in payload}
    assert "plurality" in keys
    assert "fractional_discrete" in keys


def test_run_command_json_outputs_results_for_selected_systems(capsys):
    exit_code = main(
        [
            "run",
            SCENARIO_PATH,
            "--system",
            "plurality,score",
            "--seed",
            "42",
            "--format",
            "json",
            "--sort-by",
            "system_name",
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    payload = json.loads(output)
    names = [row["system_name"] for row in payload]
    assert names == ["Plurality (FPTP)", "Score Voting"]
    assert all("winner_labels" in row for row in payload)


def test_run_command_table_output_works_with_fractional_systems(capsys):
    exit_code = main(
        [
            "run",
            SCENARIO_PATH,
            "--system",
            "plurality",
            "--include-fractional",
            "--fractional-variant",
            "discrete",
            "--fractional-sigmas",
            "0.3",
            "--seed",
            "42",
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Plurality (FPTP)" in output
    assert "Fractional Ballot Discrete" in output
