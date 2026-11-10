import csv
import json

from modelling.baselines import DemandProportionalPolicy, EqualStaticPolicy
from modelling.contracts import EnvironmentConfig
from modelling.envs.slicing_env import SafeSliceEnv
from modelling.experiments import paired_difference, run_paired_experiment


def test_experiment_report_is_paired_repeatable_and_writable(tmp_path) -> None:
    config = EnvironmentConfig(episode_steps=5)
    catalog = SafeSliceEnv(config).catalog
    policies = {"equal": EqualStaticPolicy(catalog), "demand": DemandProportionalPolicy(catalog)}
    report = run_paired_experiment(config, policies, [2, 3, 5], bootstrap_samples=100)
    repeat = run_paired_experiment(config, policies, [2, 3, 5], bootstrap_samples=100)
    assert report == repeat
    assert report.policies[0].metrics["total_reward"].sample_count == 3
    difference = paired_difference(report, "demand", "equal", "total_reward", bootstrap_samples=100)
    assert difference.sample_count == 3
    report.write(tmp_path)
    assert json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))["schema_version"] == 2
    with (tmp_path / "episodes.csv").open(encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 6
