from __future__ import annotations

import random
import unittest
from collections import Counter

from ns.generated import BASE_AND_GENERATED_TARGETS, NONLINEAR_TARGETS
from scripts.run_random import build_jobs, build_rule_plan, build_target_weights, case_key, target_family


class RandomExplorationTest(unittest.TestCase):
    def test_adaptive_jobs_cover_empty_cases_before_repeating(self) -> None:
        targets = [BASE_AND_GENERATED_TARGETS[name] for name in sorted(BASE_AND_GENERATED_TARGETS)[:2]]
        jobs = build_jobs(
            budget=2,
            targets=targets,
            depths=[2],
            widths=[10],
            rng=random.Random(7),
            mode="adaptive",
            dedupe_cases=True,
            existing_counts=Counter(),
        )

        keys = {case_key(job["target"].name, int(job["depth"]), int(job["width"])) for job in jobs}
        self.assertEqual(len(keys), 2)
        self.assertEqual([job["case_count_before"] for job in jobs], [0, 0])

    def test_adaptive_jobs_prefer_less_seen_cases(self) -> None:
        targets = [BASE_AND_GENERATED_TARGETS[name] for name in sorted(BASE_AND_GENERATED_TARGETS)[:2]]
        seen = Counter({case_key(targets[0].name, 2, 10): 4})

        jobs = build_jobs(
            budget=1,
            targets=targets,
            depths=[2],
            widths=[10],
            rng=random.Random(7),
            mode="adaptive",
            dedupe_cases=True,
            existing_counts=seen,
        )

        self.assertEqual(jobs[0]["target"].name, targets[1].name)
        self.assertEqual(jobs[0]["case_count_before"], 0)

    def test_frontier_policy_pushes_nonlinear_family(self) -> None:
        nonlinear = [NONLINEAR_TARGETS[name] for name in sorted(NONLINEAR_TARGETS)]
        critical = [
            BASE_AND_GENERATED_TARGETS[name]
            for name in sorted(BASE_AND_GENERATED_TARGETS)
            if target_family(name) == "critical_norm"
        ][:12]
        targets = nonlinear + critical

        jobs = build_jobs(
            budget=20,
            targets=targets,
            depths=[2],
            widths=[10],
            rng=random.Random(7),
            mode="adaptive",
            dedupe_cases=True,
            existing_counts=Counter(),
            target_policy="frontier",
            target_weights=build_target_weights(targets, {}),
        )

        families = Counter(str(job["target_family"]) for job in jobs)
        self.assertGreater(families["nonlinear"], families["critical_norm"])

    def test_sampled_rule_plan_keeps_nonlinear_core_rules(self) -> None:
        target = NONLINEAR_TARGETS["velocity_strain_gradient"]
        rules = build_rule_plan(target=target, rng=random.Random(3), policy="sample", min_rules=4)
        names = {rule.name for rule in rules}

        self.assertIn("Holder", names)
        self.assertIn("Biot-Savart", names)
        self.assertIn("Young", names)


if __name__ == "__main__":
    unittest.main()
