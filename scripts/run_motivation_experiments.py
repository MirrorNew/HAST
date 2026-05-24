# -*- coding: utf-8 -*-
"""Motivation experiment entrypoint skeleton.

Observation 2 and Observation 3 reuse the same candidate/evaluator APIs as the
main search. This script documents the executable group names used by the paper.
"""

OBSERVATION_2_GROUPS = ["GCC/R-only", "Absolute-cNBI", "Relative-Delta-cNBI"]
OBSERVATION_3_GROUPS = ["Relative-Free", "CostAware-Free", "Bounded-Guided"]


def main() -> None:
    print({"Observation 2": OBSERVATION_2_GROUPS, "Observation 3": OBSERVATION_3_GROUPS, "candidates_per_group": 100})


if __name__ == "__main__":
    main()
