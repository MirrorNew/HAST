# -*- coding: utf-8 -*-
"""Current HAST motivation-observation experiment contract."""

OBSERVATION_2_GROUPS = ["R/GCC-only", "Absolute-cNBI", "Relative-Delta-cNBI"]
OBSERVATION_3_GROUPS = ["Relative-Free", "CostAware-Free", "Bounded-Guided"]
CANDIDATES_PER_GROUP = 100


def main() -> None:
    print(
        {
            "Observation 2": OBSERVATION_2_GROUPS,
            "Observation 3": OBSERVATION_3_GROUPS,
            "candidates_per_group": CANDIDATES_PER_GROUP,
        }
    )


if __name__ == "__main__":
    main()
