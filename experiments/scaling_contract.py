# -*- coding: utf-8 -*-
"""Current HAST scaling experiment contract."""

FULL_EVAL_SIZES = [500, 1000, 5000, 10000]
RUNTIME_ONLY_SIZES = [500, 1000, 5000, 10000, 50000, 100000, 1000000]
SEEDS = [42, 43, 44]
METHODS = ["CoreHD-fast", "HDA-fast", "HDA-original", "HAST-Final-S", "HAST-Final-Q"]


def main() -> None:
    print(
        {
            "full_eval_sizes": FULL_EVAL_SIZES,
            "runtime_only_sizes": RUNTIME_ONLY_SIZES,
            "seeds": SEEDS,
            "methods": METHODS,
        }
    )


if __name__ == "__main__":
    main()
