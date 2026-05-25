# -*- coding: utf-8 -*-
"""Run HAST motivation experiments E1-E3."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.motivation_observation_experiments import main


if __name__ == "__main__":
    main()
