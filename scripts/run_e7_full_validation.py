# -*- coding: utf-8 -*-
"""Compatibility wrapper for E7 full validation.

E7 evaluates E6-frozen HAST-Final-Q/S only and does not reselect candidates.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.full_validation import main


if __name__ == "__main__":
    main()
