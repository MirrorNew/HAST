# -*- coding: utf-8 -*-
"""Compatibility wrapper for the canonical paper plotting entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plotting.paper_figures import main


if __name__ == "__main__":
    main()
