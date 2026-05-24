# -*- coding: utf-8 -*-
"""Build paper source tables from consolidated local experiment records.

This entrypoint writes CSV tables under ``artifacts/source_tables``. It does
not draw figures; use ``plotting/paper_figures.py`` for visualization.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.sync_recorded_source_tables import main


if __name__ == "__main__":
    main()
