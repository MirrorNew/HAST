# Local Data Layout

This folder contains data needed for local reproduction.

- `fixtures/`: tiny tracked smoke-test graphs.
- `benchmarks/raw/`: local benchmark edgelists copied from the existing experiment environment. This folder is ignored by Git and should not be uploaded.
- `baseline_records/`: local baseline step records copied from the existing experiment environment. This folder is ignored by Git and should not be uploaded.
- `search_framework_records/raw/`: local raw historical search traces for generic search frameworks and early HAST-family experiments. This folder is ignored by Git and should not be uploaded.

The GitHub repository should keep this README and `fixtures/`, but not the raw benchmark data or baseline record cache.
