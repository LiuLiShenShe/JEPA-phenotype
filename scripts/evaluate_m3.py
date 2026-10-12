#!/usr/bin/env python
"""Evaluate saved official and four-fold Sugar4D experiments without retraining."""

from __future__ import annotations

import argparse

from finalize_m3 import oracle_predictions, run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensitivity", action="store_true")
    run(parser.parse_args())
