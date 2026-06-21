#!/usr/bin/env python3
"""Unified ARGOS temporal-refiner training entrypoint.

This wrapper intentionally keeps the command short and stable while the
implementation lives in `train_temporal_refiner_fastcache.py`.
"""

from scripts.temporal_refinement.train_temporal_refiner_fastcache import main


if __name__ == "__main__":
    main()
