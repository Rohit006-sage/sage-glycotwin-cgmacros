"""Run contract tests, train all baselines, and regenerate evaluation outputs."""

from __future__ import annotations

import unittest
import time


def main() -> None:
    started = time.perf_counter()
    suite = unittest.defaultTestLoader.discover("tests")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)

    from train_cgmacros_forecast import main as train
    from evaluate_cgmacros_forecast import main as evaluate

    train()
    evaluate()
    print(f"Full pipeline runtime (tests + training + evaluation): {time.perf_counter() - started:.2f} s")


if __name__ == "__main__":
    main()
