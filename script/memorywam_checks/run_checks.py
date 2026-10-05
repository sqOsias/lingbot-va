"""运行不加载 5B checkpoint 的完整 reference 检查。"""

from __future__ import annotations

import unittest


def main() -> None:
    suite = unittest.defaultTestLoader.discover(
        "script/memorywam_checks", pattern="test_*.py",
        top_level_dir=".")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
