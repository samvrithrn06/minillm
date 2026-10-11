"""Dependency-free runner (so tests run even without pytest):  python -m tests.run_all"""
import importlib
import pkgutil
import sys
import traceback

import tests

failed = 0
for mod in pkgutil.iter_modules(tests.__path__):
    if not mod.name.startswith("test_"):
        continue
    m = importlib.import_module(f"tests.{mod.name}")
    for name in sorted(dir(m)):
        if name.startswith("test_"):
            try:
                getattr(m, name)()
                print("PASS", mod.name, name)
            except Exception:
                failed += 1
                print("FAIL", mod.name, name)
                traceback.print_exc()
sys.exit(1 if failed else 0)
