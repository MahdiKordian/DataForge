# Copyright (c) 2026 Mahdi Kordian. Licensed under the MIT License; see LICENSE.
"""Compare with ydata-profiling and Sweetviz, one (size, tool) pair at a time. Appends to benchmarks/results/tools.csv so a
long run can be stopped and resumed.   python benchmarks/run_tools.py [rows cols] [tool ...]"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd

from dataforge_engine import benchmark as BM

rows, cols = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (5_000, 20)
tools = tuple(sys.argv[3:]) or tuple(BM.TOOLS)
f = ROOT / "benchmarks" / "results" / "tools.csv"
new = BM.compare_tools(((rows, cols),), tools, timeout=1500)
old = pd.read_csv(f) if f.exists() else pd.DataFrame()
pd.concat([old, new], ignore_index=True).to_csv(f, index=False)
print(new.to_string())
