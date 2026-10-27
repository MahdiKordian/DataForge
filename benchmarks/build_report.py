# Copyright (c) 2026 Mahdi Kordian. Licensed under the MIT License; see LICENSE.
"""Builds docs/BENCHMARK.md from the CSV files in benchmarks/results/."""
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd  # noqa: E402

from dataforge import benchmark as BM  # noqa: E402

R = ROOT / "benchmarks" / "results"
env = BM.environment()
md = [f"# DataForge benchmark\n\nRun on {date.today()}. Python {env['python']}, pandas {env['pandas']}, numpy {env['numpy']}, "
      f"{env['cpu count']} CPU(s), {env['platform']}.\n",
      "Numbers depend on the machine. Compare the *shape* (how time grows with size), not the absolute seconds.\n"]
if (R / "correctness.csv").exists():
    c = pd.read_csv(R / "correctness.csv")
    md += ["## 1. Correctness against known answers\n", c.assign(**{"pass": c["pass"].map({True: "yes", False: "NO"})}).to_markdown(index=False),
           f"\n{int(c['pass'].sum())} of {len(c)} checks passed.\n"]
if (R / "speed.csv").exists():
    s = pd.read_csv(R / "speed.csv")
    md += ["## 2. Speed (seconds) by data size\n", s.pivot_table(index="task", columns=["rows", "columns"], values="seconds").round(2).to_markdown(),
           "\n## 3. Peak extra memory (MB) by data size\n",
           "Measured with `tracemalloc` (Python and NumPy allocations); an estimate, not the process total.\n",
           s.pivot_table(index="task", columns=["rows", "columns"], values="peak MB").round(0).to_markdown()]
if (R / "scaling.csv").exists():
    md += ["\n## 4. How time grows with the number of rows (20 columns)\n",
           "A slope near 1 means time doubles when rows double. Much above 1 would be a problem at scale; below 1 means a fixed cost dominates.\n",
           pd.read_csv(R / "scaling.csv").to_markdown(index=False)]
if (R / "tools.csv").exists():
    md += ["\n## 5. Other tools, same data, fresh process each\n",
           "Tools do different work and produce different outputs, so this is 'time until a complete analysis exists', not identical work. "
           "Memory is the process's peak above its own start, after imports. Empty cells with a note mean the tool failed or timed out.\n",
           pd.read_csv(R / "tools.csv").fillna("").to_markdown(index=False)]
else:
    md += ["\n## 5. Other tools\n\nNot run. Use `python benchmarks/run_tools.py` (needs ydata-profiling and sweetviz installed).\n"]
(ROOT / "docs" / "BENCHMARK.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("written docs/BENCHMARK.md")
