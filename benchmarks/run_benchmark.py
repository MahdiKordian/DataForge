# Copyright (c) 2026 Mahdi Kordian. Licensed under the MIT License; see LICENSE.
"""Run:  python benchmarks/run_benchmark.py [--quick] [--tools]
Writes CSV files to benchmarks/results/ and the report to docs/BENCHMARK.md."""
import argparse
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataforge import benchmark as BM  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true", help="smaller sizes")
ap.add_argument("--tools", action="store_true", help="also compare with ydata-profiling and Sweetviz (must be installed)")
a = ap.parse_args()
out = ROOT / "benchmarks" / "results"
out.mkdir(parents=True, exist_ok=True)

sizes = ((5_000, 20), (20_000, 20)) if a.quick else ((5_000, 20), (20_000, 20), (100_000, 20), (20_000, 100), (500_000, 20))
print("correctness...")
corr = BM.correctness()
corr.to_csv(out / "correctness.csv", index=False)
print("speed and memory...")
speed = BM.run_speed(sizes, measure_memory=True)
speed.to_csv(out / "speed.csv", index=False)
exps = BM.scaling_exponents(speed)
exps.to_csv(out / "scaling.csv", index=False)
tools = None
if a.tools:
    print("tools...")
    tools = BM.compare_tools(((5_000, 20), (20_000, 20)) if a.quick else ((5_000, 20), (20_000, 20), (100_000, 20)))
    tools.to_csv(out / "tools.csv", index=False)

env = BM.environment()
md = [f"# DataForge benchmark\n\nRun on {date.today()}. Python {env['python']}, pandas {env['pandas']}, numpy {env['numpy']}, "
      f"{env['cpu count']} CPU(s), {env['platform']}.\n",
      "Numbers depend on the machine. Compare the *shape* (how time grows with size), not the absolute seconds.\n",
      "## 1. Correctness against known answers\n", corr.assign(**{"pass": corr["pass"].map({True: "yes", False: "NO"})}).to_markdown(index=False),
      f"\n{int(corr['pass'].sum())} of {len(corr)} checks passed.\n",
      "## 2. Speed (seconds) by data size\n", speed.pivot_table(index="task", columns=["rows", "columns"], values="seconds").round(2).to_markdown(),
      "\n## 3. Peak extra memory (MB) by data size\n", speed.pivot_table(index="task", columns=["rows", "columns"], values="peak MB").round(0).to_markdown(),
      "\n## 4. How time grows with the number of rows (20 columns)\n",
      "A slope near 1 means time doubles when rows double. Much above 1 would be a problem at scale; below 1 means a fixed cost dominates.\n",
      exps.to_markdown(index=False)]
if tools is not None:
    md += ["\n## 5. Other tools, same data, fresh process each\n",
           "Tools do different work and produce different outputs, so this is 'time until a complete analysis exists', not identical work. "
           "Memory is the process's peak above its own start, after imports.\n",
           tools.fillna("").to_markdown(index=False)]
(ROOT / "docs").mkdir(exist_ok=True)
(ROOT / "docs" / "BENCHMARK.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("written docs/BENCHMARK.md")
