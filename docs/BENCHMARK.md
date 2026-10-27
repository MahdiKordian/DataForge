# DataForge benchmark

Run on 2026-10-29. Python 3.12.3, pandas 2.3.3, numpy 2.3.5, 1 CPU(s), Linux-6.18.44-fc-v185-x86_64-with-glibc2.39.

Numbers depend on the machine. Compare the *shape* (how time grows with size), not the absolute seconds.

## 1. Correctness against known answers

| check                                            | expected          | measured   | pass   |
|:-------------------------------------------------|:------------------|:-----------|:-------|
| mean of N(5,2)                                   | 5.00 +/- 0.02     | 5.000      | yes    |
| std of N(5,2)                                    | 2.00 +/- 0.02     | 2.002      | yes    |
| skew of a normal                                 | 0 +/- 0.03        | -0.004     | yes    |
| Cramer's V, identical categories                 | close to 1        | 1.000      | yes    |
| Cramer's V, independent categories               | close to 0        | 0.028      | yes    |
| Pearson blind to a parabola                      | |r| < 0.1         | 0.003      | yes    |
| NMI sees the parabola                            | NMI > 0.3         | 0.645      | yes    |
| split sizes add up                               | 5000              | 5000       | yes    |
| class share kept by the split (stratified)       | difference < 1 pt | 0.06 pt    | yes    |
| imputer uses the TRAINING median                 | 76.3417           | 76.3417    | yes    |
| transform leaves its input unchanged             | identical         | identical  | yes    |
| train, validation and test have the same columns | True              | True       | yes    |
| same data and seed give identical output         | True              | True       | yes    |
| scaled training column has mean about 0          | |mean| < 0.05     | -0.0000    | yes    |
| no NaN left after preparation                    | 0                 | 0          | yes    |

15 of 15 checks passed.

## 2. Speed (seconds) by data size

| task                   |   (5000, 20) |   (20000, 20) |   (20000, 100) |   (100000, 20) |   (500000, 20) |
|:-----------------------|-------------:|--------------:|---------------:|---------------:|---------------:|
| compare two versions   |         0.1  |          0.46 |           2.2  |           0.89 |           4.48 |
| correlations           |         0.02 |          0.09 |           0.29 |           0.13 |           0.53 |
| missing analysis       |         0.46 |          0.79 |           2.18 |           0.72 |           1.55 |
| mixed associations     |         0.66 |          1.1  |           2.46 |           1.41 |           1.28 |
| pipeline fit+transform |         0.15 |          0.36 |           2.25 |           2.13 |          12.46 |
| profile categorical    |         0.03 |          0.03 |           0.13 |           0.12 |           0.53 |
| profile numeric        |         0.06 |          0.1  |           0.48 |           0.36 |           1.18 |
| quality checks         |         0.12 |          0.28 |           1.35 |           1.47 |           6.96 |
| split                  |         0    |          0.01 |           0.05 |           0.16 |           0.68 |

## 3. Peak extra memory (MB) by data size

Measured with `tracemalloc` (Python and NumPy allocations); an estimate, not the process total.

| task                   |   (5000, 20) |   (20000, 20) |   (20000, 100) |   (100000, 20) |   (500000, 20) |
|:-----------------------|-------------:|--------------:|---------------:|---------------:|---------------:|
| compare two versions   |            1 |             4 |             16 |             18 |             94 |
| correlations           |            2 |             9 |             26 |              9 |             20 |
| missing analysis       |            3 |             6 |             21 |             12 |             49 |
| mixed associations     |            1 |             4 |              5 |              8 |              8 |
| pipeline fit+transform |            3 |            15 |             66 |             84 |            420 |
| profile categorical    |            0 |             1 |              1 |              4 |             27 |
| profile numeric        |            0 |             1 |              1 |              5 |             31 |
| quality checks         |            2 |             7 |             32 |             36 |            184 |
| split                  |            2 |             8 |             34 |             40 |            198 |

## 4. How time grows with the number of rows (20 columns)

A slope near 1 means time doubles when rows double. Much above 1 would be a problem at scale; below 1 means a fixed cost dominates.

| task                   |   time ~ rows^ |   seconds at largest |
|:-----------------------|---------------:|---------------------:|
| split                  |           1.13 |             0.675305 |
| pipeline fit+transform |           0.98 |            12.4582   |
| quality checks         |           0.9  |             6.95934  |
| compare two versions   |           0.78 |             4.47619  |
| profile categorical    |           0.68 |             0.527007 |
| correlations           |           0.67 |             0.534886 |
| profile numeric        |           0.67 |             1.18156  |
| missing analysis       |           0.23 |             1.55444  |
| mixed associations     |           0.14 |             1.28354  |

## 5. Other tools, same data, fresh process each

Tools do different work and produce different outputs, so this is 'time until a complete analysis exists', not identical work. Memory is the process's peak above its own start, after imports. Empty cells with a note mean the tool failed or timed out.

|   rows |   columns | tool                                                      |   seconds |   peak_mb_above_start |
|-------:|----------:|:----------------------------------------------------------|----------:|----------------------:|
|   5000 |        20 | DataForge (profile, quality, relationships, missing data) |   1.09612 |               5.0625  |
|   5000 |        20 | ydata-profiling (minimal=True)                            |   5.68046 |              81.793   |
|   5000 |        20 | Sweetviz (with target)                                    |  10.6541  |             159.711   |
|   5000 |        20 | ydata-profiling (default)                                 |  21.8515  |             283.879   |
|  20000 |        20 | DataForge (profile, quality, relationships, missing data) |   2.06278 |               2.14844 |
|  20000 |        20 | ydata-profiling (minimal=True)                            |   5.12904 |              94.125   |
|  20000 |        20 | Sweetviz (with target)                                    |  11.3633  |             165.562   |
|  20000 |        20 | ydata-profiling (default)                                 |  21.3178  |             305.301   |
| 100000 |        20 | DataForge (profile, quality, relationships, missing data) |   3.72109 |               0       |
| 100000 |        20 | ydata-profiling (minimal=True)                            |   6.22037 |             170.34    |
