# Frozen profiler baseline (policy 2.0.0)

This baseline re-measures the profiler against itself under policy **2.0.0** at the 100k-row (`s`) tier only.
`baseline-v2-0002` is the six-dataset tier-s control with proposal type `speed` (N=15 fresh speed samples per
build and dataset, no baseline cache). Its verdict is **inconclusive**, as required for a control. Privacy passed
on every seed. The prior policy-1.x baseline is kept as `BASELINE-v1.md`. The 2M-row (`m`) tier was **not**
re-measured under policy 2.0.0 and will be measured only for speed proposals that target it.

## Identity

| Item | Frozen value |
|---|---|
| Profiler commit | `d49f687663bd7d3e57e34bbeb156b2a4f132c9e5` |
| `profiler/src` SHA-256 | `bcbf63a7b5868756fd988b06190c6dcdacc338c17e33fe8cb845edae6a5c3161` |
| Harness commit | `d49f687663bd7d3e57e34bbeb156b2a4f132c9e5` |
| Harness SHA-256 | `a29910e8bb919841f76d08d51fd72127d570653284b7a1e4cf155819cc4c6534` |
| Metrics config | version `1.9.0`; canonical SHA-256 `4a1b0f70821e3746992ccff286b6c444b90ca85829ed8b0d797a42cae1d81078`; file SHA-256 `846521dcf72bb2f576d3ac9af89b95c4fcbee5e6b49bc9c5e838e5a54bf2be69` |
| Policy SHA-256 | `bfcd33c433bbd05624326c338b8af4225c903915d52dc626f8a9ea9e5a5d623e` |

Machine: Apple M2 Max, 12 physical / 12 logical cores, 64 GiB RAM; macOS 26.7.2 (25G332), Python 3.10.20,
OpenJDK 17.0.16, and Spark/PySpark 4.1.1. Speed used `local[8]`, an 8 GiB driver, and 2 shuffle partitions.
Fidelity workers used `local[4]` with an 8 GiB driver, up to 3 cells in parallel. Experiment wall time for
`baseline-v2-0002`: about **24728 s** (6.87 h), as reported by the run's driver log; the committed artifacts do
not record start and end times.

This baseline is valid for the current accuracy gate. The planned discriminator-first gate (P1.6e) changes the
discriminator features and adds new metrics, so it needs another baseline refresh once it merges.

## Tier-s fidelity

Values are means over five sampler seeds; brackets show min-max. Privacy is passing seeds / seeds measured.

| Dataset | Reconstruction | Ceiling | Relative to ceiling | Structure | Privacy | Three weakest reconstruction groups |
|---|---:|---:|---:|---:|---:|---|
| retail | 0.626697 [0.625573-0.629207] | 0.937465 | 0.659719 [0.658657-0.661770] | 0.286447 | 5/5 | discriminator 0.019740; constraints 0.485447; foreign_keys 0.532969 |
| numeric | 0.861848 [0.855562-0.866435] | 0.990435 | 0.867684 [0.861150-0.872503] | 0.423077 | 5/5 | discriminator 0.299205; keys 0.631660; dependence_declared 0.785146 |
| strings | 0.600300 [0.599019-0.601119] | 0.997216 | 0.601570 [0.600286-0.602389] | 0.235294 | 5/5 | discriminator 0; constraints 0.037698; marginal 0.593448 |
| temporal | 0.765141 [0.762083-0.768386] | 0.981648 | 0.767285 [0.764214-0.770545] | 0.333333 | 5/5 | discriminator 0.000002; dependence_random 0.498140; constraints 0.583563 |
| nested | 0.722738 [0.721890-0.723661] | 0.987673 | 0.726803 [0.725934-0.727905] | 0.722222 | 5/5 | discriminator 0; keys 0.631400; marginal 0.764437 |
| edge | 0.882226 [0.881701-0.883302] | 0.997108 | 0.883099 [0.882579-0.884186] | 0.366667 | 5/5 | discriminator 0; marginal 0.760367; keys 0.815750 |

## Profiling speed

Values are baseline-role medians over fifteen fresh-process samples; brackets show min-max. Peak RSS measures the
driver process, not Spark JVM/executor memory. Config: `local[8]` / 8 GiB (policy 2.0.0).

### Tier s

| Dataset | Wall seconds | CPU seconds | Peak RSS MiB |
|---|---:|---:|---:|
| retail | 6.097 [5.490-12.215] | 2.068 [1.971-2.679] | 245.9 [242.7-246.7] |
| numeric | 25.420 [24.532-26.988] | 3.359 [3.291-3.434] | 268.3 [267.2-269.3] |
| strings | 9.091 [8.727-15.935] | 3.181 [3.065-4.038] | 333.3 [331.1-391.7] |
| temporal | 4.532 [4.409-4.707] | 1.479 [1.436-1.499] | 242.0 [240.4-243.5] |
| nested | 271.654 [265.516-302.330] | 48.463 [46.847-52.813] | 399.1 [392.8-449.6] |
| edge | 13.724 [13.250-14.658] | 2.853 [2.760-3.110] | 276.6 [272.0-284.0] |

### Tier m

Not re-measured under policy 2.0.0. Measure only for speed proposals that target the 2M-row tier.

## Noise floor

The paired tier-s control reports an accuracy-mean difference of exactly 0 on every dataset (every
reconstruction-difference CI is [0, 0]). This is not measured noise. Both control roles run the same profiler, so
they emit byte-identical profiles, and the sampler seeds are paired, so every difference is exactly 0 by construction.
The control shows only that the pipeline is deterministic. It says nothing about whether +0.005 sits above noise.

The real variability of an accuracy score is its seed-to-seed spread for a fixed profile (the min-max in the
reconstruction table above):

| Dataset | Seed-to-seed spread |
|---|---:|
| retail | 0.0036 |
| numeric | 0.0109 |
| strings | 0.0021 |
| temporal | 0.0063 |
| nested | 0.0018 |
| edge | 0.0016 |

Because candidate and baseline are scored on the same seeds, the paired gate cancels much of this spread. Even so,
numeric's and temporal's spreads meet or exceed the +0.005 threshold, so small accuracy gains on those datasets will
often come out inconclusive.

Wall-time intervals are paired-bootstrap 95% CIs for fractional improvement, `(baseline - candidate) / baseline`,
using 2,000 draws at N=15 speed samples per build. Percentage-point width is the policy-relevant noise measure;
raw-seconds width is included for scale.

| Dataset | Fractional improvement CI | CI width | Raw-seconds width |
|---|---|---:|---:|
| retail | [-5.76%, +11.67%] | 17.43 pp | 1.062 s |
| numeric | [-4.18%, +2.21%] | 6.39 pp | 1.624 s |
| strings | [-1.78%, +1.68%] | 3.47 pp | 0.315 s |
| temporal | [-0.78%, +2.65%] | 3.42 pp | 0.155 s |
| nested | [-4.57%, +4.03%] | 8.60 pp | 23.364 s |
| edge | [-3.08%, +1.69%] | 4.77 pp | 0.655 s |

The 5% speed threshold exceeds the observed CI width for strings, temporal, and edge. It is **still too wide** for
retail (17.43 pp), numeric (6.39 pp), and nested (8.60 pp): at N=15 under `local[8]`/8g, a true 5% speedup on those
three datasets can still land inconclusive.

## Where the headroom is

Gaps are ceiling minus score, ranked across dataset/group means over five seeds.

1. strings / discriminator: 0.991584
2. edge / discriminator: 0.971224
3. strings / constraints: 0.962302
4. nested / discriminator: 0.950961
5. temporal / discriminator: 0.871845
6. numeric / discriminator: 0.657109
7. retail / discriminator: 0.560476
8. retail / constraints: 0.514553
9. temporal / dependence_random: 0.497455
10. retail / foreign_keys: 0.462085
11. retail / keys: 0.428275
12. temporal / constraints: 0.416437
13. strings / marginal: 0.402918
14. nested / keys: 0.368600
15. numeric / keys: 0.368340

Discriminator fidelity dominates the headroom for five of six datasets. Retail structure/constraints and strings
constraints remain major accuracy targets. Nested wall time (~272 s median at tier s) is the leading speed target at
the measured tier; tier-m speed is deferred until a proposal targets it.

Nested is slower under this speed config: 271.7 s median wall time (48.5 s Python CPU) at `local[8]`/8 GiB, against
about 197-212 s wall in earlier `local[2]`/2 GiB runs. The samples show the slowdown but not its cause; the extra
wall time is not Python CPU, so per-job Spark overhead across nested's many small jobs is the likely explanation,
not yet confirmed.
