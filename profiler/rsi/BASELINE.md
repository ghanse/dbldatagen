# Frozen profiler baseline

This baseline compares the profiler with itself. `baseline-s-0001` is the six-dataset tier-s control; its verdict is
**inconclusive**, as required for a control. `baseline-m-speed-0001` used the contract's 90-minute fallback and measured
retail, numeric, and nested; its verdict is **rejected** because nested did not complete. Privacy passed in both runs.

## Identity

| Item | Frozen value |
|---|---|
| Profiler commit | `da5d1e641ba2b53ecb301967fad6bc2d68a44cd5` |
| `profiler/src` SHA-256 | `d02bb9fe3f6177e6e2bc1f6c17284cba672fe11f0d37f22bc6c1657bb8f7881c` |
| Harness commit | `da5d1e641ba2b53ecb301967fad6bc2d68a44cd5` |
| Harness SHA-256 | `cc84dbbcf5a631dfbfac570c489f9639a868ad7bdeac3bae6f076847f37506c5` |
| Metrics config | version `1.8.0`; canonical SHA-256 `768c6bd24a4e1a749c29a40ab49c8e478533feadd681fca184bb022c576f40b8`; file SHA-256 `5fbc62012d271b69c3f575bc7b55ae8824fe7ae7dab601aecc14919168de6a3d` |
| Policy SHA-256 | `1a0f1225a27733fb52a301612c80b73849cf3f52c2b82c3e4eadb930a325e5cd` |

Machine: Apple M2 Max, 12 physical / 12 logical cores, 64 GiB RAM; macOS 26.7.2 (25G332), Python 3.10.20,
OpenJDK 17.0.16, and Spark/PySpark 4.1.1. Speed used `local[2]`, a 2 GiB driver, and 2 shuffle partitions. Fidelity
workers used `local[2]` with an 8 GiB driver.

## Tier-s fidelity

Values are means over five sampler seeds; brackets show min-max. Privacy is passing seeds / seeds measured.

| Dataset | Reconstruction | Ceiling | Relative to ceiling | Structure | Privacy | Three weakest reconstruction groups |
|---|---:|---:|---:|---:|---:|---|
| retail | 0.615095 [0.613446-0.616419] | 0.937465 | 0.653499 [0.650407-0.655510] | 0.286447 | 5/5 | discriminator 0.010813; constraints 0.468955; foreign_keys 0.524638 |
| numeric | 0.769972 [0.767434-0.770775] | 0.990435 | 0.774184 [0.771772-0.774951] | 0.307692 | 5/5 | discriminator 0.026964; dependence_declared 0.497691; keys 0.631934 |
| strings | 0.599840 [0.599039-0.600405] | 0.997216 | 0.601108 [0.600305-0.601675] | 0.235294 | 5/5 | discriminator 0; constraints 0.037698; marginal 0.593040 |
| temporal | 0.763910 [0.760911-0.767279] | 0.981648 | 0.766049 [0.763036-0.769433] | 0.333333 | 5/5 | discriminator 0.000010; dependence_random 0.498930; constraints 0.583138 |
| nested | 0.696053 [0.695776-0.696381] | 0.987673 | 0.700032 [0.699752-0.700425] | 0.507937 | 5/5 | discriminator 0; dependence_declared 0.594756; keys 0.631424 |
| edge | 0.871844 [0.870987-0.872543] | 0.997108 | 0.872679 [0.871820-0.873382] | 0.366667 | 5/5 | discriminator 0; marginal 0.733345; keys 0.815753 |

## Profiling speed

Values are baseline-role medians over five fresh-process samples; brackets show min-max. Peak RSS measures the driver
process, not Spark JVM/executor memory.

### Tier s

| Dataset | Wall seconds | CPU seconds | Peak RSS MiB |
|---|---:|---:|---:|
| retail | 3.539 [3.404-3.832] | 1.913 [1.887-1.997] | 240.2 [238.7-270.7] |
| numeric | 15.281 [15.088-16.175] | 3.170 [3.136-3.187] | 262.2 [260.5-263.7] |
| strings | 6.868 [6.557-7.126] | 3.297 [3.287-3.366] | 328.7 [326.2-387.2] |
| temporal | 3.509 [3.488-3.540] | 1.436 [1.430-1.449] | 236.6 [235.4-237.6] |
| nested | 191.438 [187.196-193.009] | 47.101 [46.340-48.608] | 392.1 [391.4-451.5] |
| edge | 7.851 [7.670-7.955] | 2.570 [2.540-2.630] | 274.6 [268.1-275.6] |

### Tier m

| Dataset | Wall seconds | CPU seconds | Peak RSS MiB |
|---|---:|---:|---:|
| retail | 9.057 [8.840-9.151] | 2.701 [2.644-2.703] | 666.3 [658.0-954.0] |
| numeric | 239.572 [238.323-243.354] | 3.169 [3.148-3.195] | 905.3 [886.5-1317.4] |
| nested | failed (0/5 baseline; 0/10 across both control roles) | - | - |
| strings | not measured by the fallback run | - | - |
| temporal | not measured by the fallback run | - | - |
| edge | not measured by the fallback run | - | - |

## Noise floor

The paired tier-s control has an accuracy-mean 95% CI of [0, 0], width 0. Every per-dataset reconstruction-difference
CI is also [0, 0], width 0. The policy's +0.005 accuracy threshold is therefore above the observed accuracy noise.

Wall-time intervals are paired-bootstrap 95% CIs for fractional improvement, `(baseline - candidate) / baseline`,
using 2,000 draws. Percentage-point width is the policy-relevant noise measure; raw-seconds width is included for scale.

| Dataset | Fractional improvement CI | CI width | Raw-seconds width |
|---|---:|---:|---:|
| retail | [-15.23%, +10.76%] | 25.99 pp | 0.951 s |
| numeric | [-2.07%, +5.61%] | 7.69 pp | 1.221 s |
| strings | [-2.65%, +6.88%] | 9.53 pp | 0.664 s |
| temporal | [-1.01%, +2.08%] | 3.09 pp | 0.109 s |
| nested | [-3.83%, +2.55%] | 6.38 pp | 12.093 s |
| edge | [-2.10%, +1.94%] | 4.05 pp | 0.320 s |

The 5% speed threshold exceeds the observed CI width only for temporal and edge. It is below the measured width for
retail, numeric, strings, and nested, so it is not uniformly above the observed speed noise floor. The policy is
unchanged.

## Known baseline failures

At the frozen 2 GiB / `local[2]` speed configuration, nested tier m failed 10/10 samples: 3 Java heap OOM errors and
7 SparkContext shutdown errors (5 baseline-role and 5 candidate-role samples total). Tier-m speed comparisons exclude
nested until a profiler change makes it complete. Making nested tier m complete at the frozen configuration is the top
speed proposal.

## Where the headroom is

Gaps are ceiling minus score, ranked across dataset/group means over five seeds.

1. strings / discriminator: 0.991584
2. edge / discriminator: 0.971224
3. strings / constraints: 0.962302
4. nested / discriminator: 0.950961
5. numeric / discriminator: 0.929350
6. temporal / discriminator: 0.871837
7. retail / discriminator: 0.569404
8. retail / constraints: 0.531045
9. numeric / dependence_declared: 0.498540
10. temporal / dependence_random: 0.496664
11. retail / foreign_keys: 0.470417
12. retail / keys: 0.427359
13. temporal / constraints: 0.416862
14. strings / marginal: 0.403325
15. nested / dependence_declared: 0.394305

Discriminator fidelity dominates the headroom for five of six datasets. Retail structure and strings constraints are
the next major accuracy targets; nested tier-m completion is the leading speed target.
