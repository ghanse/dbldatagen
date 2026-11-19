# Experiment baseline-s-0001

Verdict: **inconclusive**

## Builds

- Baseline: `da5d1e641ba2b53ecb301967fad6bc2d68a44cd5` (`<repo>`)
- Candidate: `da5d1e641ba2b53ecb301967fad6bc2d68a44cd5` (`<repo>`)
- Harness: `da5d1e641ba2b53ecb301967fad6bc2d68a44cd5` hash `cc84dbbcf5a6`
- Control run: `True`

## Measured

- Driver wall time, process CPU time, and peak RSS for `profile_tables`
- Profile JSON hash (created_at stripped) and cross-sample determinism
- Paired reconstruction / structure / privacy scores at the fidelity tier

## Not measured

- Spark executor / JVM heap and RSS (driver process RSS only)
- GPU / accelerator time
- network and disk IO bytes
- cluster-scale Spark (local[*] only)
- end-to-end dbldatagen generation from the profile

## Gates

- `sample_completeness`: pass (all_samples_ok)
- `privacy`: pass (privacy_passed)
- `control`: inconclusive (control)
- `accuracy_mean`: inconclusive (accuracy_crossing_interval)
- `accuracy_per_dataset`: pass (no_reconstruction_regression)
- `structure`: pass (no_structure_regression)

## Summary

```json
{
  "control": true,
  "experiment_id": "baseline-s-0001",
  "failed_fidelity": 0,
  "failed_speed": 0,
  "fidelity": {
    "baseline/edge": {
      "n": 5,
      "privacy_all_passed": true,
      "reconstruction_mean": 0.8718437198223816,
      "relative_mean": 0.8726791994539671,
      "structure_mean": 0.3666666666666667
    },
    "baseline/nested": {
      "n": 5,
      "privacy_all_passed": true,
      "reconstruction_mean": 0.6960525912197533,
      "relative_mean": 0.7000324875823938,
      "structure_mean": 0.507936507936508
    },
    "baseline/numeric": {
      "n": 5,
      "privacy_all_passed": true,
      "reconstruction_mean": 0.7699717796389316,
      "relative_mean": 0.7741841622590402,
      "structure_mean": 0.30769230769230776
    },
    "baseline/retail": {
      "n": 5,
      "privacy_all_passed": true,
      "reconstruction_mean": 0.6150949594281927,
      "relative_mean": 0.6534991970637535,
      "structure_mean": 0.28644688644688654
    },
    "baseline/strings": {
      "n": 5,
      "privacy_all_passed": true,
      "reconstruction_mean": 0.5998402749647594,
      "relative_mean": 0.6011075961503781,
      "structure_mean": 0.23529411764705882
    },
    "baseline/temporal": {
      "n": 5,
      "privacy_all_passed": true,
      "reconstruction_mean": 0.7639103152137162,
      "relative_mean": 0.7660489797988699,
      "structure_mean": 0.33333333333333337
    },
    "candidate/edge": {
      "n": 5,
      "privacy_all_passed": true,
      "reconstruction_mean": 0.8718437198223816,
      "relative_mean": 0.8726791994539671,
      "structure_mean": 0.3666666666666667
    },
    "candidate/nested": {
      "n": 5,
      "privacy_all_passed": true,
      "reconstruction_mean": 0.6960525912197533,
      "relative_mean": 0.7000324875823938,
      "structure_mean": 0.507936507936508
    },
    "candidate/numeric": {
      "n": 5,
      "privacy_all_passed": true,
      "reconstruction_mean": 0.7699717796389316,
      "relative_mean": 0.7741841622590402,
      "structure_mean": 0.30769230769230776
    },
    "candidate/retail": {
      "n": 5,
      "privacy_all_passed": true,
      "reconstruction_mean": 0.6150949594281927,
      "relative_mean": 0.6534991970637535,
      "structure_mean": 0.28644688644688654
    },
    "candidate/strings": {
      "n": 5,
      "privacy_all_passed": true,
      "reconstruction_mean": 0.5998402749647594,
      "relative_mean": 0.6011075961503781,
      "structure_mean": 0.23529411764705882
    },
    "candidate/temporal": {
      "n": 5,
      "privacy_all_passed": true,
      "reconstruction_mean": 0.7639103152137162,
      "relative_mean": 0.7660489797988699,
      "structure_mean": 0.33333333333333337
    }
  },
  "not_measured": [
    "Spark executor / JVM heap and RSS (driver process RSS only)",
    "GPU / accelerator time",
    "network and disk IO bytes",
    "cluster-scale Spark (local[*] only)",
    "end-to-end dbldatagen generation from the profile"
  ],
  "profile_determinism": {
    "deterministic": true,
    "divergent": {}
  },
  "speed": {
    "baseline/edge@s": {
      "cpu_s_median": 2.5700160000000007,
      "n": 5,
      "peak_rss_bytes_median": 287916032.0,
      "profile_hashes": [
        "108c5e34103663dcf352d923c17ebc6cd69af45a006674c13889c65e4e4dd13c"
      ],
      "wall_s_median": 7.8510740420315415
    },
    "baseline/nested@s": {
      "cpu_s_median": 47.101023,
      "n": 5,
      "peak_rss_bytes_median": 411123712.0,
      "profile_hashes": [
        "f0b238d1963e4cb2998fdd4a4d2a7d49edd3eff7ad9bf66b38832b8a8f20a82c"
      ],
      "wall_s_median": 191.43776737502776
    },
    "baseline/numeric@s": {
      "cpu_s_median": 3.169783,
      "n": 5,
      "peak_rss_bytes_median": 274923520.0,
      "profile_hashes": [
        "96bdf2df4cf7405abdff6450e01cc398616c3bc4c15251add205f1a00a6ccfe3"
      ],
      "wall_s_median": 15.281072916986886
    },
    "baseline/retail@s": {
      "cpu_s_median": 1.9132299999999995,
      "n": 5,
      "peak_rss_bytes_median": 251871232.0,
      "profile_hashes": [
        "46bfdfcfb000d2d9e3140008ac90d999aade022c29b7a422981ef0b69fca4744"
      ],
      "wall_s_median": 3.5390958749921992
    },
    "baseline/strings@s": {
      "cpu_s_median": 3.2972480000000006,
      "n": 5,
      "peak_rss_bytes_median": 344670208.0,
      "profile_hashes": [
        "2dc175a1bc790f63a70e3036fa48addb9f4555f4c88f0a66682f9441de7a7b50"
      ],
      "wall_s_median": 6.868172042013612
    },
    "baseline/temporal@s": {
      "cpu_s_median": 1.4361899999999999,
      "n": 5,
      "peak_rss_bytes_median": 248119296.0,
      "profile_hashes": [
        "e8dd7a0df77d55c8d543ddd680d204e5e299ba753e634aa8443a81a693d995fd"
      ],
      "wall_s_median": 3.5086244590347633
    },
    "candidate/edge@s": {
      "cpu_s_median": 2.631965,
      "n": 5,
      "peak_rss_bytes_median": 285786112.0,
      "profile_hashes": [
        "108c5e34103663dcf352d923c17ebc6cd69af45a006674c13889c65e4e4dd13c"
      ],
      "wall_s_median": 7.698400124965701
    },
    "candidate/nested@s": {
      "cpu_s_median": 48.14069,
      "n": 5,
      "peak_rss_bytes_median": 410386432.0,
      "profile_hashes": [
        "f0b238d1963e4cb2998fdd4a4d2a7d49edd3eff7ad9bf66b38832b8a8f20a82c"
      ],
      "wall_s_median": 192.99713170801988
    },
    "candidate/numeric@s": {
      "cpu_s_median": 3.1564999999999994,
      "n": 5,
      "peak_rss_bytes_median": 275775488.0,
      "profile_hashes": [
        "96bdf2df4cf7405abdff6450e01cc398616c3bc4c15251add205f1a00a6ccfe3"
      ],
      "wall_s_median": 15.267173208994791
    },
    "candidate/retail@s": {
      "cpu_s_median": 1.9034169999999997,
      "n": 5,
      "peak_rss_bytes_median": 252166144.0,
      "profile_hashes": [
        "46bfdfcfb000d2d9e3140008ac90d999aade022c29b7a422981ef0b69fca4744"
      ],
      "wall_s_median": 3.4202016670024022
    },
    "candidate/strings@s": {
      "cpu_s_median": 3.2990120000000003,
      "n": 5,
      "peak_rss_bytes_median": 345128960.0,
      "profile_hashes": [
        "2dc175a1bc790f63a70e3036fa48addb9f4555f4c88f0a66682f9441de7a7b50"
      ],
      "wall_s_median": 6.636069083004259
    },
    "candidate/temporal@s": {
      "cpu_s_median": 1.4358729999999995,
      "n": 5,
      "peak_rss_bytes_median": 248627200.0,
      "profile_hashes": [
        "e8dd7a0df77d55c8d543ddd680d204e5e299ba753e634aa8443a81a693d995fd"
      ],
      "wall_s_median": 3.48251912498381
    }
  }
}
```

## Profile determinism

Deterministic: `True`
