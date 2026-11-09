# Experiment control-xs-0001

Verdict: **inconclusive**

## Builds

- Baseline: `9876575c368de30fe1a12cc5d5763c9729441471` (`/Users/gregory.hansen/code/dbldatagen-profiler/.worktrees/p1-6a-wiring`)
- Candidate: `9876575c368de30fe1a12cc5d5763c9729441471` (`/Users/gregory.hansen/code/dbldatagen-profiler/.worktrees/p1-6a-wiring`)
- Harness: `9876575c368de30fe1a12cc5d5763c9729441471` hash `4f16a4ccc214`
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
  "experiment_id": "control-xs-0001",
  "failed_fidelity": 0,
  "failed_speed": 0,
  "fidelity": {
    "baseline/numeric": {
      "n": 2,
      "privacy_all_passed": true,
      "reconstruction_mean": 0.7196762048721892,
      "relative_mean": 0.7241868778285658,
      "structure_mean": 0.30769230769230776
    },
    "baseline/retail": {
      "n": 2,
      "privacy_all_passed": true,
      "reconstruction_mean": 0.5831704161766231,
      "relative_mean": 0.6289532531737989,
      "structure_mean": 0.28644688644688654
    },
    "candidate/numeric": {
      "n": 2,
      "privacy_all_passed": true,
      "reconstruction_mean": 0.7196762048721892,
      "relative_mean": 0.7241868778285658,
      "structure_mean": 0.30769230769230776
    },
    "candidate/retail": {
      "n": 2,
      "privacy_all_passed": true,
      "reconstruction_mean": 0.5831704161766231,
      "relative_mean": 0.6289532531737989,
      "structure_mean": 0.28644688644688654
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
    "baseline/numeric@xs": {
      "cpu_s_median": 1.6335435000000005,
      "n": 2,
      "peak_rss_bytes_median": 138067968.0,
      "profile_hashes": [
        "0fffde4a70a10fcd317396090ff0cb345417f06ea0a1dbc896777fb9e5f94873"
      ],
      "wall_s_median": 3.2156512500000645
    },
    "baseline/retail@xs": {
      "cpu_s_median": 0.716106,
      "n": 2,
      "peak_rss_bytes_median": 138354688.0,
      "profile_hashes": [
        "039799cd51c6736a229f402555679b0564f47e5b94a7b217263cd2ac99aef440"
      ],
      "wall_s_median": 1.8071740414998203
    },
    "candidate/numeric@xs": {
      "cpu_s_median": 1.7169165,
      "n": 2,
      "peak_rss_bytes_median": 139837440.0,
      "profile_hashes": [
        "0fffde4a70a10fcd317396090ff0cb345417f06ea0a1dbc896777fb9e5f94873"
      ],
      "wall_s_median": 3.397304562500267
    },
    "candidate/retail@xs": {
      "cpu_s_median": 0.6708724999999999,
      "n": 2,
      "peak_rss_bytes_median": 136781824.0,
      "profile_hashes": [
        "039799cd51c6736a229f402555679b0564f47e5b94a7b217263cd2ac99aef440"
      ],
      "wall_s_median": 1.5030653750000056
    }
  }
}
```

## Profile determinism

Deterministic: `True`
