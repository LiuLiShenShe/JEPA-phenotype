# M6-B verification ledger

- `runtime_smoke.json`: original PlantNet and PSegNet class-adapted models execute
  batch4/4096-point forward, leaf/semantic joint loss, backward and Adam updates.
  Three-step finite losses and per-process GPU memory recorded.
- `tests_m6_b.log`: 11 focused tests pass; `tests_m6_ab.log`: 29 current/historical
  segmentation/tracking tests pass. Single synthetic MeanShift seed fallback warning.
- `tests_m6_b_decision.log`:17 focused tests pass after adding research-decision
  regression cases. A tracking-only failure cannot be called an instance-method
  defect; an audited first-observation instance omission is diagnosed separately.
- `tests_m6_ab_final.log`:35 final current/historical interface tests pass, with
  expected Matplotlib/pyparsing deprecation and synthetic MeanShift warnings.
- `tests_all.log`: initial isolated full suite gives 175 pass / 1 failure because
  this segmentation environment does not install historical RGB torchvision.
  Do not alter M1 code to fix environment-only failure.
- `test_m1_original_environment.log`: the exact failing historical M1 cache test
  passes in its unchanged `.venv-m1` environment (1 pass; expected xFormers warnings).
- `tracking_sampling_diagnostic.json`: failed sample equivalence detected and not
  ignored. Six earliest M6-A frames use original radius10; subsequent frames use
  adaptive radius from their recorded timing. Exact geometry/label equivalence
  must hold across the final384-frame cache.
- `tracking_geometry_equivalence.json`: all384-frame exact geometry/sample-label
  checks pass. `tracking_replay_equivalence.json`: identical original24-plant S0
  PQ, IDF1, ID switches, continuity, eventF1 and dynamic coverage/severe-rate replay
  after frame-local randomization; old results are not overwritten.
- `organ3d_pointnet2_build.log` and `organ3d_minkowski_build.log`: official builds
  actually fail at nvcc12.6 vs TorchCUDA11.3. Safe native mismatch is not bypassed;
  S4 does not have a successful original model example or target-training result.
- `protection_verification.json`: read-only checks of all historical artifact and
  environment size/mtime fingerprints, tracked SHA256, raw891-file SHA256 and HEAD.
- `final_verification.json`: completion checker requires all seven actual60-epoch
  runs, 5760 unique finite optimizer-step records each, Validation-only selection,
  full-resolution predictions for every legal point and error rows for every leaf.

Sources: executed test/build logs and actual data/source files. General Bocha
search is unavailable (HTTP403 quota); official source URLs in `SOURCE_AUDIT.md`.
