# M6-B execution plan

## Scope and dependencies
Build actual target-domain supervised official PlantNet/PSegNet baselines, then
reuse M6-A's full-resolution metrics, fixed TrackPlant3D and unchanged measurer.
No JEPA, temporal predictor, new segmentation architecture, commit, merge or push.
Initially available GPU isGPU0; GPU1 initially belongs to an unrelated job.
Later verified release permits S2 on identical GPU1, as recorded separately in
RESOURCE_SCHEDULING.md; no outcome/protocol change or unrelated process termination.

## Ordered checkpoints
1. Audit M6-A, official sources, licenses, raw labels and whole-plant splits.
   Files: `sources/`, `SOURCE_AUDIT.md`, `DATA_AUDIT.md`. Verify fixed commits,
   source/checkpoint hashes and 24/12/12 ×16 scans. Read-only historical snapshot.
2. Build independent `.venv-m6-b` and official forward/backward smoke tests.
   Files: environment lock, runtime evidence. Measure budget before outcome
   evaluation. Organ3DNet receives a bounded dependency/checkpoint investigation;
   a concrete blocker is reported without a replacement architecture.
3. Freeze evaluation and stopping criteria before supervised experiments.
   Files: `configs/m6_b/protocol.json`, `protocol_lock.json`. Validation selects
   checkpoint and postprocessing. Previously used Test is exploratory only.
4. Implement reversible current-frame-only data and official model/loss adapters.
   Files: `src/m6b/`, `scripts/run_m6_b.py`, `tests/test_m6_b.py`.
   Test local-ID permutation invariance, nonleaf masking, normalization inverse,
   split isolation, full-resolution scoring and checkpoint provenance.
5. Actually train S1/S2 with three identical-budget seeds and S3 if runnable.
   Files: ignored `outputs/m6_b/`, training logs and load/freeze manifests.
   Do not lengthen a run because a Test score is disappointing.
6. Evaluate full-resolution Validation and selected fixed exploratory Test.
   Report plant-macro PQ/SBD, errors, small/early/core-adjacent leaves and paired
   plant bootstrap intervals. Save full predictions and deterministic figures.
7. Feed independently reindexed predictions into unchanged TrackPlant3D.
   Reuse label-independent M6-A sampling/registration; preserve all missed organs.
   Same measurer and all-organ missing penalties; report IDF1/events/coverage.
8. Verify historical protection and delivery completeness; write M6_B_REPORT.md,
   reproduction commands, model feasibility records and exactly one evidence-based
   research recommendation. Explicitly disclose unfinished or blocked work.

## Risks and verification
- Bocha search returned HTTP403 quota exhausted; save response and use direct
  official retrieval. No claim that general web search succeeded.
- RTK reference `/home/test/.codex/RTK.md` is missing; do not invent its contents.
- Upstream code/checkpoint licenses are not explicitly declared. Do not publish
  copies or weights; local outputs are ignored.
- Official quadratic point losses may constrain training throughput. Report
  actual epochs, examples, optimizer steps, loss trajectory and budget adequacy;
  budget-limited inconclusive results must not prove a new method is needed.
- Organ3DNet requires old CUDA extensions and externally hosted checkpoint.
  Test actual dependencies; do not spend unlimited time porting or substitute it.
- Validation has only 12 independent plants; no scan-level confidence intervals.
- Original measurer's absolute length/area bias remains, even with better masks.

Timings are determined from runtime smoke tests, not invented estimates.
