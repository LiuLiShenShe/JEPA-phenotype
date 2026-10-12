# M5-A-only budget amendment, before formal P3/P4

M5-A completed exactly 2000 optimizer steps using only official Train, seed 17.
No M5 downstream model or formal S500/P3/P4 had been trained at this point.

The original training-only stop condition required simultaneous loss of variance
and effective rank. It did **not** trigger: variance increased from step 500 to
2000 while pooled effective rank decreased from 8.544 to 1.525 and the leading
covariance eigenvalue captured 91.89% of pooled variance. Token rank was 1.561.
This is low-dimensional degeneration, not zero-variance collapse. The conjunction
stop rule was too narrow; it must not be retrospectively claimed to have worked.

One transparent pre-formal amendment fixes **500 steps** for every S500/P3/P4 run.
No further blind unregularized extension is permitted. The 2000 checkpoint is
retained as a diagnostic, not selected as a downstream initialization. S500 is
trained anew with a 500-step learning-rate/EMA horizon and the same uniform scan
sampling as the expanded runs; it is not the M5-A step500 checkpoint, which had a
2000-step schedule. The M5-A 120-step checkpoint is not an exact M4 replication
because the horizon and replacement sampler differ; M4 archived 120 steps remain
the exact historical control.

P4 is triggered by the **previously declared** M5-A risk rule at step500:
token std 0.05349 versus initial 0.31281 (17.1%), pooled rank 8.544 versus initial
20.709 (41.3%). Apply only the already written weights: variance=1.0,
covariance=0.01, std floor=0.1, epsilon=1e-4. The exact student context features
from the JEPA forward are regularized via a forward hook, avoiding a second
encoder forward or a second BatchNorm update. No loss/weight search and no
downstream-driven P4 tuning are allowed.

Physical GPU 0 and GPU 1 are available A6000 devices. Unlike the initial plan's
GPU-0-only assumption, disjoint partition jobs may use both to bound wall time.
Architecture/software/determinism and matched per-run budgets stay identical;
the work is not advertised as a hardware throughput benchmark. No unrelated
process is killed. No Temporal-JEPA, future prediction or Diffusion is run.
