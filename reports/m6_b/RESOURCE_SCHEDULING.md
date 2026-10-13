# Resource scheduling ledger (not an outcome/protocol amendment)

Initially GPU1 had an unrelated ongoing CUDA job and was not used. GPU0 is used
by S1 and S3, with unrelated concurrent jobs left untouched.

On2026-10-13 at11:50 China time, GPU1's previous jobs had exited. NVIDIA-SMI
showed utilization0 and only14MiB display context. Its physical model is the same
RTX A6000 as GPU0. After checking again, S2 is scheduled on free physical GPU1
(`CUDA_VISIBLE_DEVICES=1`, model's `cuda:0` maps to that card). No unrelated
process is killed, no legacy environment is changed and no CUDA mismatch bypass.

This changes only wall-clock resource scheduling, not any locked metric, seed,
sample, optimizer,60-epoch budget, checkpoint criterion or postprocessing choice.
S1/S2 use the same official model and labels. Four training-source hashes remain
unchanged. The CLI receives per-run advisory file locks to prevent duplicate
training when the original GPU0 matrix later reaches already running/completed
S2 runs; completed epoch60 resumes take zero additional updates. Logs with
`_gpu1` suffix preserve the actual S2 console history.

Inference timing during contention is descriptive, not a controlled architecture
speed contest. Final isolated same-device timing is measured separately if feasible.
Both cards are shared host resources; no exclusive reservation is claimed.

At14:28 on2026-10-13 an unrelated external Gaussian-splatting job had returned
to GPU1. S2 slowed to roughly430-470seconds/epoch while GPU0 became available
after S1/S3 training finished. Only our three S2 processes were stopped at their
saved epoch37 boundaries and terminated; the already waiting GPU0 matrix resumed
from those exact checkpoints. Each had3552 persisted updates, full model/Adam
state, torch/CUDA RNG state and epoch-specific data-loader seed. No partial
epoch updates are counted and no external process is signalled. Checkpoint
hashes and execution-time evidence are in `S2_device_migration.json`.
All60 epochs and5760 updates still apply; this is resource scheduling only.

Source: actual `nvidia-smi` output and execution logs, not assumed GPU availability.
