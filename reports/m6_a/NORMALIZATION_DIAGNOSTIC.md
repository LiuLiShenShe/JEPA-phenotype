# Official segmentation normalization discrepancy

The original frozen B1 preprocessing uses current-scan mean centering and unit-sphere scaling. This was explicitly frozen, not selected using performance, and all original results are retained.

On 2026-10-12, direct raw downloads of H5 preprocessing repeatedly timed out; subsequent successful official GitHub **blob API** retrieval revealed `change_scale` centers the **sample's bounding box midpoint**, then divides by its largest positive centered coordinate. The official FPS and 3DEPS H5 converters have the same definition. Sources and failed probes are retained in `sources/`.

This is an upstream-fidelity limitation of the first baseline, not an excuse to tune a model. A separate, newly locked **engineering diagnostic** runs the same checkpoint on the same 4096 XYZ samples using the exact official normalization. It reuses the same geometry-only tracking samples/CPD and the same measurements. No primary metric, original split, threshold, weight, clustering bandwidth or historical result is altered.

The diagnostic configuration is `configs/m6_a/official_normalization_diagnostic.json`, hashed before its first inference in `official_normalization_lock.json`. It is **post-primary-lock** and cannot replace a failed original primary endpoint or establish confirmatory success. Report the original baseline and the corrected-fidelity diagnostic separately; if the bottleneck direction changes, state that the original method-level inference is not robust. Do not select a new best model.

Execution limitation, recorded before any diagnostic aggregate score: the planned repeated tracking part is not executed. Recreating all 256-point sample indices in the original cache would repeat the expensive official boundary search; instead the diagnostic is restricted to **full-resolution per-frame segmentation PQ/SBD and detection errors on all 24 held-out plants**. It does not produce a corrected-normalization tracking/phenotyping acceptance result. Original three-pipeline tracking comparison is unchanged. No main metric is replaced by PQ to declare success.
