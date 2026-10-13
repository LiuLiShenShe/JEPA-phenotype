# Validation evidence

- M6-A tests: **18 passed**, `tests_m6_a.log` and JUnit `tests_m6_a.xml`.
- Entire repository in isolated M6 environment: **164 passed, 1 failed**, `tests_all.log/xml`. Sole failure: old M1 DINOv2 encoder test imports torchvision, absent from M6's minimal 3D environment. No old test or code was changed.
- The same failed old test rerun in its original `.venv-m1`: **1 passed**, `tests_m1_original_environment.log/xml`.
- Official PlantNet/PSegNet strict weight load and4096-point inference: successful, exact repeated labels/confidences (`official_runtime_smoke.json`).
- Real early/late scan replays (plant001/DAS37, plant046/DAS93): full-resolution labels, confidences and4096-point samples exactly repeat; 256-point sampled current GT masks also repeat (`final_verification.json`).
- Synthetic tests: PQ/SBD exact masks/merges/background contamination, detected/missed identity accounting, ID-switch continuity, false rebirth, missed vs wrongly attached new organs, sampling loss, missing morphology penalty, geometry units, distance gate and empty frame, GT-ID permutation invariance, label-bearing segmenter input rejection, plant bootstrap, fixed split/hash, wrong-link phenotype propagation and perfect-static/broken-dynamic identity loss.
- Complete delivery verification:384 full-resolution frames,24 plants,96 per-plant pipeline rows,1536 frame/pipeline rows, metric bounds, identical B1/B2 masks, unique observation/measurement joins and additive-loss closure.
- Historical protection:13,320 file size/mtime fingerprints;833 tracked files and891 Sugar4D raw files additionally SHA256-verified; no changes, same Git HEAD. Full old checkpoint byte hashes are **not** all recalculated; their size/mtime fingerprints are explicitly the protection method.
- Figure source preflight:11 pass,0 fail,3 accepted report-export warnings; direct PNG image inspection documented in `FIGURE_QA.md`.

No test result is substituted for empirical segmentation/tracking success. The primary baseline acceptance remains failed.
