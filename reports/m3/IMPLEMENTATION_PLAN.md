# M3 execution plan

1. Freeze the V4 Dataverse response, license, checksums and legacy-file SHA256 snapshot; download the real archive and verify every archive member before read-only extraction.
2. Inspect actual CSV/PLY schemas, reference measurements, genotype layout and official partitions; audit all 48 plants and 768 observations, organ identity, physical coordinates and annotation exceptions.
3. Build strict label-free XYZ temporal Dataset/DataLoader, deterministic physical-scale sampling cache, plant-disjoint official split and four genotype-balanced plant folds; lock tasks before training.
4. Implement and test oracle persistence/trend and shared lightweight PointNet single/GRU/repeated-latest models; train three seeds with identical schedules, plus plant folds and predeclared point-count sensitivity.
5. Evaluate each horizon using equal-plant MAE/RMSE, paired plant bootstrap, stage/genotype/first-observation analyses; export six requested figures and an evidence-grounded A–H report.
6. Run the complete test suite, verify raw and legacy hashes, issue engineering and research decisions separately, and stop before M4.

Risks: remote download/access; undocumented units/genotype mapping; missing or mismatched labels; limited independent plants; automatic morphology is not biological ground truth. Any blocking data failure is PARTIAL, never synthetic experimental success.

Tests: binary PLY decoding, temporal and split leakage, sample/scale determinism, future-label exclusion, model shape/gradient/control equivalence, unequal-window macro metrics and clustered comparisons; real-data smoke/integrity assertions after download.
