# M1 commit record

- Branch: `m1/plant-2d-baseline`.
- M0 base: `6058704b4046eec77a5092871991ff9e146dddb4`.
- Implementation: `d90724cb4f509850a5f337208359484bb4f4ea1c` —
  `feat(m1): implement audited 2D temporal phenotype baselines`.
- Experiment/report delivery: `3c4277d` —
  `docs(m1): deliver complete experiments with PASS and REVIEW decisions`.
- Final provenance/whitespace cleanup is a subsequent commit named
  `chore(m1): finalize report provenance and whitespace`.
  Its exact ID is resolved by `git log -1 --format='%H %s'`, because a commit
  cannot embed its own hash in its own contents without changing that hash.

No main merge and no remote push took place. Full local logs:

`git log --oneline 6058704b4046eec77a5092871991ff9e146dddb4..m1/plant-2d-baseline`

Only code, dependency locks and derived statistical reports are committed.
Raw RGB/Mask, pretrained source, weights, features, checkpoints, prediction CSVs
and generated PDF/SVG/PNG figures remain on this machine under ignored paths.
The artifacts' hashes are recorded in `reproducibility.json`.
