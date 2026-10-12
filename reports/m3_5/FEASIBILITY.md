# M3.5 feasibility report — before future model training

Real Sugar4D V4 audit: 48 independent plants, 768 valid complete actual scans,
6,778 observed leaf records, 675 unique leaf identities, and 10,800 full-grid
cells. All 768 raw PLYs were decoded, leaf identity/centroid recalculated and
joined one-to-one with the original CSV. Zero duplicate or unmatched identity
triples. Current point geometry uses XYZ only.

| Split | Plants | h1 eligible / observed target | h2 eligible / observed target | h1 new-event + / − | h2 new-event + / − |
|---|---:|---:|---:|---:|---:|
| Train | 24 | 2498 / 2431 | 2253 / 2127 | 151 / 161 | 218 / 70 |
| Validation | 12 | 1172 / 1144 | 1055 / 1002 | 67 / 89 | 98 / 46 |
| Test, descriptive only | 12 | 1268 / 1236 | 1140 / 1076 | 81 / 75 | 109 / 35 |

Three consecutive observed history scans are required, and future targets are
the next one/two actual scans. These are not compressed leaf-only observation
indices. Future not_observed targets remain eligible for visibility prediction,
but their true morphology is NaN and has no morphology loss. No future identity
is input. New events count first observations after current time through target,
not net visible-count growth or botanical emergence.

All prespecified morphology/visibility/new-event train-validation coverage
gates pass in official and four unchanged M3 folds; see `feasibility.json` for
class-specific independent plant counts, `task_coverage.csv` for split counts,
`trajectories_per_plant.csv` for each plant's available tracks and sufficiently
long histories, and `DAS_task_distribution.csv` for phase imbalance. Test scores
or test class counts did not determine feasibility.

127 leaves cease to be observable before the last scan; zero disappear and
reappear. 4,022 not_observed grid cells include before-first-observed and
after-last-observed periods, not zero leaf area. There is no whole-scan missingness
and no PLY/CSV mismatch. Leaf disappearance still cannot be identified with
confirmed biological death.

**Decision: numerically feasible for published automatic-label prediction.**
Biological interpretation is limited: 450 final-DAS destructive references give
automatic/manual total-length MAE 1.918cm and width MAE 1.086cm. These are
cross-method discrepancies, not repeated longitudinal measurement noise.
Leaf area has no manual validation and is secondary only; area improvements
alone cannot choose ORGAN-TEMPORAL-JEPA. No model claims a noise-free growth rate.

**Not feasible as independent confirmation of true biological organ growth:**
there is no longitudinal repeat-scan measurement-error model, length/width
cross-method discrepancies are substantial, and no area reference exists.
Therefore only the finite, prespecified automatic-label baseline experiments
are allowed; no biologically precise growth-rate claim or large JEPA/generative
training is licensed by these labels. A stable numeric gain must still be
reported against these measurement limitations rather than silently declared
biological information.

Protocol was hashed before training. Main outcome is h2 leaf total length,
judged against current-state MLP and equal-capacity repeated-latest GRU with
three seeds, paired plant intervals and four validation fold directions.
Public source retrieved: https://www.ipb.uni-bonn.de/data/sugar4d/index.html .
Bocha search HTTP403/quota is recorded; actual feasibility is derived from local
raw measurements and PLYs, not a training-data assertion.
