# Data/source provenance and use restrictions

## Official references read before processing

- Dataset: https://huggingface.co/datasets/rick77a/PhenoLeaf_TS
- Code: https://github.com/pisyntor/PhenoLeaf-TS
- Card: https://github.com/pisyntor/PhenoLeaf-TS/blob/0d9a24ff146d12cc8fca6bdfa2538d196754f456/docs/DATASET.md
- Preparation: https://github.com/pisyntor/PhenoLeaf-TS/blob/0d9a24ff146d12cc8fca6bdfa2538d196754f456/phenoleaf_ts/data/prepare.py
- Terms: https://github.com/pisyntor/PhenoLeaf-TS/blob/0d9a24ff146d12cc8fca6bdfa2538d196754f456/DATASET_LICENSE.md

The toolkit is Apache-2.0. The separate La Trobe University dataset terms
restrict use to noncommercial academic research/teaching with citation.
No redistribution/public hosting without written permission. Raw data, masks
and panels containing raw pixels stayed local, never sent to a remote vision API.

## Resolving acquisition blocking

1. The official download function was tried. Direct Hugging Face access failed
   with network/hostname-resolution errors. Git transport initially timed out.
   No successful dataset count was inferred from these failures.
2. At the user's request to resolve blocking with equivalent alternatives, the
   actual dataset archive was obtained through a Hugging Face mirror; it was
   not replaced with synthetic data or an unrelated dataset.
3. Code was initially retrieved from official GitHub source archives/raw URLs.
   A later depth-1 Git fetch succeeded; the genuine official checkout is now
   `0d9a24ff146d12cc8fca6bdfa2538d196754f456`, not a replacement commit.
4. Transfer: https://hf-mirror.net/datasets/rick77a/PhenoLeaf_TS/resolve/main/PhenoLeaf-TS_DS.zip
5. Frozen data commit: `ee4c8dac44593a69c8a72e1ca09d686133161d92`.
   Hugging Face API/LFS metadata for that commit was independently retrieved
   through https://hf-mirror.com/api/datasets/rick77a/PhenoLeaf_TS/tree/ee4c8dac44593a69c8a72e1ca09d686133161d92
   This proxy is not described as an official host.
6. Actual archive: **5,486,793,580 bytes**; full SHA-256 equals the frozen LFS oid:
   `608c7b07125281fb92517182c0873d1a9c87d696e2adedcb11c002173887de2f`.
   The ETag/Xet hash is different and was not used as SHA-256.
7. A second resolver redirected a range request to **us.aws.cdn.hf.co**, the
   official CDN. Its first 1,024 bytes (HTTP 206) exactly matched local bytes.
   This is supplementary evidence, not a second complete official-CDN download.
8. ZIP CRC passed. The real root is `PhenoLeaf-TS_DS`, not the name in the card.
   Every one of 34,166 member files was verified unchanged after extraction,
   including 17,082 RGB and 17,082 masks plus ancillary files.

Evidence: `download_report.json`, `official_cdn_probe.json`,
`full_dataset_audit.json`, `../../data/metadata/file_manifest.csv`.
All labels were independently recomputed over the entire real dataset.

## Tooling/documentation

Current PyTorch Dataset/DataLoader docs were fetched with Context7 from
`/pytorch/pytorch`. pytest/JUnit docs from `/pytest-dev/pytest` are retained in
`pytest_documentation.txt`. The configured Bocha search endpoint returned a
quota/403 error; official definitions/license were read from actual source
files/URLs instead of being inferred from summaries. The referenced local
`/home/test/.codex/RTK.md` was absent.

## Interpretation limits

Filename time is validated calendar/wall-clock encoding, not an externally
calibrated clock; timezone is unknown. Forecast horizons are observations,
not days. Area is image-space, not calibrated physical area or LAI. Illegal
colors remain unchanged and quarantined; leaf reappearance warnings do not
automatically imply annotation errors. Synthetic tests do not replace real QA.

The trailing replicate label in a filename is not always the curated directory
replicate: 1,632 frames across 65 plants have such aliases. Directory plant_uid
remains authoritative, as specified. Independent acquisition-prefix IDs are
one-to-one with all 318 directory plant_uids; no prefix crosses directories,
and no filename genotype disagrees with its directory. This alias is not
silently ignored, nor used to merge distinct plants or alter splits.
See `../../data/metadata/plant_identity_audit.csv` and the `identity` section of
`geometry_time_and_label_statistics.json`.
