# Protocol provenance correction

The original locked protocol SHA256 is
`8a138270bb64727ed57cfcef2a8f4e0a0685311c151a04dc65c68c94435521c0`.

During implementation a `notes` field containing two explanatory strings was
briefly added, then immediately removed. No model, seed, split, objective, epoch,
label budget, stage, selection rule or scientific threshold was changed. The
original protocol bytes and original lock were restored exactly. The transient
annotation-only file had SHA256
`bf79e05a5d2aaaac7f06319247f4e5f680416847f3565b19f418eb50c56f234d`.

Three started runs recorded that transient hash: `fold2/ssl_seed17`,
`official_count_P0_finetune_labels25_seed17`, and
`official_count_P0_finetune_labels50_seed17`. Their original checkpoint metadata
is retained, not rewritten to conceal provenance. An aggregation-only alias is
accepted after verifying the archived transient file differs exclusively in the
`notes` field. Runtime protocol verification always requires the original lock.
This was an engineering metadata-edit error, not a data-driven protocol amendment.
