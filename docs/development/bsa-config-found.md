# BSA configuration found in the published DiT config

**Discovered:** B1.3 (T2V pipeline scaffolding), while reading the published
`dit/config.json` from `meituan-longcat/LongCat-Video`.

The Plan v1 (B3.2 Open Question #1) flagged the BSA block shape as the
**#1 blocker for the Metal kernel** — kernel tile size, threadgroup memory
layout, and top-k value all depend on it. We assumed source-code spelunking
would be required to find these.

**It's right in the published config:**

```json
"bsa_params": {
    "sparsity": 0.9375,
    "chunk_3d_shape_q": [4, 4, 4],
    "chunk_3d_shape_k": [4, 4, 4]
},
"enable_bsa": false,
```

## Interpretation

- **Block shape: `[4, 4, 4]`** = latent (T_p, H_p, W_p) voxels — the same
  axis order as the DiT's `patch_size`. After patchify (1, 2, 2) on a
  latent (T_lat=N, H_lat=H/2, W_lat=W/2), the tokens are arranged as
  (T_lat, H_lat, W_lat) and BSA groups them into 4×4×4 = 64-token blocks.
- **Sparsity 0.9375** = keep `1 - 0.9375 = 0.0625 = 6.25%` of KV blocks
  per Q block. The tech report said "<10%" — config is tighter: ~6%.
- **`enable_bsa: false` in the published config** means BSA is OFF by
  default for the base DiT — it's enabled at inference time **only when
  the refinement_lora is applied** for the 720p second pass (where dense
  attention would be prohibitive).

## Sequence-length math

- Latent (T=24, H=60, W=104) — typical 480p 24-frame latent — gives
  `(T_lat * H_lat * W_lat) // 64 ≈ 60×104×24 // 64 = 2340 Q blocks`
  and the same KV blocks. Top-k (6.25%) = **146 KV blocks per Q block**
  attended.
- Same latent at 720p (T=48, H=90, W=160) after patchify (1,2,2) gives
  H_lat=45, W_lat=80 — `2160` Q-blocks (similar scale; the seq grows
  faster temporally than spatially since patch_size has `H_p=W_p=2` but
  `T_p=1`).

## Implications

- **B3.2 (Tier A pure-MLX):** block shape `[4, 4, 4]` is fixed and
  known. Top-k can be computed from sparsity * num_kv_blocks. Plan v1's
  Open Question #1 is **closed**.
- **B4.1 (Tier B Metal kernel):** kernel tile size = 64 tokens × head_dim
  = 64 × 128 = 8192 fp16 elements = 16 KB per Q block tile (fits Apple
  M-series threadgroup memory easily — 32 KB / threadgroup typical).
  Threadgroup math is unblocked.
- **B1.3 (T2V at 480p)** can ship **dense attention only** (BSA disabled).
  The coarse-pass model loads the published config with
  `enable_bsa: false` so no BSA code path is exercised. Only the
  refinement pass (B3.1) needs BSA.

## Action items

- Update Plan v1 changelog: BSA block shape resolved; remove Open
  Question #1.
- B3.2 implementation can proceed without source-spelunking.
- B1.3 (this PR) ships with BSA off — matches published default.
