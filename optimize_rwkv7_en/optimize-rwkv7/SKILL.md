---
name: optimize-rwkv7
description: Optimize an existing RWKV-7 implementation with equivalent fused CUDA TimeMix kernels. Use when asked to speed up RWKV-7 training/inference, migrate faster RWKV-LM kernels, benchmark RWKV-7 kernels, or preserve model/loss equivalence while improving throughput.
---

# Optimize RWKV-7

## Goal

Use this skill to make RWKV-7 faster without changing the architecture, parameter shapes, data order, loss semantics, optimizer semantics, or training target unless the user explicitly asks for a recipe change.

The bundled references are a known-good project-level migration of RWKV-7 TimeMix to fused CUDA kernels. Treat them as code templates, not as blind copy-paste: adapt import paths, config names, and model wrapper APIs to the target repository.

## First Pass

Before editing, inspect the target RWKV-7 implementation:

- Locate TimeMix, ChannelMix/MLP, the model block wrapper, and any custom initialization.
- Record the current parameter count and key tensor shapes.
- Identify `n_embd`, `n_layer`, RWKV head size, number of heads, dtype, sequence length constraints, and train precision.
- Check whether the current loss is plain cross entropy, fused cross entropy, or a custom L2Wrap variant.
- If the target project has active experiments, preserve run names and data ordering unless the user asks to restart or change them.

## Equivalent Kernel Migration

The safe kernel-only migration is the TimeMix path:

1. Copy `references/cuda/*` into the target kernel source directory.
2. Add a wrapper based on `references/fused_time_mix_kernels.py`.
3. Replace the old TimeMix module with an implementation based on `references/rwkv7_fast_time_mix.py`.
4. Add or adapt a small model-facing adapter from `references/rwkv7_adapter.py`.
5. Ensure the fused kernels are loaded once before first forward.
6. Keep RWKV-7 initialization intact. If the surrounding model has generic reinitialization, mark RWKV-7 linear layers to skip it or otherwise preserve their explicit initialization.

The reference fused TimeMix maps these operations:

- time shift plus six input mixes -> `tmix_mix6_bf16_v5`
- raw decay construction -> keep `w = w0 + tanh(xw @ w1) @ w2`; do not pre-apply softplus
- value residual gate -> `tmix_vres_gate_bf16_v1`
- `a` gate -> `tmix_a_gate_bf16`
- key normalization and key adjustment -> `tmix_kk_pre_bf16_v5`
- recurrent WKV recurrence -> `rwkv7_clampw_cuda`
- GroupNorm plus `r_k` residual plus `g` gate -> `tmix_lnx_rkvres_xg_bf16_v1`
- final projection -> existing `output` linear

Important constraints from the bundled kernels:

- They are bf16 kernels.
- The bundled wrapper currently requires `head_size=64`.
- `rwkv7_clampw_cuda` pads sequence length to a multiple of 16 internally, but benchmark with realistic training sequence lengths.
- H100 should use `TORCH_CUDA_ARCH_LIST=9.0`; set `TORCH_EXTENSIONS_DIR` to a stable cache location on remote machines.

## Do Not Silently Change These

Not every RWKV-LM optimization is equivalence-preserving. Do not migrate these unless the user explicitly wants a new training recipe:

- `rwkv7_cmix_bf16_v5`: this replaces the FFN/MLP with RWKV ChannelMix. It is not equivalent if the target model currently uses another MLP.
- RWKV L2Wrap cross entropy: it adds a max-logit gradient term and some versions hardcode a vocabulary shape. It changes optimization semantics.
- RWKV optimizer parameter grouping, weight decay changes, `att.w0` learning-rate multipliers, or custom layer-wise LR: these alter the recipe.
- Swapping RMSNorm/LayerNorm/GroupNorm implementations unless numerics and axes match exactly.
- Changing head count, hidden size, intermediate size, tokenizer, batch schedule, or sample ordering.
- Enabling gradient checkpointing as a speed optimization; it usually saves memory while slowing training.

Equivalent optional speedups to consider separately:

- If a small model is using FSDP with heavy overhead, benchmark DDP or no-shard strategies with the same global batch.
- Use an existing fused cross entropy only when it is mathematically the same as the current loss and supports the target vocab shape.
- `torch.set_float32_matmul_precision("high")` is normally safe for matmul-heavy bf16 training, but still validate loss and throughput.

## Validation

After migration, run validation before starting a long experiment:

- Confirm parameter count is unchanged.
- Run a CUDA smoke test for TimeMix forward/backward; `scripts/smoke_test_fast_timemix.py` is a project-level helper for this repo shape.
- Run one full model forward/backward with realistic dtype and sequence length.
- Check gradients are finite.
- If the old path is still available during development, compare a fixed seed one-batch forward/backward within a reasonable tolerance before deleting fallback code.
- Start with a short probe run and inspect tokens/s/GPU, GPU utilization, memory, and first validation step.

When benchmarking:

- Exclude first-step compile time from throughput estimates.
- Report exact GPU ids, number of GPUs, global batch, sequence length, dtype, and commit/hash.
- Keep W&B/run logging mode consistent with the user's current experiment workflow.

## Reference Files

- `references/fused_time_mix_kernels.py`: Python loader, autograd registrations, padding wrapper, and user-facing fused TimeMix ops.
- `references/rwkv7_fast_time_mix.py`: fused RWKV-7 TimeMix module preserving RWKV-7 initialization.
- `references/rwkv7_adapter.py`: minimal project adapter that exposes `RWKV7TimeMix` and loads kernels once per head size.
- `references/cuda/*.cpp` and `references/cuda/*.cu`: CUDA/C++ sources copied from the fast RWKV-7 migration.
- `scripts/smoke_test_fast_timemix.py`: smoke test template for this repo after migration.
