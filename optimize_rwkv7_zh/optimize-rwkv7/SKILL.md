---
name: optimize-rwkv7
description: 为已有 RWKV-7 实现做等效提速，主要迁移 fused CUDA TimeMix kernel。用户要求加速 RWKV-7、迁移 RWKV-LM 更快 kernel、评测 RWKV-7 kernel，或要求在保持模型/损失等效的前提下提高吞吐时使用。
---

# 优化 RWKV-7

## 目标

使用这个 skill 时，目标是让 RWKV-7 更快，但不改变架构、参数形状、数据顺序、loss 语义、optimizer 语义或训练目标，除非用户明确要求改训练 recipe。

内置 `references/` 是一套已经在项目中跑通过的 RWKV-7 TimeMix fused CUDA 迁移参考。它们应该作为代码模板使用，而不是无脑覆盖：迁移时需要适配目标仓库的 import 路径、配置命名和模型封装 API。

## 第一轮检查

动手改代码前，先检查目标 RWKV-7 实现：

- 找到 TimeMix、ChannelMix/MLP、模型 block 封装，以及所有自定义初始化逻辑。
- 记录当前参数量和关键权重 shape。
- 确认 `n_embd`、`n_layer`、RWKV head size、head 数量、dtype、序列长度约束和训练精度。
- 确认当前 loss 是普通 cross entropy、fused cross entropy，还是自定义 L2Wrap 版本。
- 如果目标项目已经有正在跑的实验，除非用户要求重启或改配置，否则保持 run 命名和数据顺序不变。

## 等效 Kernel 迁移

安全的 kernel-only 迁移范围是 TimeMix 路径：

1. 将 `references/cuda/*` 复制到目标项目的 kernel 源码目录。
2. 根据 `references/fused_time_mix_kernels.py` 添加 fused kernel 加载和 autograd 包装器。
3. 根据 `references/rwkv7_fast_time_mix.py` 替换旧 TimeMix 模块。
4. 根据 `references/rwkv7_adapter.py` 增加或适配面向模型层的轻量 adapter。
5. 确保 fused kernels 在第一次 forward 之前只加载一次。
6. 保留 RWKV-7 初始化逻辑。如果外围模型有通用 reinit，要给 RWKV-7 linear 层打 skip 标记，或用其他方式确保显式初始化不会被覆盖。

参考 fused TimeMix 的操作映射：

- time shift 加六路输入 mix -> `tmix_mix6_bf16_v5`
- 原始 decay 构造 -> 保持 `w = w0 + tanh(xw @ w1) @ w2`；不要提前做 softplus
- value residual gate -> `tmix_vres_gate_bf16_v1`
- `a` gate -> `tmix_a_gate_bf16`
- key normalization 和 key adjustment -> `tmix_kk_pre_bf16_v5`
- recurrent WKV recurrence -> `rwkv7_clampw_cuda`
- GroupNorm 加 `r_k` residual 加 `g` gate -> `tmix_lnx_rkvres_xg_bf16_v1`
- 最后输出投影 -> 保留原有 `output` linear

内置 kernel 的重要约束：

- 这些是 bf16 kernel。
- 当前 wrapper 要求 `head_size=64`。
- `rwkv7_clampw_cuda` 内部会把序列长度 pad 到 16 的倍数，但正式测速仍应使用真实训练序列长度。
- H100 上建议设置 `TORCH_CUDA_ARCH_LIST=9.0`；远程机器上建议设置稳定的 `TORCH_EXTENSIONS_DIR` 作为编译缓存目录。

## 不要静默修改

不是所有 RWKV-LM 的优化都保持等效。除非用户明确要改训练 recipe，否则不要迁移这些内容：

- `rwkv7_cmix_bf16_v5`：这会把 FFN/MLP 换成 RWKV ChannelMix。如果目标模型原本不是这个 MLP，就不是等效修改。
- RWKV L2Wrap cross entropy：它会额外加入 max-logit 梯度项，有些版本还硬编码 vocab shape，会改变优化语义。
- RWKV optimizer 参数分组、weight decay 变化、`att.w0` 学习率倍增、layer-wise LR：这些会改变训练 recipe。
- 替换 RMSNorm/LayerNorm/GroupNorm，除非数值行为和归一化维度完全一致。
- 修改 head 数、hidden size、intermediate size、tokenizer、batch schedule 或样本顺序。
- 把 gradient checkpointing 当作提速手段启用；它通常是省显存但降速。

可单独考虑的等效提速：

- 小模型如果 FSDP 开销很重，可以在相同 global batch 下评测 DDP 或 no-shard。
- 只有当 fused cross entropy 和当前 loss 数学等价，并支持目标 vocab shape 时才使用。
- bf16 matmul-heavy 训练中，`torch.set_float32_matmul_precision("high")` 通常安全，但仍要验证 loss 和吞吐。

## 验证

迁移后，先验证再启动长实验：

- 确认参数量没有变化。
- 跑 TimeMix CUDA forward/backward smoke test；本项目结构可直接用 `scripts/smoke_test_fast_timemix.py`。
- 用真实 dtype 和序列长度跑一次完整模型 forward/backward。
- 检查 gradient 全部 finite。
- 如果开发期间还保留旧路径，用固定 seed 对比一批 forward/backward，误差在合理范围内再删除 fallback。
- 先启动短 probe run，检查 tokens/s/GPU、GPU util、显存和第一次 validation。

测速时：

- 吞吐估算不要包含第一步编译时间。
- 报告精确 GPU id、GPU 数量、global batch、sequence length、dtype 和 commit/hash。
- W&B/run logging 模式保持和用户当前实验流程一致。

## 参考文件

- `references/fused_time_mix_kernels.py`：Python loader、autograd 注册、padding wrapper 和面向 TimeMix 的 fused op。
- `references/rwkv7_fast_time_mix.py`：保留 RWKV-7 初始化的 fused TimeMix 模块。
- `references/rwkv7_adapter.py`：暴露 `RWKV7TimeMix` 的项目 adapter，并按 head size 去重加载 kernel。
- `references/cuda/*.cpp` 和 `references/cuda/*.cu`：这次快速 RWKV-7 迁移使用的 CUDA/C++ 源码。
- `scripts/smoke_test_fast_timemix.py`：迁移后用于本项目结构的 smoke test 模板。
