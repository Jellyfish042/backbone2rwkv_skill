---
name: backbone-to-rwkv7
description: Replace a project's backbone network with the RWKV-7 architecture to achieve better results than architectures such as Mamba and Gated-DeltaNet.
---

Backbone to RWKV-7 is used to replace a project's backbone network with the RWKV-7 architecture.
Follow the steps below strictly in order. Do not explore the project structure before the first three steps are complete.
If any step fails, stop immediately and report the issue instead of continuing to later steps.

# 1. Inform the user of important notes

Output the following notes to the user without omitting anything:

- Warn the user that they must read these notes.
- Briefly describe the overall process, for example: "I will proceed through the following workflow..."
- Tell the user that they must manually configure an environment capable of running RWKV-7, which depends on PyTorch and CUDA, and that you will perform the necessary checks.

# 2. Ask for basic information

- Unless you already know them clearly, ask for the exact project runtime environment, such as a conda virtual environment named `example`, and the exact program entry point, such as `python train.py`. Do not present multiple-choice questions; directly ask the user to provide the information.

# 3. Verify the runtime environment

Runtime verification method: use the bundled `scripts/test_rwkv7.py` script in this skill. You must run `scripts/test_rwkv7.py` directly in the user's environment and inspect its output to verify that the RWKV-7 runtime environment is configured correctly. Do not only check the PyTorch and CUDA status.
If the environment is configured correctly, continue to the next step. Otherwise, tell the user to fix the environment configuration issue, then exit the workflow.

# 4. Identify the existing backbone network and plan the replacement strategy

## 4.1 Prerequisite knowledge: core RWKV-7 modules

```
Time-Mix (RWKV_Tmix_x070)
Purpose: A temporal mixing module that can replace Self-Attention in Transformers. It captures temporal dependencies between tokens.

Input: x [B, T, C], v_first [B, T, C]
Output: x [B, T, C], v_first [B, T, C]
B: batch size
T: sequence length
C: hidden dimension (n_embd)

Channel-Mix (RWKV_CMix_x070)
Purpose: A channel mixing module, similar to the FFN (Feed-Forward Network) in Transformers. It performs nonlinear transformations along the feature dimension to improve model expressiveness.

Input: x [B, T, C]
Output: x [B, T, C]
```

## 4.2 Analyze and plan

Follow these steps strictly:

- First, start from the project execution entry point provided by the user, trace the currently used backbone network modules step by step, and analyze how the existing backbone network works.
- Then read the RWKV-7 module implementations bundled with this skill in `references/rwkv7_channel_mix.py` and `references/rwkv7_time_mix.py`. Use that information to plan the strategy for replacing the backbone network with RWKV-7. When planning the replacement strategy, first identify which modules in the current backbone model temporal dependencies and which modules model relationships along the feature dimension. For example, in a Transformer architecture, the module that models temporal dependencies is Self-Attention, while the module that models feature-dimension relationships is usually the FFN. In a Mamba architecture, the Mamba module models both temporal dependencies and feature-dimension relationships. Unless the user explicitly specifies otherwise, the preferred default replacement strategy is to use only the RWKV-7 Time-Mix module to replace the current project's temporal-dependency modeling module, and not to replace the feature-dimension modeling module. For example, for a Transformer, the preferred replacement strategy should be to use only the RWKV-7 Time-Mix module to replace Self-Attention, while leaving the FFN unchanged. For a Mamba architecture, because the Mamba module models both temporal dependencies and feature-dimension relationships, it is not possible to replace only the temporal modeling module; therefore, recommend replacing the Mamba module with both RWKV-7 Time-Mix and Channel-Mix. Analyze other architectures case by case.

## 4.3 Output the replacement plan report

Enter Plan mode and output a report that includes the following:

- The type of data modeled by the current project's model, such as text, images, audio, bidirectional/unidirectional modeling, and so on.
- The backbone network architecture used in the current project, such as Transformer, LSTM, CNN, and so on.
- An analysis of whether the current project can be directly replaced with the RWKV-7 architecture, or what adjustments are required.
- If the current project's backbone can be replaced with the RWKV-7 architecture, recommend an RWKV-7 replacement strategy for the current project's task type, such as classification, generation, regression, and so on.
- Analyze a parameter-matching plan so that the replaced RWKV-7 model has a parameter count as close as possible to the current backbone network.
- Important notes: Other than the backbone network, preserve the original code logic as much as possible, such as scan order. The initialization scheme must follow the initialization scheme in the RWKV-7 module implementations bundled with this skill, and you must ensure that the weights are not overwritten by any other initialization logic.

After outputting the report, ask whether the user agrees with the replacement plan. If the user agrees, continue to the next step. Otherwise, discuss and revise the replacement plan with the user until agreement is reached.

# 5. Replace the backbone with the RWKV-7 architecture

- If the repository is a git repository, create a new branch, such as `convert-to-rwkv7`, and switch to that branch. Otherwise, modify the current working tree directly.
- Based on the replacement plan, generate code that replaces the current project's backbone network with the RWKV-7 architecture. During replacement, preserve the original code as much as possible and incrementally implement a branch that uses RWKV-7, for example by using a configuration file or command-line argument to control whether the original version or the RWKV-7 version is used. Directly refer to the implementations in `references/rwkv7_channel_mix.py` and `references/rwkv7_time_mix.py`, and reuse as much of that code as possible.
- Based on the project execution method provided by the user, provide the command or script for running the RWKV-7 version of the project after the replacement is complete.
