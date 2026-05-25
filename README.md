# Backbone to RWKV-7 Skills

Install the skill you need into the target project's `.codex/skills` directory.

## Available skills

| Skill | Language | Source path | Description |
| --- | --- | --- | --- |
| `backbone-to-rwkv7` | English | `backbone2rwkv_en/backbone2rwkv` | Convert a project's backbone network to RWKV-7. |
| `backbone-to-rwkv7` | Chinese | `backbone2rwkv_zh/backbone2rwkv` | Chinese version of the backbone-to-rwkv7 skill. |
| `optimize-rwkv7` | English | `optimize_rwkv7_en/optimize-rwkv7` | Optimize an existing RWKV-7 implementation with equivalent fused CUDA TimeMix kernels. |
| `optimize-rwkv7-zh` | Chinese | `optimize_rwkv7_zh/optimize-rwkv7` | Chinese version of the RWKV-7 optimization skill. |

## Install from this repository

Run from the target project root.

### macOS, Linux, or Git Bash

```sh
curl -fsSL https://raw.githubusercontent.com/Jellyfish042/backbone2rwkv_skill/main/install.sh | BACKBONE2RWKV_SKILL=backbone2rwkv BACKBONE2RWKV_LANG=en sh
```

### Windows PowerShell

```powershell
iex "& { $(irm https://raw.githubusercontent.com/Jellyfish042/backbone2rwkv_skill/main/install.ps1) } -Skill backbone2rwkv -Lang en"
```

Use `-Skill optimize-rwkv7` / `BACKBONE2RWKV_SKILL=optimize-rwkv7` to install the RWKV-7 optimization skill. Use `-Lang zh` / `BACKBONE2RWKV_LANG=zh` for Chinese.
