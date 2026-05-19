# ComfyUI-Easy-TRELLIS2

一个更适合手动维护 ComfyUI 环境的 TRELLIS.2 自定义节点分支。

本仓库基于 [PozzettiAndrea/ComfyUI-TRELLIS2](https://github.com/PozzettiAndrea/ComfyUI-TRELLIS2) 调整，核心目标是：不替你偷偷下载模型、不强行安装实验环境、不把可选加速库变成硬依赖，让 TRELLIS.2 更容易放进已有的 ComfyUI / Python / CUDA 环境里。

[TRELLIS.2](https://github.com/microsoft/TRELLIS.2) 是 Microsoft 的 image-to-3D 生成模型，可以从单张或多张图片生成 3D mesh，并支持 PBR 材质、网格处理、独立贴图和细化工作流。

## 这个分支做了什么

相对上游，本分支重点做了这些维护工作：

- 移除了 `comfy-env` / pixi 相关实验安装路径，避免节点启动时接管或重塑你的运行环境。
- 删除 `install.py` 和自动下载逻辑，模型文件必须由用户手动放到 ComfyUI 的 `models` 目录。
- 去掉 Hugging Face 下载相关直接依赖，缺文件时会明确提示需要放置哪些本地文件。
- 移除内置 RMBG / BiRefNet 节点和相关依赖，遮罩/抠图交给 ComfyUI 原生 SAM 或其他分割节点完成。
- 精简 `requirements.txt`，复用 ComfyUI 已经提供的基础依赖，避免重复安装和版本冲突。
- 将 `comfy_sparse_attn` vendored 到仓库内，降低额外 PyPI 安装压力。
- 去除 `flash-attn`、`sageattention`、`triton` 的硬依赖，默认可以回退到 ComfyUI attention 或 PyTorch SDPA。
- 去掉 `plyfile`、`zstandard` 等非直接使用依赖，以及上游示例素材/自动复制逻辑。
- 保留 TRELLIS.2 的核心生成、纹理、导出、mesh 处理和 Native sampling 节点。

换句话说，这个 fork 更偏向“干净、可控、少副作用”的 ComfyUI 节点版本。

## 安装

在 ComfyUI 的 Python 环境中安装：

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/eastmoe/ComfyUI-Easy-TRELLIS2.git
cd ComfyUI-Easy-TRELLIS2
pip install -r requirements.txt --upgrade
```

安装完成后重启 ComfyUI。

> 本分支不会运行 `python install.py`，也不会在节点加载或执行时自动从 Hugging Face 下载模型。

## 模型文件

TRELLIS.2 主模型需要手动放到：

```text
ComfyUI/models/trellis2/
```

至少需要包含：

```text
ComfyUI/models/trellis2/pipeline.json
ComfyUI/models/trellis2/ckpts/...
```

请从 [microsoft/TRELLIS.2-4B](https://huggingface.co/microsoft/TRELLIS.2-4B) 下载文件，并保持仓库内的相对路径。例如代码会按 `pipeline.json` 中声明的模型路径查找对应的：

```text
*.json
*.safetensors
```

如果文件缺失，节点会报错并列出缺少的相对路径。

### DINOv3

图像条件编码还需要 DINOv3 ViT-L 权重，放到：

```text
ComfyUI/models/dinov3/
```

支持以下文件名之一：

```text
dinov3-vitl-pretrain.safetensors
dinov3-vitl.safetensors
model.safetensors
```

可使用 [PIA-SPACE-LAB/dinov3-vitl-pretrain-lvd1689m](https://huggingface.co/PIA-SPACE-LAB/dinov3-vitl-pretrain-lvd1689m) 或你自己的等价 safetensors 文件。

## 依赖说明

`requirements.txt` 只保留本节点直接需要、且通常不是 ComfyUI 默认提供的依赖：

```text
flex_gemm_ap
cumesh_vb
o_voxel_vb_ap
drtk
trimesh
opencv-python-headless
```

以下依赖被刻意改为可选或移除：

- `flash-attn` / `sageattention`：可装可不装；未安装时使用 ComfyUI attention / PyTorch SDPA 回退。
- `triton` / `triton-windows`：不再是硬依赖。
- `comfy-sparse-attn`：已内置到 `./comfy_sparse_attn`。
- `timm` / `huggingface_hub` / `hf_transfer` / `hf_xet`：移除内置下载和 RMBG 后不再直接需要。
- `plyfile` / `zstandard`：不再作为默认依赖安装。
- `comfy-3d-viewers`：只在你使用外部 3D 预览节点或相关工作流时需要单独安装。

## 使用提示

1. 在 ComfyUI 中添加 `Load TRELLIS.2 Models` 节点。
2. 选择分辨率、精度和 attention backend。
3. 使用 `TRELLIS.2 Get Conditioning` 从图片和前景 mask 提取条件。
4. 使用 `TRELLIS.2 Image to Shape`、`TRELLIS.2 Shape to Textured Mesh`、导出或后处理节点完成 3D 生成。

TRELLIS.2 conditioning 需要前景 mask。本分支已移除内置 RMBG，请使用 ComfyUI 原生 SAM、Impact Pack、Inspire Pack 或其他你习惯的分割/遮罩节点。

可用工作流示例位于：

```text
workflows/
```

包括 geometry-only、geometry+texture、refinement、standalone texturing、mesh audit 等流程。

## 示例

![tpose](docs/tpose.png)

## 常见问题

### 为什么不自动下载模型？

这是本分支的主要改动之一。TRELLIS.2 模型体积较大，且很多用户的 ComfyUI 环境位于代理、离线、共享模型盘或自定义缓存结构中。手动放置模型能让路径、版本和磁盘占用都更可控。

### 没有 flash-attn / sageattention / triton 能运行吗？

可以。`attn_backend` 默认是 `auto`，会优先走 ComfyUI attention，并在需要时回退到 PyTorch SDPA。安装额外加速库可能提升速度，但不再是启动和基础运行的前提。

### 为什么移除了 RMBG？

RMBG/BiRefNet 带来额外模型和依赖，也会扩大安装面。本分支把抠图和分割交给更通用的 ComfyUI 节点生态，TRELLIS.2 节点专注 3D 生成本身。

## Credits

- [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) by Microsoft Research
- [PozzettiAndrea/ComfyUI-TRELLIS2](https://github.com/PozzettiAndrea/ComfyUI-TRELLIS2) 原始 ComfyUI 节点实现
- 本 fork 的维护改动见 [eastmoe/ComfyUI-Easy-TRELLIS2](https://github.com/eastmoe/ComfyUI-Easy-TRELLIS2) 提交记录

## License

请参考本仓库的 [LICENSE](LICENSE)，并同时遵守 TRELLIS.2 及相关模型权重的许可证和使用条款。
