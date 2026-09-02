# ComfyUI 的 Qwen3.8 VL

[English](README.md) | 简体中文

这是一个 ComfyUI 自定义节点包，支持通过 GGUF/`llama.cpp` 在本地运行 Qwen3.8 VL，也支持简单的 OpenAI-compatible 多模态 API。两种后端共用一个对话节点，因此同一套文本、图片和视频工作流可以切换后端使用。

> 本项目是独立社区项目，不包含模型权重。

## 快速开始

在 ComfyUI 安装目录执行：

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/xzbdqian10nian/ComfyUI-Qwen3.8-VL.git
```

1. 将一套匹配的 GGUF 主模型和 `mmproj` projector 放入：

   ```text
   ComfyUI/models/LLM/Qwen3.8/
   ```

2. 重启 ComfyUI（或等待服务加载节点后刷新浏览器）。
3. 添加 `Qwen3.8 VL Local Loader`，选择主模型和 projector。
4. 将它的 `backend` 输出连接到 `Vision LLM Chat`。
5. 填写 prompt 并执行工作流。图片连接 `IMAGE`，视频可连接 `VIDEO` 或 `IMAGE` 帧批次。

加载器使用 ComfyUI 当前配置的模型目录，因此普通 ComfyUI、便携版和云端镜像都可以使用同一套节点。若确实需要单独目录，可用 `QWEN38_MODEL_DIR` 环境变量覆盖默认目录。

### 更新

```bash
git -C ComfyUI-Qwen3.8-VL pull --ff-only
```

更新后重启 ComfyUI。如果旧版本安装在其他本地目录名下，请在那个目录执行同样的 `git pull` 命令。

## 节点

| 节点 | 用途 |
| --- | --- |
| `Qwen3.8 VL Local Loader` | 加载本地 Qwen3.8 GGUF 主模型和匹配的视觉 projector。 |
| `OpenAI-Compatible API · Environment Variable` | 使用环境变量中的 API Key 调用兼容接口。 |
| `OpenAI-Compatible API · Direct Key` | 使用节点中填写的 Key 调用兼容接口。 |
| `Vision LLM Chat` | 向选定后端发送文本、图片、图片批次、视频帧或 ComfyUI `VIDEO`。 |
| `Backend Unload` | 主动释放本地模型或关闭 API 后端。 |

节点内部 type 保持稳定以兼容旧工作流；界面展示名称已精简并与具体 API 服务商无关。

## 本地 Qwen3.8 VL 模型

本地后端需要两个匹配的文件：

```text
ComfyUI/models/LLM/Qwen3.8/
├── <Qwen3.8 主模型>.gguf
└── <匹配的 mmproj>.gguf
```

主模型包含语言模型权重，`mmproj` 是图片和视频理解所需的视觉投影模型。请使用同一模型发布版本中的两个文件，不要混用不同发布者的 projector。

32GB 显存建议从 `Q4_K_M` 或同等级的 UD Q4 版本开始。Q6/Q8 版本需要更多显存。上下文长度、batch size、micro-batch size、GPU layers 和 ComfyUI 显存清理都在节点中配置，不依赖特定平台的启动脚本。

### 模型下载

以下是 Qwen3.8-27B 的社区 GGUF 发布版本，请按需下载。

| 来源 | 常用文件 | 区别 |
| --- | --- | --- |
| [Unsloth](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) | [UD Q4_K_M](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-UD-Q4_K_M.gguf?download=true) + [mmproj-BF16](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/mmproj-BF16.gguf?download=true) | 推荐作为起点；动态量化体积较小，保留原模型的对齐方式。 |
| [Huihui AI Abliterated](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF) | [Q4_K](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF/resolve/main/Huihui-Qwen3.8-27B-abliterated-Q4_K.gguf?download=true) 或 Q4_K_L + `mmproj-model-bf16.gguf` | 社区低拒答版本；较大的 Q4_K_L 会保留更多高精度张量。 |
| [Orcarouter Uncensored](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-GGUF) | [Q4_K_M](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-GGUF/resolve/main/Qwen3.8-27B-Uncensored-Q4_K_M.gguf?download=true) + [mmproj f16](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-GGUF/resolve/main/mmproj-Qwen3.8-27B-Uncensored-f16.gguf?download=true) | 另一种社区低拒答版本；Hugging Face 可能要求登录或同意访问条件。 |

社区变体在对齐、拒答行为、质量和许可证方面可能不同。公开部署或再分发前，请阅读对应模型卡片。

## API 后端

两个 API 节点都会发送简洁的 OpenAI Chat Completions-compatible 请求。配置：

- `base_url`：服务商 API 根地址，例如 `https://api.openai.com/v1`；
- `model`：服务商提供的模型 ID；
- `prompt`：发送给模型的指令；
- `image`：可选的单张图片或 `IMAGE` 批次；
- `video_frames` 或 `video`：可选的视频输入，取决于服务商支持情况。

API 节点中的 `video_frames` 是 `IMAGE` 批次，适合支持多个 `image_url` 内容块的服务商；
`video` 是 ComfyUI 原生 `VIDEO` 对象。通用兼容性优先选择
`video_transport = frames`，明确支持 OpenAI-compatible `video_url` 内容块的服务商可选择
`video_url`，`auto` 则会在视频可以编码时优先尝试原生视频，否则使用抽帧。最终仍取决于服务商
是否支持对应的视频格式。

### 环境变量 Key

启动 ComfyUI 前设置：

```bash
export OPENAI_API_KEY='your-api-key'
```

在 `OpenAI-Compatible API · Environment Variable` 中，`api_key_env` 填 `OPENAI_API_KEY`，不要填 Key 本身。

### 直接填写 Key

临时测试时可以使用 `OpenAI-Compatible API · Direct Key`。不要把真实 Key 保存到公开工作流或提交到 Git。

### 参数行为

- `max_tokens = 0`：不发送 `max_tokens`，使用服务商默认值；
- `temperature = 0`：不发送 `temperature`，使用服务商默认值；
- `seed`：服务商支持时发送随机种子；
- 服务商没有提供输出总量时，API 进度条会保守推进，并在正常响应结束后完成。

### 环境变量 Key 的地址安全限制

环境变量节点不会把服务器上的 Key 发送到工作流任意填写的地址。它只允许访问服务器端允许列表中的精确 HTTPS 主机，默认允许 `api.openai.com`。如果要允许其他服务商，请在启动 ComfyUI 前设置：

```bash
export COMFYUI_API_ALLOWED_HOSTS='api.openai.com,api.example.com'
```

列表只接受精确的 `host` 或 `host:port`，不接受通配符。直接 Key 节点与环境变量节点分开；它可以使用其他地址，因为它只发送工作流用户在节点中明确填写的 Key，不会读取服务器环境变量。公网接口请使用 HTTPS。

## 图片和视频

- 单图直接连接到 `image`。
- 将 `IMAGE` 批次连接到 `image`，一次请求发送多张图片。
- 插件不会静默缩放图片或截断 `IMAGE` 批次；需要缩放或限制帧数时，在上游添加对应节点。
- 本地推理或支持图像帧的服务商，可将解码后的视频帧连接到 `video_frames`。
- 服务商支持视频传输时，可将 ComfyUI `VIDEO` 连接到 `video`；插件会按照 `max_video_frames` 抽帧。

## 示例工作流

[`example_workflows/`](example_workflows/) 提供五个本地后端示例：

| 文件 | 演示内容 |
| --- | --- |
| [`01_text_chat.json`](example_workflows/01_text_chat.json) | 纯文本对话 |
| [`02_single_image.json`](example_workflows/02_single_image.json) | 单张图片 |
| [`03_multiple_images.json`](example_workflows/03_multiple_images.json) | 多张图片作为一个 `IMAGE` 批次 |
| [`04_video_frames.json`](example_workflows/04_video_frames.json) | `VIDEO` 解码为图像帧 |
| [`05_comfyui_video.json`](example_workflows/05_comfyui_video.json) | ComfyUI 原生 `VIDEO` 输入 |

每个示例都包含模型/设置 Markdown 说明，并将 Chat 的回答连接到 ComfyUI 的 `Preview as Text` 展示节点。拖入后替换素材，并在加载器中选择模型目录里实际存在的一对文件。

## 兼容性和依赖

- Python 3.10+。
- 本地 GGUF 推理需要支持当前 CUDA/runtime 的 `llama-cpp-python`；请使用与现有 ComfyUI 镜像匹配的 wheel。
- API 模式优先使用现有 `openai` SDK；插件包含标准库 HTTP 回退。
- 插件不会替换 PyTorch、CUDA、NVIDIA 驱动或现有 Python 环境。
- `requirements.txt` 只保留插件所需的少量依赖。

## ComfyUI Manager 和 Registry

仓库已在 `pyproject.toml` 中加入 Comfy Registry 元数据，包括符合规范的包名、语义化版本、仓库地址、Publisher ID 和展示名称。发布到 [Comfy Registry](https://registry.comfy.org) 后，就可以在新版 ComfyUI Manager 中搜索和安装。在正式发布前，也可以直接执行 Git 安装：

```bash
git clone https://github.com/xzbdqian10nian/ComfyUI-Qwen3.8-VL.git
```

## 问题排查

**节点没有出现**

1. 确认仓库位于 `ComfyUI/custom_nodes/` 下；
2. 查看 ComfyUI 控制台中的插件导入记录；
3. 强制刷新浏览器；
4. 确认没有在第二个目录启用同一个插件的副本。

**加载器没有模型选项**

确认两个 `.gguf` 文件都位于 `ComfyUI/models/LLM/Qwen3.8/`，下载已完成，并且 projector 文件名包含 `mmproj`。

**本地模型加载失败**

确认主模型和 projector 匹配，并检查当前机器上的 `llama-cpp-python` wheel 是否支持现有运行时。显存不足时降低上下文长度、batch size 或 GPU layers。

**API 请求失败**

检查 endpoint 是否需要 `/v1`、模型 ID、Key 来源、服务商是否支持视觉输入，以及服务商是否接受当前图片/视频格式。

## 鸣谢

感谢 [Qwen Team](https://huggingface.co/Qwen) 开源 [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)，以及 [Unsloth](https://huggingface.co/unsloth)、[huihui-ai](https://huggingface.co/huihui-ai) 和 [orcarouter](https://huggingface.co/orcarouter) 提供社区 GGUF 发布版本。

本项目独立维护，与 Qwen Team 及上述社区发布者不存在隶属或官方背书关系。
