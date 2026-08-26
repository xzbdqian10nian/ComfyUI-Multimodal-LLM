# ComfyUI Qwen3.8 VL

面向 ComfyUI 的统一视觉大模型插件。当前提供本地 Qwen3.8 GGUF 和
OpenAI-compatible API 两种后端，并通过同一个 `MLLM_BACKEND` 接口连接
`Vision LLM Chat`。

插件只注册一套 Qwen3.8 VL / Vision LLM 节点。模型路径通过 ComfyUI 官方的
`folder_paths.models_dir` 获取，不写死操作系统、云平台、用户目录、显卡型号或
存储挂载点；适用于标准 ComfyUI、整合包及云端部署。

## 节点

| 节点 | 作用 |
| --- | --- |
| `Qwen3.8 VL Local Loader` | 加载本地 Qwen3.8 GGUF 主模型和视觉 projector |
| `OpenAI-Compatible API · Environment Variable` | 使用环境变量中的 API Key 调用任意兼容接口 |
| `OpenAI-Compatible API · Direct Key` | 在节点中直接填写 API Key 调用任意兼容接口 |
| `Vision LLM Chat` | 向本地模型或 API 后端发送文本、图片、视频帧 |
| `Backend Unload` | 主动释放本地模型或关闭 API 后端 |

## 本地模型

默认模型目录使用 ComfyUI 自己的模型根目录，不写死平台的实际存储挂载路径：

```text
ComfyUI/models/LLM/Qwen3.8/
```

默认文件名：

```text
Qwen3.8-27B-UD-Q4_K_M.gguf
mmproj-BF16.gguf
```

### 模型下载来源与区别

下面三种都是基于 `Qwen/Qwen3.8-27B` 的社区 GGUF。首次使用建议选择
**Unsloth UD-Q4_K_M**；去拒答版本更适合研究和受控测试，不代表基础能力一定
更强，而且安全过滤明显更少。

| 来源 | 推荐主模型与视觉 projector | 主要区别 |
| --- | --- | --- |
| [Unsloth](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) | [Qwen3.8-27B-UD-Q4_K_M.gguf](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-UD-Q4_K_M.gguf?download=true)（约 15.3 GiB） + [mmproj-BF16.gguf](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/mmproj-BF16.gguf?download=true)（约 0.87 GiB） | 默认推荐。Dynamic 3.0 的 UD 量化，保留原版 Qwen 的对齐和拒答策略，通用能力、稳定性及复现最省心。 |
| [huihui-ai Abliterated](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF) | [Q4_K](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF/resolve/main/Huihui-Qwen3.8-27B-abliterated-Q4_K.gguf?download=true)（约 15.7 GiB）或 [Q4_K_L](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF/resolve/main/Huihui-Qwen3.8-27B-abliterated-Q4_K_L.gguf?download=true)（约 19.5 GiB） + [mmproj-model-bf16.gguf](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF/resolve/main/mmproj-model-bf16.gguf?download=true) | 通过 abliteration 降低拒答。作者说明前 15 层、MTP 和视觉部分未做消融；`Q4_K_L` 把部分关键张量保留为 Q8_0，体积和显存占用更高，但通常比普通 `Q4_K` 更保真。 |
| [orcarouter Uncensored](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-GGUF) | [Qwen3.8-27B-Uncensored-Q4_K_M.gguf](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-GGUF/resolve/main/Qwen3.8-27B-Uncensored-Q4_K_M.gguf?download=true)（约 15.7 GiB） + [mmproj-Qwen3.8-27B-Uncensored-f16.gguf](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-GGUF/resolve/main/mmproj-Qwen3.8-27B-Uncensored-f16.gguf?download=true)（约 0.87 GiB） | 另一种低拒答社区分支，提供标准 `Q4_K_M`、独立视觉 projector，并标注视觉、推理和 Function Calling 支持。仓库当前需要登录 Hugging Face 并同意访问条件后下载。 |

每次只需下载一套“主模型 + 同行对应的 projector”，全部放进：

```text
ComfyUI/models/LLM/Qwen3.8/
```

三个来源可以共存在该目录，加载器会分别列出所有 `.gguf` 文件；使用时请手动
选择同一来源的一对文件，不建议混用不同仓库的主模型和 projector。低拒答模型
可能生成敏感、不准确或不适合公开场景的内容，使用者需自行审核并遵守当地法律
及各模型仓库的许可证和使用条件。

插件通过 ComfyUI 的 `folder_paths.models_dir` 定位上述目录，因此换镜像、换
显卡或换平台后仍使用当前 ComfyUI 的模型目录。确实需要外置目录时，才通过
`QWEN38_MODEL_DIR` 临时指定。模型下载完成后再启动或刷新 ComfyUI，加载器会
自动扫描目录中的 `.gguf` 文件；带有 `.aria2` 的未完成下载不会显示为可选模型。

推荐连接：

```text
Qwen3.8 VL Local Loader ──► Vision LLM Chat
```

`Vision LLM Chat` 默认 `max_tokens=4096`，用于避免长结构化回答在 1024 tokens
附近被截断。它只是最大输出上限；简单问题在模型生成结束标记后仍会提前停止，
不会被强制扩写到 4096 tokens。

对 32GB 显存设备，Q4_K_M 是建议的起点。更高精度模型需要按显存和上下文
长度重新评估；不同平台请安装与自身 CUDA/CPU 匹配的
`llama-cpp-python`，插件不会替换 Torch、CUDA 或显卡驱动。

## API 后端

两个 API 节点都是简化的 OpenAI Chat Completions-compatible 请求，填写：

- `base_url`：服务商的 API 根地址，例如 `https://api.openai.com/v1`；
- `model`：服务商提供的模型 ID；
- `prompt`：发送给模型的指令；
- 图片可接一张 `IMAGE` 或一个 `IMAGE` 批次。

### 环境变量 Key

启动 ComfyUI 前设置：

```bash
export OPENAI_API_KEY='your-api-key'
```

节点中的 `api_key_env` 只填写变量名 `OPENAI_API_KEY`，不要填写 Key 本身。

### 直接填写 Key

在 `OpenAI-Compatible API · Direct Key` 的密码输入框中填写 Key。该节点不读取环境
变量，适合临时测试；不要把真实 Key 保存到公开工作流或提交到 Git。

### 参数行为

- `max_tokens=0`：不发送 `max_tokens`，使用服务商默认值；
- 此时 API 没有可用于精确计算的输出总量，进度条每累计约 2000 个流式
  token/chunk 前进 1%，最高停在 99%，服务商正常结束响应后再到 100%；
- 明确填写 `max_tokens` 时，进度条仍按该输出上限推进；
- `temperature=0`：不发送 `temperature`，使用服务商默认值；
- `seed`：发送随机种子；支持 ComfyUI 的生成后固定、递增、递减和随机选项，
  API 服务商支持 `seed` 时可用于尽量复现采样；
- API 节点输出 `response`、`usage`、`stats`，可直接连接其他字符串节点。

## 图片和视频

- 单张图片直接连接 `image`；
- 多张图片使用 ComfyUI 的 Image Batch 或其他批量图像节点，作为一个
  `IMAGE` 批次连接；
- 插件不缩放图片，也不截断 IMAGE 批次；需要缩放或限制数量时，在上游连接 ComfyUI 图像处理节点；
- 视频可接 `video_frames`，插件会按 `max_video_frames` 均匀抽帧；
- 本地 llama.cpp 后端始终使用视频帧，API 后端根据 `video_transport` 选择
  兼容的视频数据或抽帧方式。

图片批次是在同一次请求中发送多张图片，不会自动拆成多个独立请求。

## 示例工作流

`example_workflows/` 提供 5 个可直接拖入 ComfyUI 的 Qwen3.8 本地示例：

| 文件 | 用法 |
| --- | --- |
| `01_text_chat.json` | 普通纯文本对话 |
| `02_single_image.json` | 单张图片理解 |
| `03_multiple_images.json` | 两张或多张图片作为一个 IMAGE 批次输入 |
| `04_video_frames.json` | ComfyUI VIDEO 解码成 IMAGE 视频帧批次后输入 |
| `05_comfyui_video.json` | ComfyUI 原生 VIDEO 直接输入，由插件自动抽帧 |

拖入工作流后，在加载器中选择已经下载的“主模型 + 对应 projector”，再把示例
图片或视频替换为自己的素材即可。默认示例使用
`Qwen3.8-27B-UD-Q4_K_M.gguf + mmproj-BF16.gguf`；示例里的素材文件名只用于
展示连接方式，不随插件仓库分发。

本地 Qwen3.8 已实际验证上述 5 种输入均能完成推理。视频帧工作流适合上游已经
输出 IMAGE 批次的情况；原生 VIDEO 工作流则更简洁，并由插件按照
`max_video_frames` 均匀抽帧。

## 进度、日志和语言

- 模型加载会用日志进度条显示准备显存、projector、权重和完成等阶段；
- 本地生成和 API 请求同时显示 ComfyUI 原生进度与日志进度条，不再为每次
  流式更新新增一整行日志；
- 加载权重时 llama.cpp 没有字节级回调，日志会用心跳提示仍在加载，避免被
  误认为进程卡死；加载条是阶段进度，生成条按 token/chunk 上限推进；
- 节点参数带有鼠标悬停说明；
- 节点名称、参数名和说明随 ComfyUI 的 English / 简体中文设置切换；
- 翻译文件位于 `locales/en/nodeDefs.json` 和 `locales/zh/nodeDefs.json`。

## 安装和更新

在 ComfyUI 的 `custom_nodes` 目录执行：

```bash
git clone https://github.com/xzbdqian10nian/ComfyUI-Multimodal-LLM.git
```

更新现有插件：

```bash
git -C ComfyUI-Multimodal-LLM pull --ff-only
```

然后重启 ComfyUI。API 功能使用 `openai` SDK（缺少时有标准库回退）；本地
GGUF 推理需要与用户自身 CPU/CUDA 平台匹配的 `llama-cpp-python`。由于
CUDA wheel 与系统组合有关，本仓库不强行把用户环境替换为某个云平台的版本。

## 问题排查

如果节点没有出现：

1. 确认插件目录位于 `ComfyUI/custom_nodes/ComfyUI-Multimodal-LLM`；
2. 查看 `comfyUI.log` 中是否有该目录的加载记录；
3. 确认浏览器已强制刷新；
4. 本地模型需确认两个 GGUF 文件存在且没有对应的 `.aria2` 文件；
5. API 需确认 `base_url`、模型 ID 和 Key 来源填写正确。

## 鸣谢

感谢 [Qwen Team](https://huggingface.co/Qwen) 开源
[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)，为本插件提供基础模型与
原生多模态能力；感谢 [Unsloth](https://huggingface.co/unsloth)、
[huihui-ai](https://huggingface.co/huihui-ai) 和
[orcarouter](https://huggingface.co/orcarouter) 提供 GGUF 量化及社区变体。

本插件是独立社区项目，与上述团队不存在官方隶属或背书关系。
