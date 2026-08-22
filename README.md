# ComfyUI Multimodal LLM

面向 ComfyUI 的统一视觉大模型插件。当前提供本地 Qwen3.8 GGUF 和
OpenAI-compatible API 两种后端，并通过同一个 `MLLM_BACKEND` 接口连接
`Multimodel Chat`。

插件只注册一套 `Multimodel` 节点，不创建 Python 虚拟环境，不升级 Torch、
CUDA 或显卡驱动。优先复用镜像中已有的 CUDA llama.cpp 和 API 依赖。

## 节点

| 节点 | 作用 |
| --- | --- |
| `Multimodel Local Qwen3.8 Loader` | 加载本地 Qwen3.8 GGUF 主模型和视觉 projector |
| `Multimodel API · Environment Variable` | 使用环境变量中的 API Key 调用兼容接口 |
| `Multimodel API · Direct Key` | 在节点中直接填写 API Key 调用兼容接口 |
| `Multimodel Chat` | 向本地模型或 API 后端发送文本、图片、视频帧 |
| `Multimodel Backend Unload` | 主动释放本地模型或关闭 API 后端 |

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

插件通过 ComfyUI 的 `folder_paths.models_dir` 定位上述目录，因此换镜像、换
显卡或换平台后仍使用当前 ComfyUI 的模型目录。确实需要外置目录时，才通过
`QWEN38_MODEL_DIR` 临时指定。模型下载完成后再启动或刷新 ComfyUI，加载器会
自动扫描目录中的 `.gguf` 文件；带有 `.aria2` 的未完成下载不会显示为可选模型。

推荐连接：

```text
Multimodel Local Qwen3.8 Loader ──► Multimodel Chat
```

5090/32GB 的 Q4_K_M 是当前兼容性基线。更高精度模型需要按照显存和上下文
长度重新评估；插件本身不会自动替换 Torch、CUDA、驱动或 llama.cpp wheel。

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

在 `Multimodel API · Direct Key` 的密码输入框中填写 Key。该节点不读取环境
变量，适合临时测试；不要把真实 Key 保存到公开工作流或提交到 Git。

### 参数行为

- `max_tokens=0`：不发送 `max_tokens`，使用服务商默认值；
- `temperature=0`：不发送 `temperature`，使用服务商默认值；
- API 节点输出 `response`、`usage`、`stats`，可直接连接其他字符串节点。

## 图片和视频

- 单张图片直接连接 `image`；
- 多张图片使用 ComfyUI 的 Image Batch 或其他批量图像节点，作为一个
  `IMAGE` 批次连接；
- `max_image_frames` 控制一次请求最多取多少张图片，默认 8 张；
- 视频可接 `video_frames`，插件会按 `max_video_frames` 均匀抽帧；
- 本地 llama.cpp 后端始终使用视频帧，API 后端根据 `video_transport` 选择
  兼容的视频数据或抽帧方式。

图片批次是在同一次请求中发送多张图片，不会自动拆成多个独立请求。

## 进度、日志和语言

- 模型加载会显示准备显存、加载 projector、加载权重和完成等阶段；
- 生成和 API 请求会显示 ComfyUI 进度，并把关键状态写入 `comfyUI.log`；
- 加载权重时 llama.cpp 没有字节级回调，日志会用心跳提示仍在加载，避免被
  误认为进程卡死；
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

然后重启 ComfyUI。插件只使用镜像现有依赖；只有在实际缺少 API 或本地后端
依赖时，才按镜像维护策略补装，不要为了本插件单独创建新环境。

## 问题排查

如果节点没有出现：

1. 确认插件目录位于 `ComfyUI/custom_nodes/ComfyUI-Multimodal-LLM`；
2. 查看 `comfyUI.log` 中是否有该目录的加载记录；
3. 确认浏览器已强制刷新；
4. 本地模型需确认两个 GGUF 文件存在且没有对应的 `.aria2` 文件；
5. API 需确认 `base_url`、模型 ID 和 Key 来源填写正确。
