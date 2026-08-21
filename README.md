# ComfyUI Multimodal LLM

统一的 ComfyUI 多模态 LLM 接口：本地 Qwen3.8 GGUF 和 OpenAI-compatible
API 可以接入同一个聊天节点。API 节点的请求格式参考
`ComfyUI_RH_LLM_API`，并增加了统一后端、批量图片、视频帧、工具调用和
输出面板结果。

## 当前基线

- 本地模型：`Qwen3.8-27B-UD-Q4_K_M.gguf`
- 视觉 projector：`mmproj-BF16.gguf`
- 模型目录：`/poddata/ComfyUI/models/LLM/Qwen3.8`
- 本地后端：CUDA llama.cpp（使用镜像现有依赖）
- API 后端：OpenAI Chat Completions compatible
- 输入：文本、IMAGE、IMAGE 批次视频帧、ComfyUI VIDEO
- 输出：response、reasoning、raw_response、stats

Q4_K_M 只是当前 5090 的兼容性基线，不代表 BF16/FP16/FP8。后续可以在
不改变工作流接口的情况下增加更高精度模型或其他本地后端。

## 推荐节点连接

### 本地 Qwen3.8

```text
Multimodal Local Qwen3.8 Loader
                ↓ backend
          Multimodal Chat
```

图片接 `image`。视频可以用 VHS/ComfyUI 视频节点输出 `IMAGE` 批次后接
`video_frames`；也可以把 `VIDEO` 对象直接接到 `video`，插件会为本地
llama.cpp 均匀抽取少量帧。

### API

```text
Multimodal API Backend (RH/OpenAI Compatible)
                ↓ backend
          Multimodal Chat
```

API Backend 参数：

- `base_url`：例如 `https://api.openai.com/v1` 或平台提供的兼容地址
- `model`：平台模型名
- `api_key`：可留空，优先读取 `api_key_env`
- `api_key_env`：默认 `OPENAI_API_KEY`
- `headers_json`：可选自定义请求头
- `extra_body_json`：可选平台扩展参数

建议使用环境变量保存 Key，不要直接写入工作流。API 的原生视频采用
RH 节点同样的 `video_url` data URL 格式；若服务商不支持，可将视频先
抽帧后接 `video_frames`，并把 `video_transport` 设为 `frames`。

## 新节点

- `Multimodal Local Qwen3.8 Loader`
- `Multimodal API Backend (RH/OpenAI Compatible)`
- `Multimodal Chat`
- `Multimodal Backend Unload`

## 旧工作流兼容

以下 class ID 保留不变：

- `Qwen38ModelLoader`
- `Qwen38VisionChat`
- `Qwen38Unload`

因此已有 Qwen3.8 工作流不需要迁移。新工作流建议使用通用节点，以便
以后在本地模型和 API 之间切换。

## 依赖策略

插件不会自动创建 Python 环境，也不会主动升级 Torch、CUDA 或驱动。
本地后端使用镜像中已有的视觉版 `llama-cpp-python`；API 后端优先使用
镜像已有的 `openai` SDK，若 SDK 不存在则使用 Python 标准库 HTTP 作为
轻量回退。

