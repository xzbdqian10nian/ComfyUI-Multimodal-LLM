# ComfyUI Multimodal LLM

统一的 ComfyUI 多模态 LLM 接口：本地 Qwen3.8 GGUF 使用本地对话节点，
OpenAI-compatible API 使用独立的简化 API 节点。两条路径互不混用。

## 当前基线

- 本地模型：`Qwen3.8-27B-UD-Q4_K_M.gguf`
- 视觉 projector：`mmproj-BF16.gguf`
- 模型目录：`/poddata/ComfyUI/models/LLM/Qwen3.8`
- 本地后端：CUDA llama.cpp（使用镜像现有依赖）
- API：OpenAI Chat Completions compatible
- 输入：文本、IMAGE、IMAGE 批次
- 输出：response、usage、stats（API）；response、reasoning、raw_response、stats（本地）

Q4_K_M 只是当前 5090 的兼容性基线，不代表 BF16/FP16/FP8。后续可以在
不改变工作流接口的情况下增加更高精度模型或其他本地后端。

## 推荐节点连接

### 本地 Qwen3.8

```text
Multimodel Local Qwen3.8 Loader
                ↓ backend
          Multimodel Chat
```

图片接 `image`。视频可使用 VHS/ComfyUI 视频节点输出的 `IMAGE` 批次，
再接到 `video_frames`。

### API

```text
Multimodel API（环境变量） 或 Multimodel API（直接 Key）
     ↓ response / usage / stats
```

API 节点参数：

- `base_url`：例如 `https://api.openai.com/v1` 或平台提供的兼容地址
- `model`：平台模型名
- `Multimodel API（环境变量）`：只填写环境变量名，例如 `OPENAI_API_KEY`
- `Multimodel API（直接 Key）`：直接填写 API 密钥，不读取环境变量
- `prompt`、`max_tokens`、`temperature`：常用请求参数
- `image`：可选的单图或多图 IMAGE 批次

`max_tokens` 和 `temperature` 填 `0` 时不会把对应参数发给 API，使用服务商
自己的默认值；不是把参数设置成数值 0。

建议使用环境变量保存 Key，不要直接写入工作流。例如启动 ComfyUI 前执行：

```bash
export OPENAI_API_KEY='你的密钥'
```

节点中的环境变量字段填 `OPENAI_API_KEY` 即可。环境变量只填写“变量名”，
不是把密钥本身填进去。

多图操作：先用 ComfyUI 的 `Image Batch` 节点把多路 IMAGE 合并成一个 IMAGE
批次，再接到 API 节点的 `image`。批次中的每张图会放进同一次 API
请求中，而不是拆成多个请求。文件夹批量加载节点若输出 IMAGE 批次，也可以
直接连接。节点默认最多发送 8 张，可通过 `max_image_frames` 调整。

## 新节点

- `Multimodel Local Qwen3.8 Loader`
- `Multimodel API（环境变量）`
- `Multimodel API（直接 Key）`
- `Multimodel Chat`
- `Multimodel Backend Unload`

所有可见节点统一使用 `Multimodel` 前缀。旧版 Qwen3.8 节点实现文件暂时保留，
但不再注册到 ComfyUI，因此不会和统一节点重复显示。旧工作流中的旧节点需要
替换为对应的 `Multimodel` 节点。

## 交互反馈与语言

- 本地模型加载时显示原生 ComfyUI 进度条和阶段状态。
- 本地生成使用 `Multimodel Chat`，API 生成使用 `Multimodel API`；两者都会显示
  ComfyUI 进度条，进度状态也会同步写入 `comfyUI.log`。
- 每个输入参数和输出接口都提供鼠标悬停说明。
- 节点名称、参数名称和悬停说明会随 ComfyUI 的 English / 简体中文设置切换。
- 中文翻译位于 `locales/zh/nodeDefs.json`，英文位于
  `locales/en/nodeDefs.json`；切换语言后无需改工作流参数。

## 旧工作流迁移

为了避免旧节点和新节点同时出现在菜单中，旧 class ID 不再注册。旧工作流中
出现 `Qwen38ModelLoader`、`Qwen38VisionChat` 或 `Qwen38Unload` 时，请分别替换为：

- `Multimodel Local Qwen3.8 Loader`
- `Multimodel Chat`
- `Multimodel Backend Unload`

这是为了保证节点菜单只保留一套清晰的节点；旧工作流需要手动替换一次。

## 依赖策略

插件不会自动创建 Python 环境，也不会主动升级 Torch、CUDA 或驱动。
本地后端使用镜像中已有的视觉版 `llama-cpp-python`；API 节点优先使用
镜像已有的 `openai` SDK，若 SDK 不存在则使用 Python 标准库 HTTP 作为
轻量回退。
