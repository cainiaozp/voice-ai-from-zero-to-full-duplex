# 第十课：完整全双工调度教学实验

> 这个实验是**可运行的系统行为模拟器**，不是 Moshi、Sophon 或 GPT-Live 的复刻，也不是一套训练好的人声模型。它不接网络、不做真实语音识别，没有 GPU 依赖。所有延迟数字都是**虚拟时间线的脚本设置**，不是性能测试。

## 目标

1. 让麦克风输入和 AI 播放输出的时间线同时存在。
2. 用 20ms PCM 输入帧模拟前端音频采样，以 80ms 为单位合成输出音频块。
3. 展示“话还没说完的中途静音”与“真正结束”不一样。
4. 展示用户轻声附和不会立刻打断，明确打断时会取消当前回复及队列里的旧音频。
5. 用真实的 2 层多头因果注意力实现每层增量 K/V Cache，并与完整因果注意力前向结果核对。
6. 保存日志、可视化、可听的模拟音频，为之后接入 WebRTC 和模型打基础。

## 一键运行

```bash
python -m pip install numpy matplotlib
python lesson10_simulator.py
```

生成：

- `results/timeline.png`：用户语音与 AI 播放重叠、播放队列、KV 帧数的时间线。
- `results/simulated_duplex_stereo.wav`：左右声道分别是模拟用户与 AI 的合成音调（16kHz）。
- `results/frames.csv`：每20ms的输入活动、输出活动、KV长度、队列长度和事件。
- `results/events.json`：话轮提交、附和忽略、打断事件。
- `results/metrics.json`：仅对本次虚拟时间线计算的统计值。

## 关键事件

| 时间 | 事件 |
|---|---|
| 0.32～1.36s | 用户说第一半句话 |
| 1.36～1.62s | 停顿，但意图未完成，**不接话** |
| 1.62～2.20s | 用户说完第二半句话 |
| 2.40s | 模拟话轮模型提交第一轮 |
| 2.48s | 开始模拟播放 AI 第一次回答 |
| 2.88～3.04s | 用户发出附和，不停止播放 |
| 3.24s | 用户明确插话；清空排队的旧输出 |
| 3.56s | 用户结束插话 |
| 3.76s | 模拟话轮模型提交第二轮 |
| 3.84s | 开始播放 AI 新回答 |

## 真实系统需要替换哪些部分？

- **麦克风/音频网络**：目前使用虚拟时间轴 + NumPy 音调，生产环境用浏览器 WebRTC 和音频采集。
- **音频输入编码**：目前用 RMS 和过零率提取微型向量；应替换为 Mimi 或其它 Streaming Audio Encoder。
- **主模型**：当前的 2 层注意力仅计算状态，不参与话轮决策；替换为 Moshi / Sophon，或者 ASR + LLM。
- **话轮判断**：实验使用固定脚本的答案（oracle），未用真实分类模型；替换为 DuplexJev 或其他 EOT/打断模型。
- **声音合成**：当前为数学正弦波；替换为实际 Codec Decoder 或 Vocoder。
- **打断反馈**：生产环境除了清空服务端缓冲，还必须向浏览器发送 `response.cancel` / `barge_in` 等取消信号，并清除客户端已经排队但未播放的音频。需要通过 `generation_id` 等标识拒绝过期帧。
- **拥塞控制**：生产系统需要 bounded queue、丢弃过期结果、背压及GPU调度。

## 实验准确性说明

`CausalKV.one_frame()` 按帧计算每层 Q/K/V，缓存 K 和 V，`full_causal()` 全序列因果计算仅用于验证。这个有限玩具模型的数值等价性**不**说明已训练成可理解音频的模型。

## 真实项目与文档

- Moshi：https://github.com/kyutai-labs/moshi
- Sophon：https://github.com/adventists-ai/sophon-realtime
- Sophon E2E 部署：https://github.com/adventists-ai/sophon-realtime/blob/main/docs/deploy.md
- OpenAI GPT-Live 工程：https://openai.com/index/continuous-voice-interaction-with-gpt-live/
- LiveKit Agents：https://docs.livekit.io/agents/
- vLLM Metrics：https://docs.vllm.ai/en/latest/design/metrics/
