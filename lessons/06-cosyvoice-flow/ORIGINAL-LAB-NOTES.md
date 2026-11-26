# 第六课：CosyVoice 的三段式 TTS 架构

## A：离线、CPU、无需预训练权重的 Flow Matching 教学实验

`toy_flow_matching.py` 实际训练一个小型条件速度网络。输入包括 Mel-like 频谱的中间状态 `[B,1,24,36]`、时间 `t`、3 类条件，输出相同形状的速度张量。

- `pip install torch numpy matplotlib`
- `python toy_flow_matching.py --steps 500`
- 看 `results/flow_steps.png`：同一个条件 Flow 网络从随机噪声生成三类不同的频谱形状。
- 看 `results/loss.png`：训练时网络如何学习预测速度。

这不是 CosyVoice，不是语音合成器，不能生成可播放人声。它只演示 Conditional Flow Matching 的核心训练/推理机制。默认结果来自已实际运行的 500 步训练。

## B：真正 CosyVoice2-0.5B 推理（需要联网下载权重/安装其依赖）

参考官方仓库 https://github.com/QwenAudio/CosyVoice 。请先按官方 README 完成 `git clone --recursive`、安装和权重下载。然后从仓库根目录执行：

```bash
python /path/to/cosyvoice_lesson6/cosyvoice2_inference_example.py \
  --model pretrained_models/CosyVoice2-0.5B \
  --prompt-wav asset/zero_shot_prompt.wav \
  --prompt-text '希望你以后能够做的比我还好呦。' \
  --text '你好，我正在学习语音合成模型。' \
  --stream
```

`--prompt-text` 必须与 `--prompt-wav` 真正录制的文字匹配。示例代码根据官方 `example.py` 的 `inference_zero_shot(...)` API 编写，此环境未运行真实预训练模型，不能保证本机依赖/设备完全兼容。

## 官方配置中已验证的典型参数（CosyVoice2-0.5B）

- 输入 Speech Tokens 25Hz，FSQ 词表 6561。
- Flow 输出 80 维 Mel，50Hz。
- 语音输出采样率 24,000 Hz，Mel hop 480 个采样值。
- 核心模块为 `Qwen2LM`、`CausalMaskedDiffWithXvec`/`CausalConditionalCFM`、`HiFTGenerator`。
- Flow 训练使用速度场回归；实际实现还包括因果 Chunk Mask、说话人和参考语音条件。

源代码：
https://github.com/QwenAudio/CosyVoice
官方配置：
https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B/blob/main/cosyvoice2.yaml
论文：
https://arxiv.org/abs/2412.10117
