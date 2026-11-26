# 公开参考资料与版本来源

> 优先列原始论文、模型官方仓库和官方产品文档。这里是背景阅读，不代表本仓库代码等同于这些项目。链接整理时间：2026-11-25。

## 基础信号处理

- Librosa 文档：https://librosa.org/doc/latest/index.html
- ESC-50 官方：https://github.com/karolpiczak/ESC-50

## Transformer / ASR

- Attention Is All You Need：https://arxiv.org/abs/1706.03762
- Whisper 原始论文：https://arxiv.org/abs/2212.04356
- Whisper 代码：https://github.com/openai/whisper
- vLLM PagedAttention：https://arxiv.org/abs/2309.06180

## VQ / Neural Codec / Vocoder

- VQ-VAE：https://arxiv.org/abs/1711.00937
- VQGAN：https://arxiv.org/abs/2012.09841
- SoundStream：https://arxiv.org/abs/2107.03312
- EnCodec：https://github.com/facebookresearch/encodec
- HiFi-GAN：https://github.com/jik876/hifi-gan

## 生成语音模型

- CosyVoice2：https://arxiv.org/abs/2412.10117
- CosyVoice 官方：https://github.com/FunAudioLLM/CosyVoice
- Qwen3-TTS 官方：https://github.com/QwenLM/Qwen3-TTS
- Qwen3-TTS Tokenizer 模型：https://huggingface.co/Qwen/Qwen3-TTS-Tokenizer-12Hz
- Moshi 论文：https://arxiv.org/abs/2410.00037
- Moshi 官方：https://github.com/kyutai-labs/moshi

## 实时语音服务与话轮

- Sophon-Realtime：https://github.com/adventists-ai/sophon-realtime
- Silero VAD：https://github.com/snakers4/silero-vad
- Smart Turn：https://github.com/pipecat-ai/smart-turn
- LiveKit Agents：https://docs.livekit.io/agents/
- OpenAI 实时系统工程文章（2026-08-03）：https://openai.com/index/continuous-voice-interaction-with-gpt-live/
- OpenAI Voice Agents 官方指南：https://developers.openai.com/api/docs/guides/voice-agents

## Dense / MoE

- Switch Transformers：https://arxiv.org/abs/2101.03961
- Qwen3-TTS 与 Qwen3 的模型卡请前往上述官方仓库查最新配置；不同子版本参数差异较大，不以 README 中单一数字替代 config 检查。
