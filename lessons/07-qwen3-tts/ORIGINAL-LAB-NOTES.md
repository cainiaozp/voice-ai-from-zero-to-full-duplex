# 第七课 · Qwen3-TTS：多码本 Speech Token 与低延迟流式解码

## A. CPU 上可复现的**教学模型**

```bash
pip install torch numpy soundfile matplotlib
python toy_multicodebook.py --steps 400
```

- 学习一个抽象 6-token 字母表和两个说话人条件。
- 模拟 Talker Backbone 预测第0组 Codebook；MTP 用前一组 Code ID + 深度位置 + 文本特征预测后15组。
- 本课程固定 16 层码本，但每组只有 32 个候选 ID（官方 Qwen3-TTS-12Hz 为 2048）。
- 输入 `[B,T]`，训练标签 `[B,T,16]`，预测结果 `[5,16]`。
- `results/training_and_codes.png`: 真实小网络训练Loss、准确率和生成码本ID热力图。
- `results/synthetic_01_codebooks.wav`、`04`、`08`、`16`: **教学专用的正弦波/谐波合成音频**。它们不是神经 Codec Decoder 的输出，不是中文语音；只用于理解16组码本逐渐添加细节。

> 训练数据是人为定义的映射规则，约40步便可达到100%准确率，不反映真实Qwen3-TTS的识别/合成能力。实际音频 Codec 是大量真实语音训练得到的。

## B. 使用官方预训练模型

环境：支持 CUDA 的 GPU、足够 RAM/VRAM、联网下载模型权重。

```bash
pip install -U qwen-tts soundfile
python official_qwen3_tts_inference.py
python official_tokenizer_encode_decode.py --audio your_audio.wav
```

以上官方 API 示例已参照 2026年11月的官方仓库与Tokenizer模型卡核对，但未在本机实际下载并运行预训练权重。

官方资料：
- https://github.com/QwenLM/Qwen3-TTS
- https://huggingface.co/Qwen/Qwen3-TTS-Tokenizer-12Hz
- https://arxiv.org/abs/2601.15621
