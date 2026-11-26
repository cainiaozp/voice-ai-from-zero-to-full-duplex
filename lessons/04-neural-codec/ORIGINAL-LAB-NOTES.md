# 语音第四课：Neural Audio Codec

## 1. 从数字上理解 RVQ（不下载预训练权重）

安装依赖：`pip install numpy scipy soundfile matplotlib`

运行：`python lesson4_rvq_lab.py`

示例文件 `source.wav` 来自上一课的语音合成素材。

程序会学习4组有限码本，通过 RVQ 对 **原始 PCM 的 32 采样点块** 进行压缩，并输出：
- `results/after_1_codebooks.wav` 等音频
- `results/rvq_token_ids.npy`，形状 [4, T]
- `results/rvq_comparison.png`，波形、码本ID和SNR变化

**重要**：此程序是非神经网络的教学版 VQ/RVQ，它没有神经 Audio Encoder/Decoder，因此并非 EnCodec 或 Qwen3-TTS 的实现，且字典在同一条音频上拟合，没有独立验证泛化能力。它是为了直观理解「连续特征 → 多组码本 ID → 重建」而设计。

## 2. 真实的 EnCodec 预训练模型

安装：`pip install -U encodec torch torchaudio soundfile`（torch 与 torchaudio 版本请保持兼容）

运行：`python lesson4_encodec_real.py --input source.wav --bandwidth 6`

**首次运行需联网下载预训练模型**。此代码通过源码和语法检查，但在此环境未实际下载模型权重执行。

输出：`encodec_reconstructed.wav` 和 `encodec_reconstructed.npy`。

源码和权重：https://github.com/facebookresearch/encodec

另见 Qwen3-TTS Tokenizer: https://huggingface.co/Qwen/Qwen3-TTS-Tokenizer-12Hz
