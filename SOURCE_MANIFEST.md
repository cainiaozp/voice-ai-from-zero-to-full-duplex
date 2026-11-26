# 源文件与新增内容清单

本仓库整合自本次课程中可访问的教学代码文件；没有从第三方仓库拷贝模型源码或预训练权重。

| 课程 | 已找回并保留的代码来源目录 |
|---|---|
| 01-pcm-stft-logmel | `audio_lesson1/` |
| 02-audio-cnn-esc50 | `audio_lesson2/` |
| 03-whisper-asr | `audio_lesson3/` |
| 04-neural-codec | `audio_lesson4/` |
| 05-vocoder-hifigan | `audio_lesson5.zip (for lesson5_vocoder_lab.py) + audio_lesson5/` |
| 06-cosyvoice-flow | `cosyvoice_lesson6/` |
| 07-qwen3-tts | `qwen3_tts_lesson7/` |
| 08-vad-turn-taking | `audio_lesson8/` |
| 09-moshi-full-duplex | `audio_lesson9/` |
| 10-realtime-system | `lesson10_full_duplex_lab/` |

## 此次新增

- 十课的详细中文 Markdown 讲义和合订本；55组整理问答、纠错列表及 ESC-50 故障复盘。
- `extras/phase/phase_lab.py`：相位试听生成。
- `extras/esc50-normalization/compare_normalization.py`：模拟片段归一化失真。
- `extras/dense-vs-moe/moe_toy.py`：Top-2 路由与梯度。
- `scripts/verify_repo.py`：静态结构、Python语法检查。
- 第04/05课原来演示使用的语音来源未纳入公开包，替换为当前新生成的数学合成音调。

## 不包含

- ChatGPT 平台的逐字聊天原文导出；这里只提供基于可见内容整理的教学问答。
- 未授权的私人语音、ESC-50 数据集原始 WAV、第三方预训练模型权重。
- 虚构的正式模型基准、真实服务性能测试结论。
