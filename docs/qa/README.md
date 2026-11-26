# 问答索引与学习轨迹

> 从十节课程期间的提问、纠错与延伸讨论中整理，不是聊天原文逐字转储。内容以教学上的“疑问 → 核心解答 → 验证方法”为轴。

## 按专题阅读

- [PCM / 滤波 / 频谱 / 分类](./01-audio-foundations.md)：12 个问答
- [Transformer / QKV / Whisper / 词表](./02-transformer-whisper.md)：13 个问答
- [Codec / VQ-VAE / TTS / Vocoder / Flow](./03-codec-tts-flow.md)：17 个问答
- [VAD / Moshi / OpenAI / KV / Dense-MoE / 工程](./04-full-duplex-serving.md)：13 个问答

## 四个重要的纠错案例

- [ESC-50 狗叫的黄屏不是叫声谐波主导，而是每秒独立峰值归一化的伪特征](./05-esc50-case-study.md)。
- [短暂停顿不是话轮结束，VAD 不等于 End-of-Turn](./04-full-duplex-serving.md)。
- [Codec 不是 TTS，且不是所有 TTS 必须用离散 VQ](./03-codec-tts-flow.md)。
- [静音仍可能占上下文；KV 环形缓存控制上限但不等于长期记忆](./04-full-duplex-serving.md)。

## 过程性学习导航

1. 先问“声音如何数字化、为什么是频率”，在第一课亲手生成 WAV 和 STFT。
2. 再问“为什么 CNN 能识别高低音顺序”，用 ESC-50 的反例发现数据预处理问题。
3. 追问“Whisper 是不是翻译模型，词表负责什么”，引出 Encoder/Decoder、Cross Attention。
4. 追问“Codebook/Token ID 从哪里来，为什么重建”，引出 VQ-VAE/RVQ 与 TTS 分工。
5. 追问“相位不一样声音有何区别，Flow 又是什么”，用频谱逆变换与条件速度场实验验证。
6. 继续比较 CosyVoice、Qwen3-TTS 与 Moshi，进一步问“为什么端到端实时模型能持续推理”。
7. 最后讨论话轮/打断、静音上下文、稠密/稀疏模型，以及 WebRTC 的实际部署。

## 说明

课程期间的部分问答源于图像截图、训练结果和口头讨论，无法在当前导出的源数据中恢复全部 UI 交互与逐字原文。如需完整逐字聊天，请另外通过 ChatGPT 的数据导出功能获得，并按隐私需求选择是否入库。
