# 第03课｜Whisper：音频 Encoder、文字 Decoder、词表与交叉注意力

> **本课目标**：解释 Encoder–Decoder 与翻译模型的相似处、词表与权重矩阵的角色；运行真实 Whisper tiny 观察形状和注意力。

## 一眼看懂数据流

```text
16kHz PCM → Log-Mel `[B,80,3000]` → 2层 Conv1D + Transformer Encoder `[B,1500,384]` → Text Decoder (Self/Cross Attention) → 文字 Token IDs → Tokenizer decode
```

## 1. 为什么分类 CNN 不能直接 ASR

上一课输出三个/五十个类别分数，长度固定。而 ASR 要输出一句长度不固定的文字。Whisper 使用 **Audio Encoder + Text Decoder**，和传统 Seq2Seq 机器翻译非常相似：翻译的源输入是文字 Token，而 Whisper 的源输入是语音时序特征。

Whisper 的多语言 tiny 示例：16kHz、通常按30秒窗口处理，80维 Log-Mel，输入 `[B,80,3000]`；两层卷积，其中第二层按时间步长2，下采样后给 Transformer Encoder `[B,1500,384]`。这些音频位置是时间序列，不是先得到“你好”的文字。

## 2. 为什么有三种注意力

- **Encoder Self-Attention**：音频位置的 Q/K/V 都来自音频序列；经典 Whisper Encoder 能访问窗口内全部音频位置。
- **Decoder Masked Self-Attention**：文字 Q/K/V 都来自文字历史；不能读到未来文字。
- **Decoder Cross Attention**：文字 Decoder 的 Q 查询 Encoder 输出的 K/V；使当前文字预测读取已经编码的音频信息。

```text
Q_text:  [L, D]
K_audio: [T, D]
V_audio: [T, D]
Q @ K.T: [L, T]   → softmax → 加权读取音频 V
```

**交叉注意力不是“先识别每一帧是哪个汉字”**；其权重也不能直接当成强制音素对齐。

## 3. 文字词表与 Codec 码本有什么不同

Whisper 文字 Tokenizer 决定“文字/字节片段 ↔ Token ID”的映射。Decoder 输入 ID 经 Embedding 查表得到训练的向量，再通过分类式的输出投影预测下一 Token 的 ID。多语言 Whisper-tiny 词表规模约 51,865；特殊 Token 还控制语言、transcribe/translate、时间戳等任务。

**Codec 码本的 ID 指向的是声学特征向量，跟“你、好”没有一一对应关系。**

## 4. 训练与推理

有配对 `(音频,文字)`：训练时使用 Teacher Forcing，给 Decoder 真实的先前 Token，监督下一个 Token 的 Cross Entropy；推理时使用预测出的历史继续生成。经过 Encoder 的音频隐藏向量可在同一段解码中复用。经典 Whisper 并不是原生无限持续双工模型，虽然可以分块处理音频。

## 5. 本课运行

先运行无需下载模型的 `cross_attention_toy.py`，确认注意力权重矩阵每行求和为1。真实 Whisper 需要下载模型权重，并提供你自己有权使用的语音文件；仓库不附带私人语音素材。

**注意**：Whisper 文字 Token 不是一个汉字就必须一个 ID；输出是依照 tokenizer 的分词和控制 Token 规则进行的。

## 动手运行

从**仓库根目录**执行：

```bash
python lessons/03-whisper-asr/code/cross_attention_toy.py
```

```bash
python lessons/03-whisper-asr/code/lesson3_whisper.py --audio YOUR_SPEECH.wav --model openai/whisper-tiny --language zh --skip-attention
```

> 如本课含可下载权重的官方示例，先阅读 `../../docs/installation.md`，该步骤不计入无需网络的 CPU 实验。

## 课程附带素材 / 预期结果

- `transcript.txt`：真实模型的转录
- `logmel.png`：音频特征图
- `cross_attention.png`：需要去掉 `--skip-attention`

## 验收题

- [ ] 能说明 Audio Encoder 为什么保留时间维
- [ ] 解释文字 Q 和音频 K/V 的区别
- [ ] 观察 Whisper-tiny 的 `[1,1500,384]`
- [ ] 区分词表、Embedding、Output Projection

## 论文与官方项目

- https://github.com/openai/whisper
- https://arxiv.org/abs/2212.04356

**延伸问答**：参见 [问答索引](../../docs/qa/README.md)；实验性质说明见 [证据与限制](../../docs/evidence-and-limitations.md)。
