# 全部十课讲义（单文件阅读版）

> 此文件将每课完整讲义合并在一起，便于在 GitHub 中连续阅读或离线搜索。可运行代码、图表和验收任务位于各课程目录。原课程逐课文件是后续维护的唯一来源，更新课程后请重新生成合订本。

## 第01课｜从声音到数字：PCM、频率、FFT、STFT、Mel、MFCC

> **本课目标**：能解释声音为什么是气压随时间的变化；写出波形与频谱的形状；看懂低通与高通滤波；生成并试听自己的合成波形。

### 一眼看懂数据流

```text
空气中的气压变化 → 麦克风模拟电压 → ADC 离散采样 → PCM 波形 → STFT 复数频谱 → 功率谱 → Mel 滤波器组 → Log-Mel / MFCC
```

### 1. 输入到底是什么

人耳感受的是空气压力的振动，而不是“频率本身”。麦克风把压力变化变成电压；ADC 每秒读取若干次，得到 PCM 数列。**采样率 16 kHz** 表示每秒有 16,000 个采样时刻，而不是音量范围为 0～16,000。

```text
1 秒单声道 16kHz PCM: [16000]
Batch 化:                 [B, 16000]
16-bit PCM 文件:          通常 int16，范围 -32768..32767
神经模型输入:              常转换为浮点数，约 [-1, 1]
```

图像是 `H×W×3` 像素矩阵；音频首先是按时间排列的一维振幅序列。声音的“音高”主要与周期及基频有关；“响度”大体与振幅/感知频谱有关；“音色”依赖谐波分布、包络、瞬态与更多结构。

### 2. 为什么要傅里叶变换

原始波形可以呈现周期，但很难一眼看出多个频率同时存在。FFT 将一个时间窗中的振动表示为不同频率的复数系数：复数的模表示幅度，相角表示相位。

- **FFT**：分析所选整段窗口的频率成分；不保留多段事件各自在何时发生。
- **STFT**：不断截取短窗口做 FFT；把频率谱沿时间排成二维图。
- **功率谱**：常使用 `abs(STFT)**2`，丢掉原始相位。
- **Mel 滤波组**：把线性 Hz 频率分量汇总到接近听觉分辨方式的频带，不是 CNN 自己训练出来的滤波器。
- **Log-Mel**：压缩动态范围，适合用于声学模型；不能唯一恢复原始 PCM。
- **MFCC**：进一步用 DCT 表示 Mel 对数谱形态，经典语音特征之一。

以 `n_fft=400, hop=160, center=False, sr=16000` 为例，1 秒信号 `T=16000`，STFT 时间帧数为 `floor((16000-400)/160)+1=98`；线性频率维度是 `400/2+1=201`；80个 Mel 滤波器得到 `[80,98]`。

### 3. 滤波是怎么一回事

一个低通滤波器允许低频通过、减弱高频；高通恰好相反。第一课配套合成 `220Hz、440Hz、880Hz`，可以在频谱上逐条验证。滤波器并不是简单删掉音频时间轴上的某些点，而是改变各频率成分的幅度（实际还可能改变相位）。

### 4. 与后续模型的联系

`Log-Mel` 是 Whisper 等 ASR 的常见输入；而 Moshi/Mimi 的神经 Codec 可以直接从 PCM 中学习压缩表示。STFT 和 Mel 是数字信号处理方法，不以深度学习的出现为前提。

### 5. 已知陷阱

- Mel bin 40 **不是** 40Hz；纵轴是经过非线性分组的频率索引。
- 频谱颜色代表所选参考与颜色范围内的强弱；不同图片独立归一化时，亮度不可简单横向比较。
- `center=True/False` 会改变边界补齐与时间帧数，实验要标明。
- 振幅不能直接和物理声压/感知响度做绝对等号。

### 动手运行

从**仓库根目录**执行：

```bash
python lessons/01-pcm-stft-logmel/code/lesson1.py
```

> 如本课含可下载权重的官方示例，先阅读 `docs/installation.md`，该步骤不计入无需网络的 CPU 实验。

### 课程附带素材 / 预期结果

- `figures/fft.png`：纯音与混合音的 FFT
- `figures/sequence_analysis.png`：时域切换造成的时频变化
- `samples/*.wav`：数学合成的可公开音调

### 验收题

- [ ] 能说明 16kHz PCM 数组长度
- [ ] 能算出 `[201,98]` 与 `[80,98]`
- [ ] 能区分 FFT 与 STFT
- [ ] 能解释幅度、相位与 Mel 的区别

### 论文与官方项目

- https://librosa.org/doc/latest/index.html

**延伸问答**：参见 [问答索引](docs/qa/README.md)；实验性质说明见 [证据与限制](docs/evidence-and-limitations.md)。


---

## 第02课｜AudioCNN：从时频纹理到声音分类、ESC-50 与数据泄漏

> **本课目标**：训练三类高低音顺序识别器，观察 logits / embedding；理解真实 ESC-50 时频结构和错误归一化造成的伪特征。

### 一眼看懂数据流

```text
PCM → STFT → Log-Mel `[B,1,80,98]` → Conv2D → Pooling → Embedding `[B,512]` → Linear(512,3) → Softmax
```

### 1. 我们要分类什么

教学任务用合成的三种声音：`low_to_high`（先低后高）、`high_to_low`（先高后低）及 `simultaneous`（高低同时出现）。音频片段为1秒、16kHz。它们频率成分可以接近，但**时间排列不同**；因此需要二维时频纹理而非仅仅一个整段 FFT。

训练脚本生成不同频率、相位、音量、噪声和切换时刻，使模型学到比较通用的时频变化规律。网络约6千余个参数；最后 `Linear(512,3)` 固定输出三个 logits，经过 Softmax 得到类别分布。**单样本54.6%的最大 Softmax 概率，不等于整个模型54.6%的准确率。**

### 2. 真实声音的时频线索

ESC-50 包含50种环境声音。典型图形：雨声是持续宽频颗粒纹理，海浪有慢速能量起伏，引擎/吸尘器有相对稳定的窄频线，警笛有上升/下降的频率轨迹，鸟鸣/蟋蟀呈高频重复短线，婴儿哭/公鸡有谐波组合，狗叫/玻璃破碎有突发宽频事件，雷声低频能量更明显。

这些只是统计倾向，单个图案不保证唯一类别。每类只有40条数据，迁移到50类分类时需要分离好训练/验证/测试数据，宜使用原始的五折分组进行交叉验证。

### 3. 问答中发现的归一化陷阱（重点案例）

用户发现真实 ESC-50 的 dog 图有大片亮黄色，进一步用统计数据定位：`dog: ≥0.9 的像素占80.1%，p50≈1.00；rooster 40.6%，rain 1.1%，crickets 1.0%`。原因并不主要是谐波，而是**每个1秒片段独立按峰值归一化**，再拼成5秒。安静片段内“最大值”也来自背景噪声甚至数值下限；减去自身最大值后，背景会被抬到接近1。

教学脚本中的数学形式为：

```python
## db: [mel_bins, frames]
relative_db = db - db.amax()  # 如果每秒都重新计算，跨秒强度失真
normalized = (relative_db.clamp(-80, 0) + 80) / 80
```

这不是 Matplotlib 画错，也不是“亮黄代表真实声音很响”。对于稀疏事件，推荐做实验对比：**1秒独立归一化 / 5秒统一峰值 / 固定 dB 参考**。详细讨论见 [`docs/qa/05-esc50-case-study.md`](../../docs/qa/05-esc50-case-study.md)。

### 4. 标注粒度与弱监督

5秒录音被标成 dog，不表示随机截取的每一秒都有狗叫。把无事件片段强行标为 dog 会引入噪声标签、促使模型记住录音背景。对比5秒整段分类、事件感知裁剪，或 MIL（多个片段聚合成录音级分类）是合理的进阶实验。

### 5. 模型可解释性

`Embedding [B,512]` 是模型提取的全局特征，不是512个频率，也不是512个 Speech Token。经过全局池化它可能舍弃大量时间和音色细节，所以它不能直接替代 Whisper 的时序 Encoder 或可重建声音的 Codec Latent。

### 动手运行

从**仓库根目录**执行：

```bash
python lessons/02-audio-cnn-esc50/code/train_audio_cnn.py --epochs 6
```

> 如本课含可下载权重的官方示例，先阅读 `docs/installation.md`，该步骤不计入无需网络的 CPU 实验。

### 课程附带素材 / 预期结果

- `figures/samples.png`：三类声音的 Log-Mel
- `samples/*.wav`：合成分类样例
- 生成的 `results/training_curves.png`、`history.csv`、`audio_cnn.pt`（默认不会提交 Git）

### 验收题

- [ ] 三分类可训练并打印混淆矩阵
- [ ] 解释 logits 与 embedding 的区别
- [ ] 解释为什么 ESC-50 不能任意随机切分
- [ ] 能够复现按片段归一化造成的伪黄色

### 论文与官方项目

- https://github.com/karolpiczak/ESC-50

**延伸问答**：参见 [问答索引](docs/qa/README.md)；实验性质说明见 [证据与限制](docs/evidence-and-limitations.md)。


---

## 第03课｜Whisper：音频 Encoder、文字 Decoder、词表与交叉注意力

> **本课目标**：解释 Encoder–Decoder 与翻译模型的相似处、词表与权重矩阵的角色；运行真实 Whisper tiny 观察形状和注意力。

### 一眼看懂数据流

```text
16kHz PCM → Log-Mel `[B,80,3000]` → 2层 Conv1D + Transformer Encoder `[B,1500,384]` → Text Decoder (Self/Cross Attention) → 文字 Token IDs → Tokenizer decode
```

### 1. 为什么分类 CNN 不能直接 ASR

上一课输出三个/五十个类别分数，长度固定。而 ASR 要输出一句长度不固定的文字。Whisper 使用 **Audio Encoder + Text Decoder**，和传统 Seq2Seq 机器翻译非常相似：翻译的源输入是文字 Token，而 Whisper 的源输入是语音时序特征。

Whisper 的多语言 tiny 示例：16kHz、通常按30秒窗口处理，80维 Log-Mel，输入 `[B,80,3000]`；两层卷积，其中第二层按时间步长2，下采样后给 Transformer Encoder `[B,1500,384]`。这些音频位置是时间序列，不是先得到“你好”的文字。

### 2. 为什么有三种注意力

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

### 3. 文字词表与 Codec 码本有什么不同

Whisper 文字 Tokenizer 决定“文字/字节片段 ↔ Token ID”的映射。Decoder 输入 ID 经 Embedding 查表得到训练的向量，再通过分类式的输出投影预测下一 Token 的 ID。多语言 Whisper-tiny 词表规模约 51,865；特殊 Token 还控制语言、transcribe/translate、时间戳等任务。

**Codec 码本的 ID 指向的是声学特征向量，跟“你、好”没有一一对应关系。**

### 4. 训练与推理

有配对 `(音频,文字)`：训练时使用 Teacher Forcing，给 Decoder 真实的先前 Token，监督下一个 Token 的 Cross Entropy；推理时使用预测出的历史继续生成。经过 Encoder 的音频隐藏向量可在同一段解码中复用。经典 Whisper 并不是原生无限持续双工模型，虽然可以分块处理音频。

### 5. 本课运行

先运行无需下载模型的 `cross_attention_toy.py`，确认注意力权重矩阵每行求和为1。真实 Whisper 需要下载模型权重，并提供你自己有权使用的语音文件；仓库不附带私人语音素材。

**注意**：Whisper 文字 Token 不是一个汉字就必须一个 ID；输出是依照 tokenizer 的分词和控制 Token 规则进行的。

### 动手运行

从**仓库根目录**执行：

```bash
python lessons/03-whisper-asr/code/cross_attention_toy.py
```

```bash
python lessons/03-whisper-asr/code/lesson3_whisper.py --audio YOUR_SPEECH.wav --model openai/whisper-tiny --language zh --skip-attention
```

> 如本课含可下载权重的官方示例，先阅读 `docs/installation.md`，该步骤不计入无需网络的 CPU 实验。

### 课程附带素材 / 预期结果

- `transcript.txt`：真实模型的转录
- `logmel.png`：音频特征图
- `cross_attention.png`：需要去掉 `--skip-attention`

### 验收题

- [ ] 能说明 Audio Encoder 为什么保留时间维
- [ ] 解释文字 Q 和音频 K/V 的区别
- [ ] 观察 Whisper-tiny 的 `[1,1500,384]`
- [ ] 区分词表、Embedding、Output Projection

### 论文与官方项目

- https://github.com/openai/whisper
- https://arxiv.org/abs/2212.04356

**延伸问答**：参见 [问答索引](docs/qa/README.md)；实验性质说明见 [证据与限制](docs/evidence-and-limitations.md)。


---

## 第04课｜神经音频编解码、VQ-VAE、Codebook 与 Residual VQ

> **本课目标**：通过码本 ID 的最近邻查表、残差多级量化、波形重建，理解 Speech Token 为什么能降低生成序列长度。

### 一眼看懂数据流

```text
PCM `[B,1,N]` → Audio Encoder → Latent `[B,D,T]` → VQ/RVQ → Codes `[B,K,T]` → Codebook Lookup → Decoder → PCM
```

### 1. 为什么要重建，而不是分类

“可恢复表示”指 Encoder 在**Decoder 不能看到原始数据**的前提下，只提供一个压缩的中间表示；模型通过原始与重建信号的差异学习哪些信息必须保留。这不是为了生成另一张猫图或另一句话，而是为了获得以后可以复用的声学表示体系。

VQ-VAE 在图像中是 `RGB → Conv2D Encoder → [B,H',W',D] → Codebook ID → Decoder → 重建RGB`；音频中改成一维时间：`PCM → Conv1D Encoder → [B,D,T] → RVQ ID → 音频 Decoder`。

### 2. 为什么有 Codebook 与 ID

Codebook `[1024,256]` 表示1024个可以学习的256维向量。每个 Encoder 输出向量选择最近的一行，保存整数 ID。例如 ID=18 不代表“猫耳朵”或汉字“你”，而是**当前模型的第18个特征向量**。改变模型权重后相同 ID 可以代表完全不同的内容。

在理想编码条件下，1024种候选只需10bit标识选择；原来的256维FP16向量需要512字节。实际端到端压缩收益还要考虑码本/模型权重与编码格式，量化会带来信息损失。码本与 Encoder、Decoder 联合训练，可用 Codebook Loss/Commitment Loss、STE 或 EMA。

### 3. RVQ 是对残差做进一步量化

第一个码本选近似向量，第二个编码剩余误差，第三个继续修正。最终多组 Codebook 向量相加，重建一个 Latent 时间位置；**一个时间位置的多个 Token ID 并不是多个相继发音**。

### 4. Codec 为什么能让 TTS 更轻

例如24kHz 2秒 PCM 有48000个采样位置。以12.5Hz、8码本为例，2秒有25个 Codec 时间帧、合计200个 ID。它降低大模型必须逐采样点自回归的序列负担，但并不意味着整个系统的 FLOPs 自动减少240倍——Depth/多码本生成和 Decoder 本身仍有开销。

训练 Codec 可以只用未标注 WAV；之后训练 TTS Talker，则通常需要文字/录音配对，通过 Codec Encoder 为真实声音生成离散目标。

### 5. 两个实验，不要混淆

`lesson4_rvq_lab.py` 是**对 PCM 的短帧直接做 K-Means + RVQ**，用于听懂码本与残差，**没有神经 Encoder/Decoder，不是 EnCodec**。仓库自带 `source.wav` 为数学合成音调。

`lesson4_encodec_real.py` 则依赖官方 EnCodec 预训练权重；需要正确的 torch/torchaudio 环境与下载权限。运行后比较原始和重建 WAV，同时检查码本层数、帧率和 SNR。

### 动手运行

从**仓库根目录**执行：

```bash
python lessons/04-neural-codec/code/lesson4_rvq_lab.py
```

```bash
python lessons/04-neural-codec/code/lesson4_encodec_real.py --input lessons/04-neural-codec/code/source.wav --bandwidth 6
```

> 如本课含可下载权重的官方示例，先阅读 `docs/installation.md`，该步骤不计入无需网络的 CPU 实验。

### 课程附带素材 / 预期结果

- `results/rvq_token_ids.npy`：RVQ码本 ID `[K,T]`
- `results/rvq_comparison.png`：不同层数重建质量
- `results/after_*_codebooks.wav`：可试听的重建

### 验收题

- [ ] 能解释最近邻量化与残差
- [ ] 能区分每帧与每个PCM采样
- [ ] 能说明 Codebook 是训练参数而非人工标签
- [ ] 能区分 Codec 和 TTS

### 论文与官方项目

- https://arxiv.org/abs/1711.00937
- https://arxiv.org/abs/2107.03312
- https://github.com/facebookresearch/encodec

**延伸问答**：参见 [问答索引](docs/qa/README.md)；实验性质说明见 [证据与限制](docs/evidence-and-limitations.md)。


---

## 第05课｜Vocoder 与 HiFi-GAN：从频谱回到可播放的波形

> **本课目标**：亲耳比较完整 STFT、只保留幅度和 Mel 后的恢复；理解 Generator、MPD/MSD 与 Codec Decoder 的关系。

### 一眼看懂数据流

```text
TTS 声学模型 → Mel `[B,80,T]` → 神经 Vocoder（多级上采样）→ PCM `[B,1,N]`
```

### 1. 为什么有 Mel 还不能直接播放

PCM 是直接可播放的振幅序列；复数 STFT 保存幅度与相位，满足条件时经 iSTFT 能基本恢复波形；幅度谱丢失相位；Mel 进一步把频率分量聚合。由 Mel 不能唯一确定原始 PCM。

把正弦波统一平移180°，其单独播放的音高与稳态响度基本不变；但两条同频同幅正弦叠加时，相差180°的成分可能完全抵消。复杂声音的**相对相位与瞬态**会影响听感。因此 Vocoder 需要合成时间结构合理的波形，而不只是回填随机相位。

### 2. 传统算法与神经模型对比

- iSTFT：已知复数谱，可逆变换。
- Griffin–Lim：只知道幅度，反复估计一个与谱大体一致的相位；无需深度学习。
- Mel + Griffin–Lim：需要先近似恢复线性频率幅度谱，再估计相位，误差更大。
- HiFi-GAN：用真实 PCM/Mel 配对训练 Generator，输出波形；判别器只在训练时参加，用于鼓励真实感。

### 3. HiFi-GAN 核心结构

输入 `[B,80,T]`，Conv1D 形成高通道隐藏特征；ConvTranspose1D 按 `×8、×8、×2、×2` 等配置逐级恢复时间分辨率；MRF 多感受野残差网络强化不同时间尺度的声学结构；最终一通道 PCM。

训练时多周期判别器 MPD 更关注周期结构，多尺度判别器 MSD 观察不同时间分辨率；与频谱重建和特征匹配损失共同训练。**推理时不要运行判别器。**

### 4. 与 Neural Codec Decoder 的区别

Vocoder 的输入通常是 Mel 或声学特征；离散 Codec Decoder 的输入通常来自码本查表恢复的 Latent。两者都能使用卷积上采样网络，区别主要在输入表示、训练目标和系统分工。CosyVoice2 典型走 `Flow → Mel → HiFT`，Qwen3-TTS-12Hz 典型走 `多码本 Token → Codec Decoder`。

### 5. 实验解释

`lesson5_vocoder_lab.py` 用传统算法对同一音频做四条路线，不是预训练 HiFi-GAN；`tiny_vocoder_architecture.py` 只展示未训练网络的形状与反向传播，**生成的随机波形没有语音意义**。

录音的 Mel 参数必须匹配所用 Vocoder 的训练配置（采样率、FFT、hop、fmin/fmax、mel 定义及归一化）。不能随意拿 Whisper 的 Log-Mel 交给别的 Vocoder。

### 动手运行

从**仓库根目录**执行：

```bash
python lessons/05-vocoder-hifigan/code/lesson5_vocoder_lab.py --input lessons/05-vocoder-hifigan/code/demo_synthetic.wav --output-dir outputs/lesson05 --iterations 16
```

```bash
python lessons/05-vocoder-hifigan/code/tiny_vocoder_architecture.py
```

> 如本课含可下载权重的官方示例，先阅读 `docs/installation.md`，该步骤不计入无需网络的 CPU 实验。

### 课程附带素材 / 预期结果

- `outputs/lesson05/01_original.wav` 等四条重建对照
- `outputs/lesson05/stft_vs_logmel.png`
- `outputs/lesson05/waveforms_comparison.png`

### 验收题

- [ ] 能说明 Mel 丢失哪些信息
- [ ] 能解释为什么绝对相位与相对相位不同
- [ ] 看懂 `[B,80,100] → [B,1,25600]` 的上采样
- [ ] 区分 Generator 与判别器

### 论文与官方项目

- https://github.com/jik876/hifi-gan
- https://arxiv.org/abs/2010.05646

**延伸问答**：参见 [问答索引](docs/qa/README.md)；实验性质说明见 [证据与限制](docs/evidence-and-limitations.md)。


---

## 第06课｜CosyVoice 与 Flow Matching：从语义语音 Token 生成连续 Mel

> **本课目标**：理解语音 Token、条件速度场与 Vocoder 分工；亲手训练一个从噪声出发生成不同频谱纹理的小型 Flow 模型。

### 一眼看懂数据流

```text
文字 → Text/Speech LLM → 语音 Token → 条件 Flow Matching → Mel → HiFT Vocoder → PCM
```

### 1. 为什么有 Speech Token，还需要 Flow

CosyVoice2 的 LLM 主要决定语言内容的离散语音结构；而完整声学波形包含音色、谐波、韵律和动态细节。其 Flow 网络接受语音 Token、说话人条件、当前状态与生成进度，并构造连续 Mel，最终交由 HiFT 变成 PCM。这是系统架构选择，并不是所有 TTS 必须包含 Flow。

### 2. Flow 训练时真正的标签

真实 Mel 是 `x1`，随机高斯噪声 `x0`，训练时随机选择 `[0,1]` 中的 `t`：

```python
x_t = (1-t)*x0 + t*x1
velocity_target = x1 - x0
velocity_pred = flow_model(x_t, t, condition)
loss = ((velocity_pred - velocity_target)**2).mean()
```

这里线性插值路径只是易于理解的例子；CosyVoice 的实际实现还有特定调度、掩码、CFG 和条件网络。**训练时知道 x1，推理时不知道。**

推理以随机噪声为起点，反复调用速度网络，通过数值积分（如 Euler）逐步推进到接近真实条件数据分布的样本。

### 3. 两种时间不能混

音频时间是 Mel 的横轴，例如50Hz表示每秒50帧。Flow 时间 `t` 是生成算法从噪声到成品的进度。例如对一个 `[80,50]` Mel 块做10次 Flow 迭代，是**10次修改整张声学特征块**，而不是只产生10个真实音频帧。

### 4. Flow、Diffusion、GAN 的关系

Flow 学速度场；Diffusion 常学去噪或得分相关目标；GAN 让 Generator 与判别器对抗。它们都属于生成建模路线，但损失、采样方法、计算路径不同。增大 Flow 数值积分步数并不总是线性提升听感，步数增加会增加计算成本。

### 5. 配套实验

`toy_flow_matching.py` 是**条件生成二维合成频谱纹理**的小型 CNN 速度场，条件分别是频率上升、下降与双频率。它不生成真实中文语音。

`cosyvoice2_inference_example.py` 调用**官方预训练 CosyVoice2**，需要在上游仓库安装模型、给出参考语音与相应文字，不能当作离线 toy 脚本的一部分。

### 动手运行

从**仓库根目录**执行：

```bash
python lessons/06-cosyvoice-flow/code/toy_flow_matching.py --steps 500
```

```bash
python lessons/06-cosyvoice-flow/code/cosyvoice2_inference_example.py --help
```

> 如本课含可下载权重的官方示例，先阅读 `docs/installation.md`，该步骤不计入无需网络的 CPU 实验。

### 课程附带素材 / 预期结果

- `figures/flow_steps.png`：从噪声到合成频谱
- `figures/loss.png`：实际教学训练过程
- 训练生成的模型权重（默认不纳入版本控制）

### 验收题

- [ ] 能解释 Flow 的 x0、x1、xt、t 和速度
- [ ] 区分训练与推理的目标信息
- [ ] 区分 Flow 进度与声音真实时间
- [ ] 明确 Flow 不能直接替代 Vocoder

### 论文与官方项目

- https://github.com/FunAudioLLM/CosyVoice
- https://arxiv.org/abs/2412.10117

**延伸问答**：参见 [问答索引](docs/qa/README.md)；实验性质说明见 [证据与限制](docs/evidence-and-limitations.md)。


---

## 第07课｜Qwen3-TTS：语义码本、MTP 与 Codec 流式解码

> **本课目标**：能画出时间维 T 与码本维 K 的二维结构，理解一个大主干和小预测器如何分工，运行 toy 多码本生成。

### 一眼看懂数据流

```text
文本 Token / 音色条件 → Talker Transformer → C0 → MTP Code Predictor → C1…C15 → 12.5Hz Codec Decoder → PCM
```

### 1. 第七课为什么不再讲 Flow

**Qwen3-TTS 是模型家族**，这里重点是12.5Hz 多码本路线：它使用语义与声学码本，后接因果神经 Codec Decoder。并不意味着 Qwen3-TTS 所有版本都没有 Flow；25Hz 路线有不同的声学生成架构。

### 2. 一个80ms时间帧有多少 ID

24kHz 的80ms有1920个 PCM 采样点；12.5Hz表示每秒约12.5个音频时间帧。在12Hz路线中，一个时间位置有16个码本 ID。典型配置每组码本为2048个候选向量。2秒大约25帧，即约400个码本 ID。**不意味着大模型进行400次完整前向**。

`codes.shape ≈ [B,T,16]` 是二维时间×码本深度上的结构。在推理中，大型 Talker Backbone 主要预测语义码本 C0，再通过较小的多 Token 预测模块 MTP 预测其余码本；Codec Decoder 查表并逐步恢复 PCM。前者沿时间推进，后者沿同一时间帧的码本层级推进。

### 3. 与 CosyVoice 的系统差异

- CosyVoice2：`Speech Token → Flow Matching → Mel → HiFT Vocoder`。
- Qwen3-TTS-12Hz：`多码本 Speech Token → 训练好的神经 Codec Decoder → PCM`。

共同点是大模型不直接逐采样点生成 PCM。两条路线的首包与 RTF 优势依赖实际硬件与实现；**不能仅凭省掉 Flow 就断言整体吞吐更高**。

### 4. 如何训练

阶段一训练 Codec（录音→编码→重建录音），码本与 Encoder、Decoder 学会保留高质量声学信息。阶段二用文字/配对录音训练 Talker；对真实录音运行已训练 Codec Encoder，生成 `[T,16]` 的目标码本矩阵，再逐层用 Cross Entropy 学习对应 ID。

### 5. 实验的真实边界

`toy_multicodebook.py` 使用一个小 GRU 和 MTP 学人为定义的 ID 关系，输出合成音调；该 toy 100% 的准确率不是 Qwen3-TTS 真正的语言语音指标。`official_qwen3_tts_inference.py` 和 `official_tokenizer_encode_decode.py` 是官方模型的调用入口，需要独立安装及下载权重。

### 动手运行

从**仓库根目录**执行：

```bash
python lessons/07-qwen3-tts/code/toy_multicodebook.py --steps 400
```

```bash
python lessons/07-qwen3-tts/code/official_tokenizer_encode_decode.py --audio YOUR_AUDIO.wav --device cuda:0
```

> 如本课含可下载权重的官方示例，先阅读 `docs/installation.md`，该步骤不计入无需网络的 CPU 实验。

### 课程附带素材 / 预期结果

- `figures/training_and_codes.png`：码本训练曲线
- `samples/synthetic_*_codebooks.wav`：数学合成音调
- 真实 Codec 解码得到的 WAV（需自行运行）

### 验收题

- [ ] 能解释 `[B,T,16]`
- [ ] 能说明 C0 与 C1～C15 的区别
- [ ] 能区分 MTP 和 Transformer 主干的计算责任
- [ ] 不把模拟声和真实 TTS 音质混为一谈

### 论文与官方项目

- https://github.com/QwenLM/Qwen3-TTS
- https://huggingface.co/Qwen/Qwen3-TTS-Tokenizer-12Hz

**延伸问答**：参见 [问答索引](docs/qa/README.md)；实验性质说明见 [证据与限制](docs/evidence-and-limitations.md)。


---

## 第08课｜VAD、End-of-Turn 与打断检测

> **本课目标**：区分“停止说话”和“轮到 AI 说”，分析固定静音阈值的抢话问题，理解双工系统的 AEC/缓冲与打断控制。

### 一眼看懂数据流

```text
麦克风 PCM → VAD / 语义与声学 Turn Detector → CONTINUE / END → AI 回复；说话期间继续监听 → 打断 / 附和 / 噪声区分
```

### 1. VAD 不等于话轮结束

VAD 判断当前是否属于语音活动，能量阈值 RMS 是最基础示例；学习式 VAD 能更好地区分语音与音乐/噪声，但通常不能仅依 VAD 判断意图是否完成。

用户说“我想问一下……”，中间静音600ms，VAD 可能输出非语音，但 **Turn Detector** 可能根据语义、语调和说话节奏判定 Continue。若用户说“Whisper 为什么用 Cross Attention？”再停顿，则更倾向 End。

### 2. 固定静音阈值的代价

等450ms可能在长停顿时抢话；等1000ms减少误判，却增加用户完整说完后的响应时间。一个有效话轮检测器需要同时优化 False Cutoff Rate 与 End-of-Turn Latency。

### 3. 打断与回声消除

AI 播放时麦克风仍然接收输入：用户可能说“等一下”，也可能只是“嗯嗯”附和，或扬声器声音漏入麦克风。因此需结合 AEC 回声消除、VAD、语义、播放器控制与取消策略。

要取消的不只是 GPU 中尚在生成的音频，还包括服务器发送队列与浏览器未播放缓存。工程上可给每次回答加 `generation_id`，丢弃过期代次的音频。

### 4. 如何训练

训练标签不是整段声音类别，而是**对话中的截止位置**：同一段长录音截在尚未完成处标 Continue，截在真实终止位置标 End。随机裁短片段再继承最终 End 标签，会引入错误监督。

### 5. 本课两条代码路径

`lesson8_turn_lab.py` 使用可听的合成信号 + 人工语义标签，能重现固定超时与上下文决策的差异；它不是训练好的神经 Turn Detector。`silero_real_vad.py` 调用预训练 Silero 对你自己有权使用的语音求时间戳，但这仍然**不是**话轮语义判断。

### 动手运行

从**仓库根目录**执行：

```bash
python lessons/08-vad-turn-taking/code/lesson8_turn_lab.py
```

```bash
python lessons/08-vad-turn-taking/code/silero_real_vad.py YOUR_SPEECH.wav
```

> 如本课含可下载权重的官方示例，先阅读 `docs/installation.md`，该步骤不计入无需网络的 CPU 实验。

### 课程附带素材 / 预期结果

- `figures/turn_timeline.png`：语音活动和话轮策略
- `samples/synthetic_turns.wav`：人工合成声学片段
- 运行后 `results/events.csv`：事件日志

### 验收题

- [ ] 能解释为什么 VAD 不能替代 Turn Detector
- [ ] 理解静音等待阈值与抢话/延迟的权衡
- [ ] 解释打断为何要清理播放缓冲
- [ ] 能设计端点监督数据

### 论文与官方项目

- https://github.com/snakers4/silero-vad
- https://github.com/pipecat-ai/smart-turn
- https://docs.livekit.io/agents/

**延伸问答**：参见 [问答索引](docs/qa/README.md)；实验性质说明见 [证据与限制](docs/evidence-and-limitations.md)。


---

## 第09课｜Moshi 与 Mimi：Temporal / Depth Transformer 全双工建模

> **本课目标**：解释双音频流与可选文字流、连续因果推理、时间与码本两个自回归维度，运行逐帧 KV Cache 演示。

### 一眼看懂数据流

```text
24kHz 用户 PCM → Mimi Encoder(12.5Hz) → Temporal Transformer(KV Cache) → Depth Transformer(码本) → Mimi Decoder → AI PCM（同时继续接收用户音频）
```

### 1. Moshi 为什么和串行 ASR→LLM→TTS 不同

串行系统通常在输入分段后进行识别、文本推理与 TTS；Moshi 把用户音频、AI 自己的音频和用于指导自己输出的文字对齐到统一的时间结构中，允许持续听并按需发声。**Moshi 的 Inner Monologue 是其输出端的文字表示，不意味着必须先把用户的整句音频识别为文字。**

### 2. Mimi 压缩与两个轴

Mimi 使用24kHz音频、12.5Hz低帧率，一帧约80ms。音频编码为多组码本：语义 VQ 与声学 RVQ 组合，不能粗略地说所有层都只修正同一个语义残差。

- **Temporal Transformer**：沿音频时间位置自回归，在收到新的压缩输入时追加每层 KV，建模长期上下文。
- **Depth Transformer**：在当前生成位置的码本深度方向按依赖关系生成多码本音频表示。大型时间主干不需要对每个码本位置反复运行全套大网络。
- **Mimi Decoder**：把新输出的音频编码变回可连续播放的 PCM。

实际 Moshi 还有语义与声学编码之间的延迟对齐，不能机械认为同一80ms的8个 ID 都同时生成。

### 3. 用户持续讲话会发生什么

“因果注意力”只是规定过去位置不能看到未来位置，**不规定接口要不要等待整句话**。流式模型逐帧追加可用的输入，历史 K/V 对给定因果位置保持稳定。没有说话时，持续接收音频并不必然要求大型模型一直以完全相同的频率推理；这由部署系统是否允许休眠与恢复来决定。

### 4. 上下文与静音问题

音频静音本身有时间与节奏意义，若静音位置也送入 Temporal Transformer 就会占用上下文。可通过有限窗口/Ring KV Cache、会话休眠、摘要等处理。**有限 KV 窗口不等于拥有长期记忆**；长会话若想记得数十分钟前的内容还需外部状态或摘要。

### 5. 实验的边界

`lesson9_moshi_toy.py` 训练的是使用合成的 Listen/Speak/Stop 标签的小型因果 Temporal 和码本 Depth 模型，并输出左/右声道的数学音调；**它不是公开 Moshi 的人声质量或实际毫秒级延迟复现**。

### 动手运行

从**仓库根目录**执行：

```bash
python lessons/09-moshi-full-duplex/code/lesson9_moshi_toy.py --steps 350 --output outputs/lesson09
```

> 如本课含可下载权重的官方示例，先阅读 `docs/installation.md`，该步骤不计入无需网络的 CPU 实验。

### 课程附带素材 / 预期结果

- `figures/full_duplex_timeline.png`：逐帧状态与 KV
- `samples/full_duplex_stereo.wav`：两路合成音调
- `outputs/lesson09/per_frame_log.csv`：逐帧输出

### 验收题

- [ ] 解释 Temporal 与 Depth 两个维度
- [ ] 解释历史 K/V 为什么可以缓存
- [ ] 能听出双方同时发声与打断的时间重叠
- [ ] 明确合成策略与真实全双工能力的区别

### 论文与官方项目

- https://github.com/kyutai-labs/moshi
- https://arxiv.org/abs/2410.00037

**延伸问答**：参见 [问答索引](docs/qa/README.md)；实验性质说明见 [证据与限制](docs/evidence-and-limitations.md)。


---

## 第10课｜架构收官：WebRTC、模型 Worker、Session、KV、打断与长会话

> **本课目标**：设计完整 S2S 低延迟服务，运行控制面模拟器，区分 GPU 计算、网络、音频播放与上下文管理。

### 一眼看懂数据流

```text
浏览器麦克风(WebRTC/AEC) ↔ Realtime Media Gateway ↔ Session Orchestrator ↔ Streaming Model Worker(Encoder/LLM/Decoder) ↔ PCM 播放
```

### 1. 模型架构与系统架构不是一回事

一个训练好的端到端 S2S 模型不等于可提供长期服务的产品。需要媒体网关（网络传输、抖动缓冲、重采样、AEC）、会话路由、GPU 模型 Worker、打断与播放控制、指标与异常恢复。原生模型可将感知、话轮与声音生成整合在一个主干，不意味着所有子功能都必须拆成独立服务。

### 2. 一次用户对话的时间关系

持续接收20ms PCM帧；模型可能每80ms形成一段 Codec 时间位置；若用户尚未说完，继续监听但不必一定提交回答；确认说完后，模型生成第一块音频就立即送往播放队列；用户打断时取消正在生成、排队和浏览器待播放的旧音频。

`generation_id` 是一种推荐的幂等和防旧帧机制；它不是特定官方协议强制要求的字段。

### 3. KV Cache 管理

每个活跃会话保存不同的每层 KV；模型权重通常可以跨会话共享。Prefix Cache 复用相同前缀，而 Session KV 保持连续时间上下文，两者机制不同。音频场景的历史长度可能很快增长：12.5Hz 一小时就是45,000个模型时间位置（不代表模型可支持这么长的有效上下文）。

Ring KV 固定单会话的注意力窗口和显存上限；更长时间重要信息则需要摘要、外部记忆、替换实例或重新 Prefill。静音不是免费的历史：如果仍输入主 Transformer，它可能持续消耗上下文位置。

### 4. 实时性能指标

- 用户表达完成到第一段**实际播放**声音的延迟，而不只是网络收到第一 Token。
- 有效打断到旧音频停止播放的延迟。
- Frame processing deadline、输出缓冲欠载、输入抖动、GPU 调度尾延迟、KV 占用及多会话负载。

例如模型时间帧为80ms，若长期串行处理一帧超过80ms，队列会积压。但平均低于80ms也不保证 P95/P99 不会破坏播放节奏。

### 5. 外部项目的参考路线

- Moshi：原生多流语音建模，适合研究模型内部逐帧推理。
- Sophon-Realtime：模型家族及话轮/打断控制，适合集成研究。
- LiveKit：WebRTC 和语音 Agent 编排，适合搭产品。
- OpenAI GPT-Live：公开了媒体前端、有状态连续推理、异步任务、上下文压缩等工程思想；**内部模型参数和精确实现并未全部公开**。

### 6. 本课实验是调度模拟，不是性能压测

`lesson10_simulator.py` 用合成音调和虚拟时间验证持续监听、输出重叠、增量 KV 等效和旧代次取消。280ms等数字来自**人为设定的模拟时钟**，不得在 README 宣称为实测 OpenAI/Moshi/Sophon 低延迟。

### 动手运行

从**仓库根目录**执行：

```bash
python lessons/10-realtime-system/code/lesson10_simulator.py --out outputs/lesson10
```

> 如本课含可下载权重的官方示例，先阅读 `docs/installation.md`，该步骤不计入无需网络的 CPU 实验。

### 课程附带素材 / 预期结果

- `figures/timeline.png`：输入/播放/KV 时间线
- `samples/simulated_duplex_stereo.wav`：双路模拟声音
- `outputs/lesson10/{frames.csv,events.json,metrics.json}`：运行日志

### 验收题

- [ ] 能画出完整 S2S 服务拓扑
- [ ] 能说明停止 GPU 不等于浏览器立即停止旧音频
- [ ] 能解释 Session KV 与 Prefix Cache
- [ ] 能区分模拟 RTT 与真实端到端测量

### 论文与官方项目

- https://github.com/kyutai-labs/moshi
- https://github.com/adventists-ai/sophon-realtime
- https://openai.com/index/continuous-voice-interaction-with-gpt-live/
- https://docs.livekit.io/agents/

**延伸问答**：参见 [问答索引](docs/qa/README.md)；实验性质说明见 [证据与限制](docs/evidence-and-limitations.md)。
