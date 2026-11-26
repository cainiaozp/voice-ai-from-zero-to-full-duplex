# 安装、运行、复现策略

## 分层依赖与 Python 版本

优先 Python 3.10 或 3.11。PyTorch、Torchaudio 和 GPU CUDA 版本必须按官方矩阵配套安装。`requirements/base.txt` 足以运行大部分数字信号处理实验。`requirements/pytorch.txt` 适合 CPU 神经网络 toy；Whisper、Codec 和 TTS 的大型预训练权重是可选进阶内容。

```bash
# 1. 数字信号处理
pip install -r requirements/base.txt

# 2. 如果需要训练 toy 神经网络
pip install -r requirements/pytorch.txt

# 3. 如果运行 Hugging Face Whisper
pip install -r requirements/whisper.txt
```

`requirements/pretrained-codecs.txt` / `pretrained-tts.txt` 用于给出依赖方向，**不保证自动解决所有 CUDA ABI 冲突**。建议每种上游模型一个独立虚拟环境。

## 什么叫真正可复现

至少记录：代码 commit、Python/torch/CUDA、GPU 类型、模型版本、采样率、随机种子、输入 WAV 的来源和授权、前处理参数、推理延迟测点。toy 学习结果与真实模型指标不可混用。

## 推荐练习顺序

1. 第01课：`lesson1.py`，确认 STFT 与 Mel 形状。
2. 第03课：`cross_attention_toy.py`，确认矩阵乘法与 Softmax。
3. 第02课：`train_audio_cnn.py --epochs 1`，快速检查梯度，随后完整6轮。
4. 第04/05课：从随机合成音频重建，听懂相位与 RVQ。
5. 第06/07课：CPU 上训练条件生成 toy，观察每个生成轴。
6. 第08课：VAD/话轮模拟，随后换成 Silero 真实 VAD。
7. 第09/10课：输入+输出并行、增量 KV、代次取消。
8. 有 GPU 后再部署真实 Whisper / CosyVoice / Qwen3-TTS / Moshi / Sophon，并测真实音频与服务延迟。

## 运行地点

本仓库内多数脚本以自身路径作为输出目录；默认 `results/`、`outputs/` 以及 `.pt` 权重均已被 `.gitignore` 排除。课程 `figures/` 和 `samples/` 为可公开的教学示意结果，不需要每次都重新生成才能看讲义。

## GitHub 安全发布

不要把自己的发音录音、用户数据、个人语音克隆参考、录音笔内容、ESC-50 原始 WAV、下载的模型权重、API Key 放到公开 Git 历史中。如果文件曾经提交过，即使后来删除还可能存在于 Git 历史，需要单独清理。
