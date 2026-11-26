# 第九课：Moshi 多流全双工模型教学实验

该代码是**教学模型**，不是 Moshi 的权重，也不代表 Moshi 的声音质量或毫秒级延迟。

## 内容
- `lesson9_moshi_toy.py`：训练一个带两层因果注意力和逐层 KV Cache 的微型 Temporal Transformer，预测 LISTEN/SPEAK/STOP；训练一个在码本维度自回归的微型 Depth Transformer；逐帧推理；合成双声道音调来演示两路同时活跃。
- `results/full_duplex_timeline.png`：用户输入、模型决策、双声道波形、KV Cache 长度。
- `results/full_duplex_stereo.wav`：左声道模拟用户音调，右声道模拟 AI 音调。**它不是真实语言声音。**
- `results/per_frame_log.csv`：每帧的时间、用户输入、动作、生成码本、各层缓存长度。
- `results/train_history.csv`：小模型训练曲线数据。
- `results/toy_weights.pt`：教学模型状态字典。

## 安装与运行
```bash
pip install torch numpy matplotlib soundfile
python lesson9_moshi_toy.py --steps 350 --output results
```

## 应看到的形状
- 每帧 PCM：1920 个 24kHz 采样点，持续 80ms。
- 用户活动张量：`[1,1]`（这里只用 0/1 表示活动，不是真正 Mimi Token）。
- Temporal：两层，每层四个 KV Head、每头12维，55帧推理完成缓存形状 `[1,4,55,12]`。
- Depth：8组码本，每组教学词表32个候选 ID，真实 Mimi 通常为2048个候选 ID。
- 训练数据是人工制造的行为规则，非真实对话音频及文本数据。

## 结果解释
- 被标注为「SPEAK」时，模型生成教学码本 ID（如 `[5,8,11,14,17,20,23,26]`）。这些 ID **不能传入 Mimi Decoder**。
- 对比 full causal attention 与 incremental KV-cache，最大 logit 差在浮点误差范围内（本次测试约 2.4e-6）。
- 由于声道播放人为设置了一帧缓冲，用户中断时可以观察到短时双声道重叠；该80ms不是实际 Moshi 的测量。
- 语音活动和轮次标签是人工监督，不是用户音频与模型的真实联合建模。

## 官方资料
- https://arxiv.org/abs/2410.00037
- https://github.com/kyutai-labs/moshi
