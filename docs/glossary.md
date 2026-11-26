# 术语表（从图像学习者的视角）

| 术语 | 简单解释 | 图像领域类比 |
|---|---|---|
| PCM | 按时间记录的数字振幅采样 | 按位置排列的像素数组 |
| Sample Rate | 每秒读取多少次声音采样 | 空间采样密度（不完全等价） |
| FFT | 将一段波形分解到频率坐标 | 2D傅里叶变换的1D版本 |
| STFT | 按窗得到时间-频率矩阵 | 在不同局部窗口抽取频率特征 |
| Phase | 各频率分量的相对振动位置 | 频域相位决定空间/时间结构 |
| Mel | 非线性听觉频带聚合 | 手工设计的特征变换 |
| Log-Mel | 对Mel功率谱做对数变换 | 预处理后的单通道特征图 |
| Audio Encoder | 将声音编码成模型中间表示 | 图像 Encoder / ViT |
| Decoder | 从隐藏表示/上下文生成目标数据 | 图像 Decoder / Seq2Seq Decoder |
| Self Attention | 序列内部的 Query 读同序列 Keys/Values | ViT 中 Token 互相关注 |
| Cross Attention | 当前序列的 Q 访问另一序列 K/V | 文本向图像特征做条件查询 |
| Tokenizer/Vocab | 文本片段与整数 ID 的对应规则 | 离散图像 Token 的索引规则 |
| Embedding | ID 查表得到训练过的连续向量 | Patch / 类别向量嵌入 |
| Codebook | 有限组可训练向量 | VQ-VAE 的离散潜变量字典 |
| VQ | 连续向量查最近码本 | 把连续像素特征离散化 |
| RVQ | 多个码本依次修正量化残差 | 多级/分层残差量化 |
| STE | 为不可导量化步骤人为指定近似梯度 | 量化网络常用训练技巧 |
| Codec | 压缩音频为表示并可重建 | 图像 VQ-VAE/有损压缩器 |
| Vocoder | 从声学特征合成波形 | 从潜变量重建图像的 Decoder |
| Mel Vocoder | Mel → PCM 的专门解码器 | 特定 Latent → Image Decoder |
| Flow Matching | 学从噪声走向真实分布的速度场 | 图像 Flow / Diffusion 类生成 |
| Euler | 按预测速度进行数值积分 | 数值求解生成轨迹 |
| VAD | 判断当前有没有语音活动 | 简单时序事件检测 |
| End-of-Turn | 判断用户是否表达完成 | 时间上下文分类 |
| Barge-in | AI 正说话时用户有效打断 | 会话控制事件 |
| Duplex | 同时监听、同时说话 | 双向流的同步建模 |
| KV Cache | 每个 Transformer 层保留历史 K/V | 自回归时序计算复用 |
| Ring KV | 固定容量覆盖旧位置的 KV | 滑动历史窗口 |
| Dense Model | 主要参数路径对每个 Token 相同 | 常规 CNN/Transformer |
| MoE | Router 按Token选择少数专家网络 | 动态选择多个子网络 |
| RTF | 音频合成时间/音频时长 | 处理速度比 |
| TTFA | 从指定起点到首次**可听**音频的延迟 | 真实首帧交付时间 |
