# Transformer / QKV / Whisper / 词表

> 本文是依据本次课程中**可见的提问与讲解按主题整理、重新表述**的学习问答。
> 不是聊天平台导出的逐字原始记录；某些问题合并了多轮追问。

---

## Q01：传统 LLM 与实时模型都用因果注意力，为什么一个等输入完成，一个持续处理？

**答：** 注意力掩码约束能看哪些位置，实时持续推进取决于输入表示、训练和服务调度。传统 API 常按请求 prefill；流式音频模型可按帧输入、增量缓存历史 K/V。

**推导 / 直觉：** 对同一个固定因果序列一次性计算与增量计算，输出在浮点误差范围内一致。

**课程位置：**Transformer专题/第09课

---

## Q02：Q、K、V 分别从哪里来？

**答：** 当前层输入隐藏向量 X 分别乘以可训练矩阵 WQ、WK、WV；不是人为指定某个头负责语义、某个头负责语音。

**推导 / 直觉：** 手算 Q=XWQ、K=XWK、V=XWV，再求 attention softmax。

**课程位置：**Transformer专题

---

## Q03：WQ、WK、WV 是怎样学出来的？

**答：** 训练初始化后由下一个Token预测等损失反向传播优化；这些权重保存在模型参数文件和 GPU 推理权重中。

**推导 / 直觉：** 检视 safetensors 中 attention.q_proj/k_proj/v_proj 等参数形状。

**课程位置：**Transformer专题

---

## Q04：为什么 Transformer 分层、每层都有自己的 KV？

**答：** 每层的隐藏状态不同、投影矩阵不同，故每层需要缓存对应的 K/V；只保存最终层 K/V 不能代替前层的注意力计算。

**推导 / 直觉：** 打印各层 cache shape，而不是只看总显存。

**课程位置：**Transformer专题/第09课

---

## Q05：KV Cache 存储在哪里？

**答：** 通常是推理进程的 GPU 显存张量，与磁盘中的模型权重不同；可有 CPU/NVMe 卸载等变种。缓存生命周期由服务会话调度决定。

**推导 / 直觉：** 在推理服务中观察 KV pool 指标与独立会话。

**课程位置：**Transformer专题/第10课

---

## Q06：为什么 GPU 显存占用很多，但利用率却是0%？

**答：** 权重和 KV Cache 常驻显存不等于 GPU 算术单元持续运行。nvidia-smi GPU-Util 近似反映采样窗口内设备忙碌时间；没有请求时可能0%。

**推导 / 直觉：** 在空闲和压测期间分别观察 GPU-Util、显存与请求队列。

**课程位置：**推理部署专题

---

## Q07：增量计算是不是只算新 Token 就完全不用历史？

**答：** 只需新增历史 K/V，不重算旧投影；新 Query 仍须读取历史 K，做点积与加权聚合，所以计算不是常数开销。

**推导 / 直觉：** 比较 Q 长度与 K 长度：Q可能为1，K会随上下文增长。

**课程位置：**Transformer专题

---

## Q08：GQA / MLA 如何减少 KV 显存？

**答：** GQA 让更多 Q 头共享较少 KV 头；MLA 可在特定架构中缓存低秩潜在表示。它们与 PagedAttention 这种缓存分配策略是不同维度的优化。

**推导 / 直觉：** 用公式 `2×layers×T×KV_heads×head_dim×bytes` 估算。

**课程位置：**Transformer专题

---

## Q09：Whisper 是不是类似翻译模型？

**答：** 是 Encoder–Decoder 风格：翻译从源文字编码到目标文字，Whisper 从语音时频特征编码到文字。Decoder 都通过 Cross Attention 读取源序列。

**推导 / 直觉：** 绘制两者的 Q 文本 / K,V 源序列。

**课程位置：**第03课

---

## Q10：为什么 Whisper Encoder 不能直接输出文字？

**答：** Encoder 保留时序音频隐藏特征；Decoder 需要自回归地产生长度不确定的文字 Token 序列。

**推导 / 直觉：** 观察 `[B,1500,384]` 与词表 logits `[B,L,V]`。

**课程位置：**第03课

---

## Q11：Cross Attention 的 Q/K/V 来自哪里？

**答：** Q 来自文字 Decoder 当前状态，K/V 来自音频 Encoder；这让文字预测查询对应音频信息。

**推导 / 直觉：** 运行 `cross_attention_toy.py`，检验权重矩阵形状 `[L,T]`。

**课程位置：**第03课

---

## Q12：词表 Vocabulary 起什么作用？

**答：** Token ID 与文字/字节片段由 Tokenizer 关联；Decoder 的 Embedding 负责 ID→连续向量，输出层对全词表候选打分。语音 Encoder 本身不使用文字词表把声音查成汉字。

**推导 / 直觉：** 运行 `openai-whisper` tokenizer 的 encode/decode 对照。

**课程位置：**第03课

---

## Q13：Whisper 能不能翻译？

**答：** 多语言 Whisper 通过任务特殊 Token 指定 transcribe 或 translate；经典 translate 主要为音频转英语文字，不代表任意语种互译。

**推导 / 直觉：** 查看生成前缀语言/任务 Token。

**课程位置：**第03课

---
