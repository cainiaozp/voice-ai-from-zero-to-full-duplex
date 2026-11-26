# ESC-50 dog 黄屏归一化故障复现（合成数据）

这不是 ESC-50 数据集本身，而是用5秒合成声音复现**每秒独立峰值归一化**导致静音背景变亮的问题。

```bash
pip install -r requirements/base.txt
python extras/esc50-normalization/compare_normalization.py --out outputs/esc50_compare.png
```

把 `--input YOUR_AUDIO.wav` 设为自己有权使用的 WAV，可对照真实录音。输出两个图：每1秒局部峰值和整5秒统一峰值。原会话 dog 的80.1%等数字来自用户当时的统计，不是这份模拟脚本的预期值。

参见 [归一化问题详解](../../docs/qa/05-esc50-case-study.md)。
