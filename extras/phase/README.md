# 相位实验：为什么 180° 单听一样，叠加却可能消失？

```bash
pip install -r requirements/base.txt
python extras/phase/phase_lab.py --out outputs/phase
```

`single_0.wav` 与 `single_180.wav` 单独听通常是同一个音高；`sum_180.wav`（理想同频同幅且对齐）应接近静音；`sum_0.wav` 则增强。`phase_comparison.png` 画出波形。这里是数学理想案例，实际双扬声器房间内受反射与位置影响。
