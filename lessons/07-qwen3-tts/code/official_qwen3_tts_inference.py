"""使用官方 qwen-tts 包生成中文语音（需要联网下载权重和支持 CUDA 的 GPU）。
运行前：pip install -U qwen-tts soundfile
官方参考：https://github.com/QwenLM/Qwen3-TTS
"""
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    device_map="cuda:0",
    dtype=torch.bfloat16,
)

print("Supported speakers:", model.get_supported_speakers())
wavs, sr = model.generate_custom_voice(
    text="你好，我正在学习语音模型的多码本生成原理。",
    language="Chinese",
    speaker="Vivian",
)
sf.write("qwen3_tts_output.wav", wavs[0], sr)
print("Saved qwen3_tts_output.wav, sr=", sr)
