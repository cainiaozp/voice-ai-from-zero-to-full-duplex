"""使用官方 Qwen3-TTS-Tokenizer-12Hz 直接编码/解码一段 WAV 音频。
运行前：pip install -U qwen-tts soundfile
适合验证 12.5Hz 的多码本表示；真实模型需要 CUDA 和预训练权重。
"""
import argparse
import soundfile as sf
from qwen_tts import Qwen3TTSTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--audio', required=True, help='本地 WAV 文件路径')
parser.add_argument('--device', default='cuda:0')
args = parser.parse_args()

codec = Qwen3TTSTokenizer.from_pretrained(
    'Qwen/Qwen3-TTS-Tokenizer-12Hz',
    device_map=args.device,
)
encoded = codec.encode(args.audio)
print('encoded Python type:', type(encoded))
print('encoded example:', str(encoded)[:400])
wavs, sr = codec.decode(encoded)
sf.write('qwen3_codec_reconstruction.wav', wavs[0], sr)
print('Decoded:', len(wavs[0]), 'samples, sampling rate=', sr)
