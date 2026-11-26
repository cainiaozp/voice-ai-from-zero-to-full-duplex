# Lesson 8: VAD, turn-taking, barge-in

1. Install: `pip install numpy matplotlib soundfile`
2. Run: `python lesson8_turn_lab.py`
3. Open `results/turn_timeline.png`; play `results/synthetic_turns.wav` (harmonic sound **NOT intelligible speech**).
4. Edit `threshold = 0.014` or the `.45` fixed endpoint delay and run again.
5. For real pre-trained VAD: `pip install torch silero-vad soundfile numpy scipy`; record `voice.wav`, run `python silero_real_vad.py voice.wav`. First run may download model weights.

## Important distinction

- The primary lab illustrates frame-wise RMS **energy gating**, not Silero VAD.
- The context-aware policy uses hand-labeled utterance status for teaching, **not a trained turn-detector** and not automatically transcribed.
- The last audio burst is an illustrative user interruption while a hypothetical agent was playing audio. No echo canceller, STT, or duplex model is implemented in the lab.
