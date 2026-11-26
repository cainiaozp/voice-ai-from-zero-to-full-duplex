"""Static structural checks; no network or model downloads needed."""
from pathlib import Path
import ast
import sys

ROOT=Path(__file__).resolve().parents[1]
LESSONS=[
 '01-pcm-stft-logmel','02-audio-cnn-esc50','03-whisper-asr','04-neural-codec',
 '05-vocoder-hifigan','06-cosyvoice-flow','07-qwen3-tts',
 '08-vad-turn-taking','09-moshi-full-duplex','10-realtime-system']

def main():
    errors=[]
    for lesson in LESSONS:
        p=ROOT/'lessons'/lesson
        if not (p/'README.md').is_file(): errors.append(f'missing readme: {lesson}')
        if not list((p/'code').glob('*.py')):errors.append(f'no Python lab: {lesson}')
    count=0
    for f in ROOT.rglob('*.py'):
        if f.name=='verify_repo.py':continue
        try:ast.parse(f.read_text(encoding='utf8'),filename=str(f))
        except SyntaxError as e:errors.append(f'syntax: {f.relative_to(ROOT)}:{e.lineno}: {e.msg}')
        count+=1
    for rel in ['README.md','GITHUB_UPLOAD.md','docs/qa/README.md','docs/qa/05-esc50-case-study.md','LICENSE']:
        if not (ROOT/rel).exists():errors.append(f'missing {rel}')
    print(f'Checked {len(LESSONS)} lessons, {count} Python files')
    if errors:
        for e in errors:print('FAIL:',e)
        raise SystemExit(1)
    print('PASS: essential structure and Python syntax')

if __name__=='__main__':main()
