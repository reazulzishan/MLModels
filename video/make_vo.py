import json, re, sys
from pathlib import Path
import numpy as np, soundfile as sf
sys.path.insert(0, str(Path.home() / "Movies/CS Funda/assets/voice"))
from cs_funda_voice import MODEL, SPEAKER, prep
from TTS.api import TTS
here = Path(__file__).parent
items = re.findall(r"\[([YR]\d+)\]\n(.+?)(?=\n\[|\n\n#|\Z)", (here / "script.md").read_text(), re.S)
SAY = [("LightGBM", "Light G B M"), ("XGBoost", "X G Boost"), ("LSTM", "L S T M"), ("CNN", "C N N"), ("LLMs", "L L Ms"), ("ppm", "P P M"),
       ("bKash", "bee Kash"), ("SMS", "S M S"), ("ScamGuard", "Scam Guard"), ("PIN", "pin"), ("Mauna Loa", "Mauna Low-ah")]
only = set(sys.argv[1:])
tts = TTS(MODEL, progress_bar=False); sr = tts.synthesizer.output_sample_rate
meta = json.loads((here / "vo/scenes.json").read_text()) if (here / "vo/scenes.json").exists() else {}
for sid, text in items:
    if only and sid not in only: continue
    text = " ".join(text.split()); t2 = text
    for a, b in SAY: t2 = re.sub(rf"\b{re.escape(a)}\b", b, t2)
    w = np.asarray(tts.tts(text=prep(t2), speaker=SPEAKER), dtype=np.float32)
    sf.write(here / f"vo/{sid}.wav", w, sr)
    meta[sid] = {"id": sid, "text": text, "dur": round(len(w) / sr, 2)}
    print(sid, meta[sid]["dur"], flush=True)
(here / "vo/scenes.json").write_text(json.dumps(meta, indent=1))
print("total Y", round(sum(m["dur"] for k, m in meta.items() if k[0] == "Y"), 1), "| total R", round(sum(m["dur"] for k, m in meta.items() if k[0] == "R"), 1))
