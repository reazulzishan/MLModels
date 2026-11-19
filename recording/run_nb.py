import sys, time, nbformat
from pathlib import Path
from nbclient import NotebookClient
src, dst = sys.argv[1], sys.argv[2]
nb = nbformat.read(src, 4); t = time.time()
try:
    NotebookClient(nb, timeout=7200, kernel_name="python3", resources={"metadata": {"path": str(Path(__file__).resolve().parents[1] / "notebooks")}}).execute()
    print("OK", round((time.time() - t) / 60, 1), "min")
finally:
    nbformat.write(nb, dst)
    for i, c in enumerate(nb.cells):
        if c.cell_type == "code":
            for o in c.get("outputs", []):
                if o.get("output_type") == "error": print("ERROR in cell", i, o["ename"], o["evalue"][:300])
                txt = o.get("text", "") or str(o.get("data", {}).get("text/plain", ""))
                if txt and any(k in txt for k in ("PR-AUC", "official test", "epoch", "LOCKED", "device")): print(f"[{i}]", txt.strip()[:600])
