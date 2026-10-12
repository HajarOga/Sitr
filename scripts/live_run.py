"""One real run: load the model, analyse the sample document, print everything."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import config, core
from app.document_processing import readers
from app.llm import engine

t0 = time.time()
state = engine.install_state()
print(f"installed={state['installed']} device={engine.resolve_device()} "
      f"dtype={engine.resolve_dtype(engine.resolve_device())}", flush=True)

path = sorted((config.DATA_DIR / "synthetic").glob("*.txt"))[0]
stored = readers.store_upload(path.name, path.read_bytes())
document = readers.read(stored, path.name)
readers.discard(stored)
print(f"document: {path.name}, {len(document.blocks)} block(s), "
      f"{len(core.document_text(document))} chars", flush=True)

print("\nloading the model (first call)...", flush=True)
engine.load()
print(f"loaded in {time.time() - t0:.1f}s", flush=True)

engine.reset_calls()
started = time.time()
analysis = core.analyze(document, "EXTERNAL_VENDOR", "a vendor is fixing the line")
print(f"\nanalysis finished in {time.time() - started:.1f}s "
      f"({analysis.summary()['model_calls']} model calls, "
      f"{analysis.summary()['model_seconds']}s of model time)\n", flush=True)

print(f"{'value':34} {'type':16} {'conf':5} {'rule':16} action")
for d in analysis.detections:
    print(f"{d.original_text[:33]:34} {d.type:16} {d.confidence:<5} "
          f"{str(d.rule_id):16} {d.action}")

print("\nreasons:")
for d in analysis.detections:
    print(f"  {d.type:16} {d.rule_id or '-':16} {str(d.reason)[:100]}")

print(f"\nguard notes ({len(analysis.notes)}):")
for note in analysis.notes:
    print("  *", note)

print("\nAI summary:", analysis.summary_text)
print("\nmodel calls:")
for call in analysis.calls:
    print(f"  {call['purpose']:24} {call['tokens_out']:>5} tok  {call['seconds']:>6}s  "
          f"{call['device']}")

applied = core.apply_decisions(analysis)
print(f"\nrelease: {len(applied['applied'])} applied, {len(applied['pending'])} left out "
      f"for a person, incomplete={applied['incomplete']}")
for change in applied["applied"]:
    print(f"  {change['action']:8} {change['type']:16} {change['masked']:22} "
          f"→ {change['replacement'] or '(removed)'}  [{change['decided_by']}]")
