import json
from datetime import datetime
from pathlib import Path

# --- Persistence ---
# saved_portfolios.json stores named long/short combinations with their metrics
SAVED_PATH = Path(__file__).parent / 'saved_portfolios.json'


# --- Load ---

def load_saved() -> list[dict]:
    if not SAVED_PATH.exists():
        return []
    try:
        return json.loads(SAVED_PATH.read_text())
    except Exception:
        return []


# --- Save / Upsert ---

def save_portfolio(
    name: str,
    long_flags: dict,
    short_flags: dict,
    long_display: str,
    short_display: str,
    metrics: dict,
) -> None:
    saved = load_saved()
    entry = {
        'name': name,
        'long_flags': long_flags,
        'short_flags': short_flags,
        'long_display': long_display,
        'short_display': short_display,
        'metrics': {k: round(float(v), 4) for k, v in metrics.items()},
        'saved_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
    }
    # Upsert: replace an existing entry with the same name, otherwise append
    for i, s in enumerate(saved):
        if s['name'] == name:
            saved[i] = entry
            break
    else:
        saved.append(entry)
    SAVED_PATH.write_text(json.dumps(saved, indent=2))


# --- Delete ---

def delete_portfolio(name: str) -> None:
    saved = [s for s in load_saved() if s['name'] != name]
    SAVED_PATH.write_text(json.dumps(saved, indent=2))
