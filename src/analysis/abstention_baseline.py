"""
Abstention-enabled LLM annotation baseline (camera-ready, addresses Reviewer 3AuY).

Relation to the original pipeline (batched chat-interface annotation):
  * Same system-prompt body (prompts/annotation_prompt.md: role, principles,
    label definitions and examples), loaded and modified in code.
  * Differences from the original: one comment per call instead of numbered
    batches of ~20-50; a different user-message wording; JSON instead of CSV
    output; API access via OpenRouter with provider-default sampling/reasoning
    instead of the chat interfaces.
  * The --forced run keeps the original forced-choice instruction; the default
    run additionally allows a No-Persona abstention. forced vs default therefore
    isolates abstention; original vs --forced is the format/access difference.
  * The No-Persona definition given to the models adds examples (generic praise,
    remarks addressed to the channel) beyond the human guideline wording.
  * SAME three models (called via OpenRouter), run on the 300 human-gold
    comments (data/splits/test.csv).

The exact prompt used is saved to results/abstention_baseline/system_prompt_used.txt.
Results are cached to JSONL, so the script is resumable.

Usage (from repo root):
  python src/analysis/abstention_baseline.py --limit 10      # smoke test
  python src/analysis/abstention_baseline.py                 # full run
  python src/analysis/abstention_baseline.py --models claude # one model
"""
import argparse
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

# ----------------------------------------------------------------- CONFIG
GOLD_PATH = "data/splits/test.csv"          # 300 human-gold comments
ID_COL, TEXT_COL = "comment_id", "comment"
PROMPT_MD = "prompts/annotation_prompt.md"
OUT_DIR = Path("results/abstention_baseline")

# All three models are called through OpenRouter (OpenAI-compatible API).
# Verify each slug at https://openrouter.ai/models before the full run.
# temperature=None -> provider default (matches the original, which used defaults).
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODELS = {
    "gpt":    {"model": "openai/gpt-5.2",              "temperature": None},
    "gemini": {"model": "google/gemini-3.5-flash",     "temperature": None},
    "claude": {"model": "anthropic/claude-sonnet-4.6", "temperature": None},
}

LABELS = {
    "intent": ["Pro-Migration", "Anti-Migration", "Trapped/Regretful", "Neutral/Observation"],
    "primary_driver": ["Economic Necessity", "Family Obligation",
                       "Systemic/Political Anger", "Patriotism/Love"],
    "value_orientation": ["Collectivist-Family", "Collectivist-Nation", "Individualist-Self"],
    "affect": ["Despairing/Sad", "Angry/Frustrated", "Hopeful/Motivated", "Pragmatic"],
}
NO_PERSONA = "No-Persona"
MAX_ATTEMPTS = 3

# Change 1: forced choice -> choice with abstention.
FORCED_CHOICE = ("Select EXACTLY ONE label from EACH of the four categories. Pick the best fit\n"
                 "even when signals are mixed; choose the dominant one.")
ABSTAIN_CHOICE = (
    "First decide whether the comment expresses a persona about migration or brain\n"
    "drain at all, i.e. a stance, motivation, value, or emotion about leaving or\n"
    "staying in Nepal. If it does not (for example generic praise of the video or\n"
    "speaker, a remark addressed to the channel, or content unrelated to migration),\n"
    "mark it No-Persona and assign no category labels. Otherwise, select EXACTLY ONE\n"
    "label from EACH of the four categories. Pick the best fit even when signals are\n"
    "mixed; choose the dominant one.")

# Change 2: CSV/no-blanks output block -> per-comment JSON with abstention.
JSON_OUTPUT = """================================================================================
OUTPUT FORMAT
================================================================================
You will be given ONE comment. Return ONLY a JSON object, no other text:

{"no_persona": true|false, "intent": <label|null>, "primary_driver": <label|null>,
 "value_orientation": <label|null>, "affect": <label|null>}

If "no_persona" is true, set all four labels to null. If it is false, give a valid
label for all four categories.

Valid values per field:
- intent            : Pro-Migration | Anti-Migration | Trapped/Regretful | Neutral/Observation
- primary_driver    : Economic Necessity | Family Obligation | Systemic/Political Anger | Patriotism/Love
- value_orientation : Collectivist-Family | Collectivist-Nation | Individualist-Self
- affect            : Despairing/Sad | Angry/Frustrated | Hopeful/Motivated | Pragmatic
"""
# Control condition (--forced): original forced-choice instruction, per-comment JSON, no abstention.
JSON_OUTPUT_FORCED = """================================================================================
OUTPUT FORMAT
================================================================================
You will be given ONE comment. Return ONLY a JSON object, no other text:

{"intent": <label>, "primary_driver": <label>, "value_orientation": <label>, "affect": <label>}

Every comment gets all four labels, no blanks.

Valid values per field:
- intent            : Pro-Migration | Anti-Migration | Trapped/Regretful | Neutral/Observation
- primary_driver    : Economic Necessity | Family Obligation | Systemic/Political Anger | Patriotism/Love
- value_orientation : Collectivist-Family | Collectivist-Nation | Individualist-Self
- affect            : Despairing/Sad | Angry/Frustrated | Hopeful/Motivated | Pragmatic
"""
FORCED = False  # set by --forced
# ------------------------------------------------------------------------


def build_system_prompt():
    md = Path(PROMPT_MD).read_text(encoding="utf-8")
    sec = md.split("## System Prompt", 1)[1]
    system = re.search(r"```\n(.*?)\n```", sec, re.DOTALL).group(1)
    assert system.count(FORCED_CHOICE) == 1, "forced-choice sentence not found verbatim"
    if not FORCED:
        system = system.replace(FORCED_CHOICE, ABSTAIN_CHOICE)
    cut = system.index("OUTPUT FORMAT")
    cut = system.rfind("\n", 0, system.rfind("=" * 20, 0, cut))
    return system[:cut].rstrip() + "\n\n" + (JSON_OUTPUT_FORCED if FORCED else JSON_OUTPUT)


_clients, _client_lock, _write_lock = {}, threading.Lock(), threading.Lock()
_LOOKUP = {a: {l.lower(): l for l in ls} for a, ls in LABELS.items()}


def get_client():
    with _client_lock:
        if "or" not in _clients:
            import os
            from openai import OpenAI
            _clients["or"] = OpenAI(base_url=OPENROUTER_BASE_URL,
                                    api_key=os.environ["OPENROUTER_API_KEY"])
        return _clients["or"]


def call_model(cfg, system, user):
    kw = dict(model=cfg["model"],
              messages=[{"role": "system", "content": system},
                        {"role": "user", "content": user}])
    if cfg["temperature"] is not None:
        kw["temperature"] = cfg["temperature"]
    r = get_client().chat.completions.create(**kw)
    return r.choices[0].message.content


def parse_json(text):
    if not text:
        return None
    m = re.search(r"\{.*\}", re.sub(r"```(?:json)?", "", text), re.DOTALL)
    try:
        return json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None


def validate(obj):
    if not isinstance(obj, dict):
        return None
    np_ = False if FORCED else obj.get("no_persona")
    if isinstance(np_, str):
        np_ = {"true": True, "false": False}.get(np_.strip().lower())
    if np_ is True:
        return {"no_persona": True, **{a: NO_PERSONA for a in LABELS}}
    if np_ is not False:
        return None
    out = {"no_persona": False}
    for a in LABELS:
        v = obj.get(a)
        if not isinstance(v, str) or v.strip().lower() not in _LOOKUP[a]:
            return None
        out[a] = _LOOKUP[a][v.strip().lower()]
    return out


def annotate(name, cfg, system, cid, text):
    raw, err = None, None
    for attempt in range(MAX_ATTEMPTS):
        try:
            raw = call_model(cfg, system,
                             f'Annotate the following YouTube comment.\n\n"{text}"')
            parsed = validate(parse_json(raw))
            if parsed is not None:
                return {"id": cid, "model": name, "raw": raw, "labels": parsed, "error": None}
            err = "invalid_output"
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
        if attempt < MAX_ATTEMPTS - 1:
            time.sleep(2 ** attempt)
    return {"id": cid, "model": name, "raw": raw, "labels": None, "error": err}


def load_ok(path):
    if not path.exists():
        return {}
    recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    return {r["id"]: r for r in recs if r["labels"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--forced", action="store_true",
                    help="control: original forced-choice prompt, per-comment, no abstention")
    args = ap.parse_args()
    global FORCED, OUT_DIR
    FORCED = args.forced
    if FORCED:
        OUT_DIR = Path("results/forced_single")

    system = build_system_prompt()
    df = pd.read_csv(GOLD_PATH, dtype={ID_COL: str})
    if args.limit:
        df = df.head(args.limit)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "system_prompt_used.txt").write_text(system, encoding="utf-8")
    (OUT_DIR / "run_config.json").write_text(json.dumps({
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "endpoint": OPENROUTER_BASE_URL, "forced": FORCED,
        "models": {m: MODELS[m] for m in args.models},
        "sampling": "provider defaults (temperature/reasoning not set)",
    }, indent=1), encoding="utf-8")

    for name in args.models:
        cfg = MODELS[name]
        out_path = OUT_DIR / f"{name}.jsonl"
        ok = load_ok(out_path)
        todo = [(r[ID_COL], r[TEXT_COL]) for _, r in df.iterrows() if r[ID_COL] not in ok]
        print(f"[{name}] {len(ok)} cached, {len(todo)} to run")
        with ThreadPoolExecutor(args.workers) as ex, open(out_path, "a", encoding="utf-8") as f:
            futs = [ex.submit(annotate, name, cfg, system, cid, txt) for cid, txt in todo]
            for i, fut in enumerate(as_completed(futs), 1):
                rec = fut.result()
                with _write_lock:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                if i % 50 == 0:
                    print(f"  {i}/{len(todo)}")
        ok = {k: v for k, v in load_ok(out_path).items() if k in set(df[ID_COL])}
        n_np = sum(r["labels"]["no_persona"] for r in ok.values())
        print(f"[{name}] valid={len(ok)}/{len(df)}  no_persona={n_np} "
              f"({n_np / max(len(ok), 1):.1%})  failures={len(df) - len(ok)}")


if __name__ == "__main__":
    main()
