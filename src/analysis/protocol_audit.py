"""
protocol_audit.py: single source of every number in the camera-ready protocol audit.

Replaces analyze_abstention.py and fair_comparison.py. Fixes from code review:
  * Macro-F1 = sklearn f1_score(labels=<classes present in the reference>,
    average="macro", zero_division=0). On the full 300 this is identical to the
    definition behind the encoder (0.34) and human-ceiling figures. On subsets
    without No-Persona references (gold-persona, persona-only LOO), a false
    abstention or a majority-vote tie counts as an error on the true class instead
    of adding a phantom zero-F1 class. False-abstention rates are reported separately.
  * No hardcoded reference numbers: the original batched condition and the humans
    are recomputed here from the repo files.
  * Leave-one-annotator-out (LOO) everywhere humans are compared, including the
    per-script breakdown, plus a persona-only LOO variant.
  * Any missing input is a hard error (nothing is silently skipped).

Conditions:
  batched-chat  original labels: forced choice, numbered batches of ~20-50, CSV,
                collected in the chat interfaces (data/annotations/label/)
  forced        per-comment, API via OpenRouter, original forced-choice prompt, JSON
  abstain       per-comment, API via OpenRouter, same prompt + No-Persona option, JSON
  forced vs abstain isolates abstention; batched-chat vs forced is the
  format/access difference (batching, chat UI vs API, CSV vs JSON, defaults).

Usage (repo root):  python src/analysis/protocol_audit.py
Writes:             results/protocol_audit.txt, results/protocol_audit.json
"""
import glob
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

AXES = ["intent", "primary_driver", "value_orientation", "affect"]
NP, TIE = "No-Persona", "__tie__"
GOLD = "data/pools/sample/step3/gold_final.csv"
SHEETS = {x: f"data/pools/sample/annotation_sheet_{x}.csv" for x in "ABC"}
MODELS = {"gpt": "GPT-5.2", "gemini": "Gemini 3.5 flash", "claude": "Sonet 4.6"}
COND_DIRS = {"forced": "results/forced_single", "abstain": "results/abstention_baseline"}
ENCODERS = {"XLM-R": "FacebookAI__xlm-roberta-base",
            "mBERT": "google-bert__bert-base-multilingual-cased"}
SCRIPTS = ["latin", "roman_nepali", "devanagari", "code_mixed"]
OUT_TXT, OUT_JSON = Path("results/protocol_audit.txt"), Path("results/protocol_audit.json")

lines, numbers = [], {}


def out(s=""):
    print(s)
    lines.append(s)


def cid(s):
    return hashlib.sha1(str(s).strip().lower().encode("utf-8")).hexdigest()[:12]


# ------------------------------------------------------------------ metrics
def mf1(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=object).astype(str)
    y_pred = np.asarray(y_pred, dtype=object).astype(str)
    labels = sorted(set(y_true))
    return float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))


def fleiss(mat):
    mat = np.asarray(mat, dtype=object)
    cats = sorted(set(mat.ravel()))
    idx = {c: i for i, c in enumerate(cats)}
    n = mat.shape[1]
    C = np.zeros((mat.shape[0], len(cats)))
    for r, row in enumerate(mat):
        for v in row:
            C[r, idx[v]] += 1
    pj = C.sum(0) / C.sum()
    Pi = ((C ** 2).sum(1) - n) / (n * (n - 1))
    Pe = (pj ** 2).sum()
    return float((Pi.mean() - Pe) / (1 - Pe)) if Pe < 1 else float("nan")


def cohen(a, b):
    a, b = np.asarray(a), np.asarray(b)
    po = (a == b).mean()
    pe = sum((a == c).mean() * (b == c).mean() for c in set(a) | set(b))
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def prf(pred, ref):
    pred, ref = np.asarray(pred, bool), np.asarray(ref, bool)
    tp = (pred & ref).sum()
    p = tp / pred.sum() if pred.sum() else 0.0
    r = tp / ref.sum() if ref.sum() else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def majority(frames, a):
    res = []
    for row in zip(*[f[a].values for f in frames]):
        row = list(row)
        res.append(next((v for v in row if row.count(v) >= 2), TIE))
    return np.array(res, dtype=object)


# ------------------------------------------------------------------ load
gold = pd.read_csv(GOLD, dtype={"comment_id": str}).set_index("comment_id")
ids = gold.index
gold_np = (gold["intent"] == NP).values
gp = ~gold_np  # gold-persona rows

humans = {}
for x, f in SHEETS.items():
    s = pd.read_csv(f, dtype={"comment_id": str}).set_index("comment_id")
    assert set(ids) <= set(s.index), f"sheet {x} missing ids"
    s = s.loc[ids]
    npf = s["is_no_persona"].astype(str).str.strip().isin(["1", "1.0", "True", "true"])
    humans[x] = pd.DataFrame({a: np.where(npf, NP, s[a].astype(str).str.strip()) for a in AXES},
                             index=ids)
script = pd.read_csv(SHEETS["A"], dtype={"comment_id": str}).set_index("comment_id").loc[ids, "script"]

cond = {"batched-chat": {}, "forced": {}, "abstain": {}}
for short, folder in MODELS.items():
    files = glob.glob(f"data/annotations/label/{folder}/**/*.csv", recursive=True)
    assert files, f"no original labels for {folder}"
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    d.columns = [c.strip().lower() for c in d.columns]
    d["comment_id"] = d["comment"].apply(cid)
    d = d.drop_duplicates("comment_id").set_index("comment_id")
    assert set(ids) <= set(d.index), f"original {folder} missing ids"
    cond["batched-chat"][short] = d.loc[ids, AXES].apply(lambda c: c.astype(str).str.strip())
    for cname, cdir in COND_DIRS.items():
        p = Path(cdir) / f"{short}.jsonl"
        assert p.exists(), f"missing {p}"
        ok = {}
        for l in open(p, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                if r["labels"]:
                    ok[r["id"]] = r["labels"]
        assert set(ids) <= set(ok), f"{p}: {len(set(ids) - set(ok))} ids without valid labels"
        cond[cname][short] = pd.DataFrame.from_dict(ok, orient="index").loc[ids, AXES]

systems = {}
for cname, per in cond.items():
    for short, df in per.items():
        systems[f"{cname} | {short}"] = df
    systems[f"{cname} | majority"] = pd.DataFrame(
        {a: majority(list(per.values()), a) for a in AXES}, index=ids)
for name, folder in ENCODERS.items():
    seeds = []
    for f in sorted(glob.glob(f"src/training/Results_v2/{folder}/seed*_test_predictions.csv")):
        d = pd.read_csv(f)
        d["comment_id"] = d["comment"].apply(cid)
        d = d.drop_duplicates("comment_id").set_index("comment_id")
        assert set(ids) <= set(d.index), f"{f} missing ids"
        seeds.append(d.loc[ids, [f"{a}_pred" for a in AXES]].set_axis(AXES, axis=1).astype(str))
    assert len(seeds) == 5, f"{name}: expected 5 seeds, found {len(seeds)}"
    systems[f"encoder | {name}"] = seeds


def score(sysobj, a, mask, ref):
    if isinstance(sysobj, list):
        return float(np.mean([mf1(ref, s[a].values[mask]) for s in sysobj]))
    return mf1(ref, sysobj[a].values[mask])


def row(name, vals, w=34):
    return f"{name:{w}}" + "".join(f"{v:10.2f}" for v in vals) + f"{np.mean(vals):9.2f}"


HDR = lambda w=34: f"{'':{w}}" + "".join(f"{a[:9]:>10}" for a in AXES) + f"{'mean':>9}"

# ------------------------------------------------------------------ 1. agreement
out("1. Fleiss' kappa among three raters (per axis, 5-way incl. No-Persona)")
out(f"   all = 300 comments; gold-persona = {gp.sum()} comments humans judged to carry a persona")
out(f"{'raters':34}" + "".join(f"{a[:9]:>10}" for a in AXES) + f"{'mean':>9}")
numbers["kappa"] = {}
groups = {"humans": [humans[x] for x in "ABC"],
          **{c: list(cond[c].values()) for c in cond}}
for subset, mask in [("all", np.ones(len(ids), bool)), ("gold-persona", gp)]:
    for g, frames in groups.items():
        ks = [fleiss(np.column_stack([f[a].values[mask] for f in frames])) for a in AXES]
        numbers["kappa"][f"{g} | {subset}"] = ks
        out(row(f"{g} ({subset})", ks))
out()
np_h = np.column_stack([(humans[x]["intent"] == NP).values for x in "ABC"])
np_a = np.column_stack([(cond["abstain"][m]["intent"] == NP).values for m in MODELS])
numbers["kappa_np_decision"] = {"humans": fleiss(np_h), "abstain": fleiss(np_a)}
out(f"   No-Persona decision kappa: humans {fleiss(np_h):.2f} | abstain LLMs {fleiss(np_a):.2f}")
out()

# ------------------------------------------------------------------ 2. NP vs gold
out("2. No-Persona decision vs adjudicated gold (abstain condition)")
out(f"{'model':12}{'abstain%':>10}{'P':>7}{'R':>7}{'F1':>7}{'kappa':>8}")
numbers["np_vs_gold"] = {}
for m in list(MODELS) + ["majority"]:
    pred = (np_a.sum(1) >= 2) if m == "majority" else (cond["abstain"][m]["intent"] == NP).values
    p, r, f = prf(pred, gold_np)
    k = cohen(pred, gold_np)
    numbers["np_vs_gold"][m] = {"rate": float(pred.mean()), "P": p, "R": r, "F1": f, "kappa": k}
    out(f"{m:12}{pred.mean():10.1%}{p:7.2f}{r:7.2f}{f:7.2f}{k:8.2f}")
out()

fa = {m: float((cond["abstain"][m]["intent"] == NP).values[gp].mean()) for m in MODELS}
fa["majority"] = float((np_a.sum(1) >= 2)[gp].mean())
numbers["false_abstention_on_gold_persona"] = fa
out(f"   false abstention on the {gp.sum()} gold-persona comments: "
    + ", ".join(f"{m} {v:.1%}" for m, v in fa.items()))
out()

# ------------------------------------------------------------------ 3. vs adjudicated gold
out("3. Macro-F1 vs adjudicated gold (NOTE: human rows are NOT independent of gold)")
numbers["f1_vs_gold"] = {}
for subset, mask in [("all 300", np.ones(len(ids), bool)), (f"gold-persona {gp.sum()}", gp)]:
    out(f"   [{subset}]")
    out(HDR())
    hv = [np.mean([mf1(gold[a].values[mask], humans[x][a].values[mask]) for x in "ABC"]) for a in AXES]
    out(row("human (mean A,B,C; non-indep.)", hv))
    numbers["f1_vs_gold"][f"human | {subset}"] = hv
    for n, s in systems.items():
        v = [score(s, a, mask, gold[a].values[mask]) for a in AXES]
        numbers["f1_vs_gold"][f"{n} | {subset}"] = v
        out(row(n, v))
    out()

# ------------------------------------------------------------------ 4. LOO
ROT = {"A": ("B", "C"), "B": ("A", "C"), "C": ("A", "B")}


def loo(extra_mask=None, persona_only=False):
    res = {"human (held-out)": {a: [] for a in AXES}}
    res.update({n: {a: [] for a in AXES} for n in systems})
    nitems = {a: [] for a in AXES}
    for x, (y, z) in ROT.items():
        for a in AXES:
            mask = (humans[y][a] == humans[z][a]).to_numpy().copy()
            if persona_only:
                mask &= (humans[y][a] != NP).values
            if extra_mask is not None:
                mask &= extra_mask
            ref = humans[y][a].values[mask]
            nitems[a].append(int(mask.sum()))
            res["human (held-out)"][a].append(mf1(ref, humans[x][a].values[mask]))
            for n, s in systems.items():
                res[n][a].append(score(s, a, mask, ref))
    return {n: [float(np.mean(d[a])) for a in AXES] for n, d in res.items()}, nitems


for title, po in [("4a. LOO macro-F1 (reference = agreement of the other two annotators)", False),
                  ("4b. LOO macro-F1, persona-only references (both reference annotators chose a persona)", True)]:
    res, nitems = loo(persona_only=po)
    numbers[f"loo{'_persona' if po else ''}"] = res
    out(title)
    out("   items per rotation (mean): " + ", ".join(f"{a}={np.mean(nitems[a]):.0f}" for a in AXES))
    out(HDR())
    for n, v in res.items():
        out(row(n, v))
    out()

# ------------------------------------------------------------------ 5. per-script LOO
out("5. Per-script LOO mean macro-F1 (4 axes); indicative only for small groups")
sel = ["human (held-out)", "batched-chat | majority", "forced | majority",
       "abstain | majority", "encoder | XLM-R", "encoder | mBERT"]
out(f"{'':34}" + "".join(f"{g[:12]:>14}" for g in SCRIPTS))
out(f"{'n comments':34}" + "".join(f"{int((script == g).sum()):14d}" for g in SCRIPTS))
per_script = {}
for g in SCRIPTS:
    res, nitems = loo(extra_mask=(script == g).values)
    per_script[g] = {n: float(np.mean(res[n])) for n in sel}
    per_script[g]["_mean_items"] = float(np.mean([np.mean(nitems[a]) for a in AXES]))
out(f"{'LOO items (mean)':34}" + "".join(f"{per_script[g]['_mean_items']:14.0f}" for g in SCRIPTS))
for n in sel:
    out(f"{n:34}" + "".join(f"{per_script[g][n]:14.2f}" for g in SCRIPTS))
numbers["per_script_loo"] = per_script

OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
OUT_JSON.write_text(json.dumps(numbers, indent=1, default=float), encoding="utf-8")
print(f"\nSaved {OUT_TXT} and {OUT_JSON}")
