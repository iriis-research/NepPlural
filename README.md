# NepPlural: A Human-Anchored Benchmark for Persona Annotation in Nepali Migration Discourse

NepPlural is a dataset and evaluation suite for modelling competing socio-cultural
personas in **Nepali** public discourse on youth migration and brain drain. It pairs
a corpus of code-mixed (Devanagari, Romanized Nepali, English) YouTube comments with
a four-axis persona taxonomy, a **human-adjudicated gold standard**, a **protocol
audit of LLM annotation**, a five-encoder baseline suite, and a human-judged
**steerability** evaluation.

Paper: *NepPlural: A Human-Anchored Benchmark for Persona Annotation in Nepali
Migration Discourse.* Proceedings of the First Workshop on Pluralistic Value
Alignment of LLMs (PlurVA-LLM), AACL-IJCNLP 2026. Paper link: *forthcoming in the
ACL Anthology* (a preprint, `NepPlural.pdf`, is included in this repository).

## Key findings

- **Human gold.** Three native Nepali speakers reach moderate agreement (mean
  Fleiss' κ = 0.55; 0.64 on whether a comment carries a persona at all). They judge
  116 of 300 substantive-looking comments (38.7%) to express no migration persona.
- **LLM annotation reliability is driven by protocol.** The same three frontier
  models (GPT-5.2, Gemini 3.5 Flash, Claude Sonnet 4.6), with the same
  instructions, reach:
  - κ ≈ 0.17 when comments are annotated **in batches through chat interfaces**
    (our original pipeline),
  - κ ≈ 0.72, above the human level, when annotated **one comment at a time**
    with a forced-choice schema,
  - κ ≈ 0.65 when annotated one at a time **with an abstention (No-Persona) option**.
- **Agreement is not validity.** Under per-comment forced choice the models agree
  with each other more than humans do, yet assign personas to every comment humans
  judge to carry none (macro-F1 against human gold ≈ 0.50). Allowing abstention
  recovers No-Persona comments (F1 0.79 to 0.87, with 5 to 16% false abstentions)
  and brings the models to the level of individual native speakers against
  leave-one-annotator-out references (macro-F1 0.75 to 0.86 vs. 0.75).
- **Encoders** (5 models x 5 seeds) trained on the batched LLM labels reach at best
  0.34 macro-F1 against human gold, far below zero-shot LLMs under the abstention
  protocol, so training-label quality is the main constraint. (Multilingual vs.
  Nepali-specific differences are confounded with pretraining scale.)
- **Steerability** (108 human-judged generations, three frontier models): single
  persona axes are adopted with 59 to 77% fidelity, full four-axis personas only
  33% of the time. No congruence gap is detected between Western-congruent and
  Nepali-collectivist personas at this scale (95% CIs rule out only gaps of about
  20 points or more).

> **Important for users of the LLM labels.** The labels in `data/annotations/`
> were produced by the original batched chat-interface protocol, which the paper
> shows to be unreliable. Do not treat them as ground truth. Use the human gold
> (`data/splits/test.csv`, `data/pools/sample/step3/gold_final.csv`) for
> evaluation, and the per-comment outputs in `results/` for LLM comparisons.

## Main results

Mean macro-F1 over the four axes on the 300 human-gold comments (Tables 4 and 5 of
the paper). **LOO** = against leave-one-annotator-out references; **LOO-P** = LOO
restricted to items where both reference annotators chose a persona.

| System | Gold | LOO | LOO-P |
|---|---|---|---|
| Human (held-out annotator) | 0.80* | 0.75 | 0.75 |
| LLM majority, batched chat | 0.39 | 0.42 | 0.67 |
| LLM majority, per-comment forced | 0.50 | 0.53 | 0.88 |
| LLM majority, per-comment abstain | 0.74 | 0.85 | 0.86 |
| XLM-RoBERTa base (5 seeds) | 0.34 | 0.36 | 0.49 |
| mBERT cased (5 seeds) | 0.34 | 0.35 | 0.49 |
| NepBERTa (5 seeds) | 0.21 | 0.20 | 0.24 |
| IRIIS RoBERTa 125M (5 seeds) | 0.20 | 0.19 | 0.24 |
| IRIIS BERT 110M (5 seeds) | 0.20 | 0.19 | 0.24 |

\* Not independent: annotators contributed to the gold. Use the LOO column for
human comparisons. Per-model, per-axis and per-script breakdowns are in
`results/protocol_audit.txt` and `results/fair_comparison.txt`; encoder per-seed and
per-class results are in `src/training/Results_v2/`.

## Taxonomy

Every comment is annotated on four axes, plus an explicit No-Persona class:

| Axis | Classes |
|---|---|
| Intent (stance) | Pro-Migration, Anti-Migration, Trapped/Regretful, Neutral/Observation |
| Primary Driver | Economic Necessity, Family Obligation, Systemic/Political Anger, Patriotism/Love |
| Value Orientation | Collectivist-Family, Collectivist-Nation, Individualist-Self |
| Affect | Despairing/Sad, Angry/Frustrated, Hopeful/Motivated, Pragmatic |
| No-Persona | The comment expresses no migration persona on any axis |

Annotation guidelines: `data/annotations/AnnotationGuidelines.pdf`.

## Dataset

| | Count |
|---|---|
| Filtered corpus | 2,148 comments |
| Substantive pool (router) | 1,894 |
| Routed No-Persona (router) | 254 |
| Human gold (3 annotators, adjudicated) | 300 |

- **Sources:** comment sections of three Nepali YouTube channels (IDS, Thaha
  Research, The Nepali Comment).
- **Router:** a deterministic, auditable keyword-and-pattern router (no LLM)
  separates obvious off-topic comments; every routing decision carries a
  machine-readable reason.
- **Human gold:** a stratified sample of 300 comments from the substantive pool,
  labelled independently by three native speakers. Gold labels are the 2-of-3
  majority per axis, with three-way ties adjudicated. Individual annotator sheets
  are released alongside the adjudicated gold.

## Repository structure

```
data/
  filtered/                 cleaned, anonymized comments
  annotations/              guidelines; original batched LLM labels; majority vote
  LLM_Judge_Verification/   20% LLM-as-a-Judge check of the batched labels + stats
                            (Claude Opus 4.8, batched; corroborating only, see Appendix E)
  pools/                    substantive_pool.csv, no_persona.csv
    sample/                 individual annotator sheets (A, B, C)
      step3/                adjudicated gold (gold_final.csv) and agreement reports
  splits/                   train / val / test (test = the 300 human-gold comments)
  steerability/             personas, generation set, fidelity sheets and results
    fidelity/               annotator sheets (A, B, C), blinding key, per-generation scores
prompts/
  annotation_prompt.md      annotation instructions used by all LLM protocols
  llm_judge_prompt.md       LLM-as-a-Judge verification prompt
  eval_prompt.md            steerability evaluation prompt
results/
  abstention_baseline/      per-comment LLM labels WITH abstention (+ exact prompt, run config)
  forced_single/            per-comment LLM labels, forced choice (+ exact prompt, run config)
  protocol_audit.txt/.json  every number reported in Sections 4 and 5 of the paper
  fair_comparison.txt       LOO and per-script tables (Appendix D)
src/
  preprocessing/            filtering, No-Persona routing, pooling, sampling
  annotation/               majority voting, review resolution
  analysis/                 agreement, gold, splits, protocol audit, figure, steerability
  validation/               20% sampling and LLM-as-a-Judge checks (notebooks)
  training/                 multi-task encoder trainer + aggregation (Results_v2/)
paper/figures/              Figure 1 (protocol audit)
```

## Reproducing the results

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**1. Corpus, pools and gold**

```bash
python src/preprocessing/prepare_pools.py
python src/preprocessing/join_llm_labels.py
python src/preprocessing/stratified_sample.py
# human annotation happens here (sheets in data/pools/sample/)
python src/analysis/score_annotations.py --sheets \
    data/pools/sample/annotation_sheet_A.csv \
    data/pools/sample/annotation_sheet_B.csv \
    data/pools/sample/annotation_sheet_C.csv
python src/analysis/finalize_gold.py
python src/analysis/build_splits.py
```

**2. LLM annotation protocols** (requires an OpenRouter API key; already-run
outputs are included in `results/`, and cached rows are skipped)

```bash
export OPENROUTER_API_KEY=...
python src/analysis/abstention_baseline.py --forced   # per-comment, forced choice
python src/analysis/abstention_baseline.py            # per-comment, with abstention
# --limit 10 for a smoke test, --models claude to run a single model
```

Models (via OpenRouter): `openai/gpt-5.2`, `google/gemini-3.5-flash`,
`anthropic/claude-sonnet-4.6`.

Each run writes the exact system prompt (`system_prompt_used.txt`) and model
settings (`run_config.json`) next to its outputs. Models are called with
provider-default sampling; exact reproduction depends on provider model versions.

**3. Protocol audit and Figure 1** (no API calls)

```bash
python src/analysis/protocol_audit.py      # agreement, validity, LOO, per-script
python src/analysis/make_fig_protocol.py   # paper/figures/protocol_audit.pdf
```

**4. Encoders** (GPU)

```bash
for m in FacebookAI/xlm-roberta-base google-bert/bert-base-multilingual-cased \
         IRIIS-RESEARCH/BERT_Nepali_110M IRIIS-RESEARCH/RoBERTa_Nepali_125M \
         Prazzwal07/nepberta; do
  python src/training/train_multitask.py --model "$m"
done
python src/training/aggregate_results.py
```

Defaults match the paper: seeds 42 to 46, learning rate 1e-5, batch size 16,
maximum length 192, 10 epochs, best-validation checkpoint. Use `--dry_run` to
validate data and label maps without loading a model. Training data are the
batched LLM majority labels (1,661 train / 184 val); see the warning above.

**5. Steerability**

```bash
python src/analysis/build_steerability_set.py
# generations are collected manually (see data/steerability/generation_instructions.md)
python src/analysis/build_fidelity_sheets.py
# human fidelity annotation happens here
python src/analysis/score_steerability.py
```

Steerability generations used Claude Opus 4.6, GPT 5.6 Luna and Gemini 3.6 Flash at
temperature 1.0 (different model versions from the annotation study). The six
personas and six questions are in `data/steerability/personas.json` and Appendix G
of the paper.

## Evaluation notes

- **Human reference.** Annotators contributed to the gold, so scoring them against
  it overstates human performance (0.80). The paper's human reference is the
  **leave-one-annotator-out** score (0.75): each annotator is scored against the
  labels on which the other two agree, and every system is scored on the same items.
- **Macro-F1** is computed over the classes present in the reference;
  majority-vote ties count as errors.
- **Small groups.** Several classes have fewer than 20 gold examples, and the
  Devanagari (21) and code-mixed (11) script groups are small; treat per-class and
  per-script scores as indicative only.
- **Model drift.** Proprietary models change over time. The released per-comment
  outputs in `results/` are the fixed artifacts; re-running the API scripts may
  give different labels.

## Data statement and ethics

The full data statement (Bender and Friedman, 2018 schema) is in
[`DATA_STATEMENT.md`](DATA_STATEMENT.md). In brief:

The dataset is derived from public YouTube comments on a politically sensitive
topic. Author identifiers, timestamps, and platform comment IDs are removed, and
user mentions are replaced with a placeholder; only anonymized comment text is
released. Verbatim public comments can in principle be located by search; please
do not attempt to re-identify authors. Labels describe the persona a comment
expresses, not the truth of any claim, and must not be used to profile, target, or
make decisions about individuals or communities. The dataset is intended for
research on pluralistic alignment, low-resource NLP, LLM annotation methodology,
and cultural representation in language models. It is not suitable for deployment
decisions about individuals.

Comment text was sent to commercial model providers for annotation under their API
and interface terms. The three annotators are native Nepali speakers who annotated
as part of the research effort; guidelines are released in `data/annotations/`.

## License

Code and annotations are released under CC BY 4.0 (see `LICENSE`). Comment text
originates from public YouTube comments and remains subject to the rights of its
authors and the platform's terms.

## Citation

```bibtex
@inproceedings{nepplural2026,
  title     = {{NepPlural}: A Human-Anchored Benchmark for Persona Annotation in {N}epali Migration Discourse},
  author    = {Nyachhyon, Jinu and Khatiwada, Anish and Timilsina, Bimal and Pandey, Rakhee and Acharya, Ashish},
  booktitle = {Proceedings of the First Workshop on Pluralistic Value Alignment of LLMs (PlurVA-LLM)},
  year      = {2026},
  publisher = {Association for Computational Linguistics}
}
```

## Contact

Please open a GitHub issue for questions, bug reports, or label corrections. For
other enquiries, contact the corresponding author listed in the paper.
