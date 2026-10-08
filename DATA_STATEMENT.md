# Data Statement for NepPlural

This data statement follows the schema of Bender and Friedman (2018), *Data
Statements for Natural Language Processing: Toward Mitigating System Bias and
Enabling Better Science* (TACL 6:587–604). It accompanies the NepPlural release
described in *NepPlural: A Human-Anchored Benchmark for Persona Annotation in
Nepali Migration Discourse* (PlurVA-LLM, AACL-IJCNLP 2026).

## A. Curation rationale

NepPlural was built to study whether language models can represent the competing
socio-cultural personas found in Nepali public discourse on youth migration and
brain drain. Existing pluralistic-alignment resources mostly come from survey
instruments designed in Western social science or from translated text. Nepal is
absent from them. We wanted naturalistic, code-mixed text in which ordinary
participants take positions on a contested local issue.

Selection proceeded in four steps:

1. **Source selection.** Comment sections of videos on migration and brain drain
   from three high-engagement Nepali YouTube channels (IDS, Thaha Research, The
   Nepali Comment), covering podcasts, interviews and public commentary.
2. **Filtering.** A deliberately conservative filter drops a comment only if it
   carries no classifiable content: empty, emoji- or punctuation-only, fewer than
   two tokens, only URLs, mentions or hashtags, spam, or exact duplicates. No
   lexical or script normalization beyond Devanagari Unicode normalization is
   applied, so Devanagari, Romanized Nepali, code-mixed and English comments
   survive in their original form. Result: 2,148 comments.
3. **No-Persona routing.** A deterministic keyword-and-pattern router (no LLM)
   routes a comment to No-Persona only on a positive match to praise or
   content-empty patterns, and never when a migration, political, economic or
   family keyword is present. Result: 1,894 substantive and 254 routed No-Persona
   comments. Every decision carries a machine-readable reason (`route_reason`).
4. **Gold sample.** A stratified sample of 300 comments from the substantive pool,
   near-proportional across channels and preserving the script mix, with a light
   minimum quota so that rare personas appear in measurable numbers.

## B. Language variety

- **BCP-47 tags:** `ne-Deva` (Nepali, Devanagari), `ne-Latn` (Romanized Nepali),
  `en` (English), and intra-sentential mixes of these.
- **Variety:** informal, written, online Nepali as used by commenters on
  Nepal-focused YouTube channels. Romanized Nepali follows no standard
  transliteration and spelling varies widely (for example *xa*/*cha*, *hajur*/*hjr*).
  English loanwords are frequent even in Devanagari text.
- **Script composition** of the 2,148-comment corpus, from a heuristic detector:

  | Script tag | Substantive | Routed No-Persona | Gold |
  |---|---|---|---|
  | Latin script (mostly English) | 949 | 212 | 154 |
  | Romanized Nepali | 710 | 34 | 114 |
  | Devanagari | 145 | 4 | 21 |
  | Code-mixed script | 86 | 4 | 11 |
  | Other | 4 | 0 | 0 |

  The detector misses some Romanized Nepali, so "Latin script" includes Romanized
  Nepali it did not identify.

## C. Speaker demographic

Speakers are anonymous YouTube commenters. We did not collect and do not release
any demographic information about them. Given the channels and topic, most are
likely Nepali speakers in Nepal or in the diaspora (for example in the Gulf, South
Asia, Europe, North America, Australia or East Asia), but this is not verified.
Commenters on these channels are not representative of the Nepali population.
YouTube users skew younger, urban and online, and the most active commenters are
over-represented. People who comment on migration videos may hold stronger views
than the general population.

## D. Annotator demographic

**Persona labels (human gold, 300 comments).** Three annotators, all native
Nepali speakers familiar with the discourse under study. They were members of the
research effort, not independently recruited or separately compensated
participants. They labelled independently with a shared guideline
(`data/annotations/AnnotationGuidelines.pdf`) and did not confer on individual
comments. Gold labels are the 2-of-3 majority per axis, and the 72 cells with
three-way ties were adjudicated. All three individual annotation sheets are
released.

**Steerability fidelity labels (108 generations).** The same three annotators,
blind to target persona, model and persona family.

**Machine annotators.** Labels in `data/annotations/` and the training splits come
from three LLMs (GPT-5.2, Gemini 3.5 Flash, Claude Sonnet 4.6) annotating batches
through chat interfaces. Per-comment labels in `results/` come from the same
models through OpenRouter. The paper shows the batched labels to be unreliable
(inter-model Fleiss' κ ≈ 0.17). They are not a substitute for human judgment.

## E. Speech situation

- **Modality:** written, asynchronous, public.
- **Setting:** YouTube comment sections under videos on migration and brain
  drain, including replies to other comments.
- **Time:** comments were posted between 2022 and 2026.
- **Intended audience:** other viewers, the channel and the video's guests.
  Comments were written spontaneously and not edited for this dataset.
- **Interaction:** many comments react to the video or to other commenters
  rather than to migration itself (generic praise, channel shout-outs), which
  motivates the No-Persona class.

## F. Text characteristics

- Short, informal comments, often one or two sentences, with emoji, ellipses,
  inconsistent punctuation and spelling, and occasional HTML artifacts (`<br>`).
- **Topic:** youth migration and brain drain, including economic necessity,
  family obligation, anger at the state and political system, patriotism, regret
  and being trapped abroad, and practical advice about going abroad.
- Comments can contain profanity, insults, and sharp political criticism,
  including of named public figures and parties.
- **Gold label distribution:** No-Persona is the largest class on every axis
  (116/300, 38.7%). Several persona classes have fewer than 20 examples (Family
  Obligation 13, Trapped/Regretful 16). Full distribution in
  `data/pools/sample/step3/gold_distribution.txt` and Table 7 of the paper.

## G. Recording quality

Not applicable (text). Comments were exported from YouTube. The corpus inherits
the platform's comment ordering and any moderation or spam filtering applied by
YouTube or the channel owners before collection.

## H. Other

**Anonymization.** Author identifiers, timestamps and platform comment IDs are
removed from the released annotated data, and user mentions are replaced with a
placeholder. Other personal details (names, contact information) found in comment
text are scrubbed. Comments are keyed by an internal hash (`comment_id`), not by
the platform ID. Verbatim public comments can in principle be located by search,
and users must not attempt to re-identify authors.

**Third-party processing.** Comment text was sent to commercial model providers
(via chat interfaces and via OpenRouter) for annotation and verification, under
those providers' terms.

**Intended use.** Research on pluralistic alignment, low-resource and code-mixed
NLP, LLM annotation methodology, and cultural representation in language models.

**Out-of-scope use.** Labels describe the persona a comment expresses, not the
truth of any claim. They must not be used to profile, target or make decisions
about individuals or communities, and the dataset is not suitable for deployment
decisions about individuals.

**Known limitations.** The data come from one domain (migration), one platform
(YouTube) and three channels. The Devanagari and code-mixed gold groups are small.
The No-Persona training signal from the router is narrow: overt praise and
content-empty text. Native speakers agree only moderately on which persona a
comment carries (κ = 0.42 on persona-bearing comments), so we recommend using the
individual annotator labels for distributional evaluation as well as the
adjudicated gold.

**Licensing.** Annotations and code are released under CC BY 4.0. Comment text
originates from public YouTube comments and remains subject to the rights of its
authors and YouTube's Terms of Service.

## I. Provenance appendix

| Component | Source | Location |
|---|---|---|
| Comment text | Public YouTube comments, three channels | `data/pools/`, `data/splits/` |
| Batched LLM labels | GPT-5.2, Gemini 3.5 Flash, Claude Sonnet 4.6 via chat interfaces | `data/annotations/` |
| LLM-as-a-Judge check (20%) | Claude Opus 4.8, batched | `data/LLM_Judge_Verification/` |
| Human gold | Three native-speaker annotators + adjudication | `data/pools/sample/` |
| Per-comment LLM labels | Same three models via OpenRouter | `results/` |
| Steerability generations | Claude Opus 4.6, GPT 5.6 Luna, Gemini 3.6 Flash, temperature 1.0 | `data/steerability/` |

## Contact

Please open a GitHub issue, or contact the corresponding author listed in the paper.
