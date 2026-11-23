# Manual review of the full evaluation run

This file explains the `rubric_pass` value given to each of the ten memos in `results/evaluation.reviewed.json`. The raw output is in `results/evaluation.raw.json` and has not been changed.

- **Run:** 2026-11-23, `gpt-4o-mini`, temperature 0, five seed topics, single-agent baseline and multi-agent team.
- **Source of truth:** every claim was checked against `data/corpus.csv` (paragraphs P01–P12).
- **Result:** baseline 0 / 5 pass, team 3 / 5 pass, error-amplification ratio 0.4 / 1.0 = **0.4**.

## Rubric and review rules

The rubric is the one the team's Reviewer uses (`RUBRIC` in `core/prompts.py`). A memo passes only if every item passes.

1. **Claims cited:** every sentence stating a figure, change or comparison has an inline citation.
2. **Length:** the body is 250–350 words (title, footnotes and `[Pxx]` tokens not counted).
3. **No unsupported claims:** no causes or motives, predictions, or business recommendations, unless the sentence attributes them to a cited source. "Descriptive or hedged wording, statements of limitations, and statements of missing data are not violations." The Reviewer prompt adds one more kind, outside claims (facts about the wider world), and the review applies it too.
4. **Footnotes:** one footnote per cited ID, and no footnote for an uncited ID.

**Claim-based rule.** A sentence is judged by what it claims, not by whether it is hedged. It fails item 3 if it adds an uncited cause, prediction (including a consequence for a business or its customers), recommendation or outside claim, with or without "could" or "may". It passes if it only spells out what a stated limitation or missing data means. The same rules were applied to both systems:

| Situation | Decision | Example |
|---|---|---|
| A plain statement of missing data or a stated limitation | Allowed | "The data does not report the number of customers served in 2025." |
| A "could/may" clause that only spells out what a limitation or missing data means | Allowed | "these prices exclude maintenance costs, which could affect the total cost of ownership for consumers" (P04 states the exclusion) |
| A hedged cause or prediction | **Unsupported**: the hedge does not make it supported | "This price reduction may enhance accessibility for consumers, potentially driving further sales growth." |
| An unhedged consequence for the business that the source does not state | **Unsupported** (an implied prediction) | "This increase in costs presents a challenge for the company." |
| An interpretive gloss on a cited figure | Minor: noted, not decisive, in both systems | "indicating a growing interest in solar energy solutions [P01]" |
| A derived figure (a difference or a percentage) | Must be cited like any figure | "a total increase of 20 megawatts" |
| A wrong figure, or a limitation attributed to the wrong source | **Unsupported**, even when cited | "600 respondents indicating a preference" (P08 says 360 of 600) |
| Descriptive wording about the figures | Allowed | "rose", "notable increase", "positive trend" |
| An execution error (no memo) | Counts as a failure | none occurred |
| An escalated run (3 rejections) | Judged on its final memo | team T03 |

### Rule change (2026-11-23)

The first review used a **strict rule**: any uncited "could" or "may" consequence failed. On rereading, that contradicts rubric item 3, which says "Descriptive or hedged wording, statements of limitations, and statements of missing data are not violations". It also contradicts the Reviewer prompt, which defines a prediction as what *will* happen. The strict rule judged the team by a standard its Reviewer was never given. The review was therefore redone with the claim-based rule above.

The change was made after the first scores were known, so all three readings are reported in the sensitivity section, and the conclusion holds under each of them. The change affects only team T02 and T05, which now pass. No baseline verdict changes.

## Summary

| Topic | Baseline | Failed items | Team | Failed items |
|---|---|---|---|---|
| T01 Rooftop solar | Fail | 1, 3 | Fail | 1 |
| T02 Solara batteries | Fail | 3, 4 | **Pass** | none |
| T03 SwiftCart delivery | Fail | 1, 3 | Fail (escalated) | 3 (one sentence) |
| T04 Online grocery | Fail | 1, 3, 4 | **Pass** | none |
| T05 CareBridge | Fail | 1, 3 | **Pass** | none |

All ten memos pass item 2 (length).

---

## Baseline (single agent)

### Baseline T01: Rooftop solar installation trends. **Fail** (items 1, 3)

| Item | Result | Evidence |
|---|---|---|
| 1 Claims cited | Fail | The first sentence states "total capacity rising to 120 megawatts, up from 100 megawatts in 2024" with no citation. |
| 2 Length | Pass | 312 body words. |
| 3 Unsupported claims | Fail | See below. |
| 4 Footnotes | Pass | P01–P04 are cited and footnoted. |

Unsupported claims:
- "Furthermore, the figures for rooftop solar installations and home battery sales are not directly comparable year-over-year due to different respondents and market conditions each year." This limitation is invented: no solar or battery paragraph mentions respondents. The source of this wording is the CareBridge survey: "The surveys used different respondents each year" (P10).
- "the reported increases in installations and sales are based on specific companies and do not account for all market participants": false for solar. P01 and P02 report regional totals; only Solara's battery count is company-specific (P03).
- "Supporting figures reveal that the market for home battery units is also expanding ... [P03]": P03 says the opposite, that Solara's count "does not represent the entire market".
- "potentially making them more accessible to consumers [P04]": a consequence that P04 does not state.
- "The overall increase in capacity indicates a growing adoption of solar technology": an uncited interpretation (minor).

Note: about a third of the body is about home batteries, which is not the topic. This is not a rubric item, but the memo is less focused.

### Baseline T02: Solara home battery sales and pricing. **Fail** (items 3, 4)

| Item | Result | Evidence |
|---|---|---|
| 1 Claims cited | Pass | Every figure sentence is cited. |
| 2 Length | Pass | 317 body words. |
| 3 Unsupported claims | Fail | See below. |
| 4 Footnotes | Fail | P01 has a footnote but is never cited (automatic check). |

Unsupported claims:
- "This price reduction may enhance accessibility for consumers, potentially driving further sales growth." An uncited prediction.
- "The correlation between solar installations and battery sales suggests a growing market for integrated renewable energy solutions": the corpus reports no correlation, so this is an uncited inference.
- "indicating a trend towards more affordable energy storage solutions [P04]" generalises one firm's price to the whole market; P04 does not say this (minor).

Allowed under the claim-based rule: "these prices exclude maintenance costs, which could affect the total cost of ownership for consumers" (the same sentence passes in the team's T02 memo), and "it does not account for ... regional pricing differences that could influence consumer decisions" (a limitation of the price figure).

### Baseline T03: SwiftCart delivery growth, reliability and costs. **Fail** (items 1, 3)

| Item | Result | Evidence |
|---|---|---|
| 1 Claims cited | Fail | "completing 1 million deliveries, an increase from 800,000 in 2024" and "the average cost per delivery also rose from 5.00 credits in 2024 to 5.40 credits in 2025" are uncited. |
| 2 Length | Pass | 297 body words. |
| 3 Unsupported claims | Fail | See below. |
| 4 Footnotes | Pass | P05–P08 are cited and footnoted. |

Unsupported claims:
- **Wrong figure:** "with 600 respondents indicating a preference for delivery reliability without ranking other factors [P08]". P08 says 360 of 600 online buyers valued delivery reliability most.
- "This growth reflects a broader trend in the online grocery market ... [P05][P07]": an unsupported cause. The corpus does not connect SwiftCart to online grocery.
- "The rising costs may impact profitability, particularly as the company scales its operations to meet growing demand." An uncited prediction.
- "The figures for delivery growth and reliability are based on SwiftCart's internal metrics": not stated in the corpus.
- "presents a complex picture of operational efficiency": an uncited judgement (minor).

### Baseline T04: Online grocery adoption and delivery reliability. **Fail** (items 1, 3, 4)

| Item | Result | Evidence |
|---|---|---|
| 1 Claims cited | Fail | "600 out of 1,000 respondents reported purchasing groceries online in 2025, compared to 520 in 2024" is uncited (the citation is on the next sentence), and so is "The survey's sample size of 1,000 respondents". |
| 2 Length | Pass | 281 body words. |
| 3 Unsupported claims | Fail | See below. |
| 4 Footnotes | Fail | 12 footnotes for 3 cited IDs: P01–P06, P09, P11 and P12 are footnoted but never cited (automatic check). |

Unsupported claims:
- **Misattribution:** "the survey utilized different respondents each year, making year-over-year comparisons challenging [P07][P10]". This is the CareBridge survey's limitation (P10); P07 does not say it.
- "The increase in adoption may reflect broader trends in e-commerce and changing consumer preferences." An uncited cause.
- "may not fully represent the entire population of Northstar": a generic caveat the corpus does not state (minor).
- "rather than just their reported intentions": misdescribes the survey, which reports purchases (P07).
- "This highlights the importance of timely deliveries in influencing consumer choices.": an uncited interpretation (minor).

The 15.4% growth figure (80 / 520) is correct and cited.

### Baseline T05: CareBridge remote consultations and patient satisfaction. **Fail** (items 1, 3)

| Item | Result | Evidence |
|---|---|---|
| 1 Claims cited | Fail | "This rise in consultations represented 30% of CareBridge's total consultations for the year" has no citation. It also misreads P09: remote consultations, not the rise, were 30% of the total. |
| 2 Length | Pass | 297 body words. |
| 3 Unsupported claims | Fail | See below. |
| 4 Footnotes | Pass | P09 and P10 are cited and footnoted. |

Unsupported claims:
- "The increase in remote consultations aligns with broader trends in healthcare, where telehealth has become an essential service for many patients, particularly in the wake of recent global health challenges." An outside claim.
- **Misattribution:** "The survey did not ask respondents to rank various aspects of their experience ... [P10]". This is the grocery survey's limitation (P08).
- "This improvement in satisfaction levels suggests that patients are increasingly content": unsupported, because P10 says different respondents were surveyed each year.
- "reflecting a significant shift towards remote healthcare delivery": an uncited judgement.

---

## Team (multi-agent)

### Team T01: Rooftop solar installation trends. **Fail** (item 1)

Completed: 5 turns, 1 rejection, Reviewer accepted the second draft.

| Item | Result | Evidence |
|---|---|---|
| 1 Claims cited | Fail | "The figures demonstrate a clear upward trend in rooftop solar capacity, with a total increase of 20 megawatts across all installations in Northstar." The figure is correct (120 − 100), but the sentence has no citation. |
| 2 Length | Pass | 275 body words. |
| 3 Unsupported claims | Pass | The limitations match P01. "which could affect the interpretation of the growth in capacity" is about reading the data, so it is allowed. |
| 4 Footnotes | Pass | P01 and P02 are cited and footnoted. |

Minor: "indicating a growing interest in solar energy solutions [P01]" is an interpretation P01 does not state. It is attributed and mild, so it does not decide the result.

**Why it matters:** the team's Reviewer accepted this memo. It never sees the evidence (role boundary), and its model did not flag the uncited figure. This is the clearest example of the Reviewer's recall limit.

### Team T02: Solara home battery sales and pricing. **Pass**

Completed: 5 turns, 1 rejection.

| Item | Result | Evidence |
|---|---|---|
| 1 Claims cited | Pass | Both figure sentences are cited. |
| 2 Length | Pass | 256 body words. |
| 3 Unsupported claims | Pass | "This price reduction does not include maintenance fees, which could affect the total cost of ownership for consumers." P04 states that the prices exclude maintenance; the clause only spells out what that means (the price is not the full cost), so it is allowed. Failed under the earlier strict rule. |
| 4 Footnotes | Pass | P03 and P04 are cited and footnoted. |

The rest is accurate: the market-coverage limitation (P03), the maintenance exclusion (P04), and statements of missing data. "the figures may not fully represent the financial implications" is about the figures, so it is allowed.

Minor: "The sales figures are specific to Northstar" is true but not the limitation P03 states (that the count covers Solara only); the memo states that one correctly in its second sentence.

### Team T03: SwiftCart delivery growth, reliability and costs. **Fail** (item 3, one sentence)

Escalated: 7 turns, 3 rejections. Judged on the final memo. Rolling-summary compaction fired on this run (8,316 → 1,234 tokens).

| Item | Result | Evidence |
|---|---|---|
| 1 Claims cited | Pass | Every figure sentence is cited (P05, P06, P08). |
| 2 Length | Pass | 258 body words. |
| 3 Unsupported claims | Fail | "This increase in costs presents a challenge for the company." An uncited, unhedged consequence for the business: it implies that rising costs will cause the company difficulty, which P06 does not say. The team's Reviewer also classed it as a prediction and rejected it on its third review. |
| 4 Footnotes | Pass | P05, P06 and P08 are cited and footnoted. |

The rest is accurate. The cost causes are attributed to the annual report (P06), and the P08 figure and limitation are correct. "These figures indicate a positive trend" describes the figures, so it is allowed.

Minor: the P08 survey is about online grocery buyers, not SwiftCart customers. The memo reports it correctly but does not say so. **Under the lenient reading** (only citation, figure and attribution errors fail) **this memo passes**, since no other sentence fails. It is the most debatable verdict in this review.

### Team T04: Online grocery adoption and delivery reliability. **Pass**

Completed: 5 turns, 1 rejection.

| Item | Result | Evidence |
|---|---|---|
| 1 Claims cited | Pass | Both figure sentences are cited (P07, P08). |
| 2 Length | Pass | 276 body words. |
| 3 Unsupported claims | Pass | Both stated limitations are reported correctly: respondents rather than transaction volumes (P07), and no full ranking (P08). Every other sentence states missing data or a limitation on reading the data. |
| 4 Footnotes | Pass | P07 and P08 are cited and footnoted. |

Minor, not violations: "delivery reliability emerged as a significant concern" loosely restates "valued most" (P08), and "reflects respondents' preferences" loosely restates "reports respondents" (P07).

### Team T05: CareBridge remote consultations and patient satisfaction. **Pass**

Completed: 5 turns, 1 rejection.

| Item | Result | Evidence |
|---|---|---|
| 1 Claims cited | Pass | Every figure sentence is cited (P09, P10). |
| 2 Length | Pass | 268 body words. |
| 3 Unsupported claims | Pass | "the data does not provide insights into the specific factors contributing to patient satisfaction or dissatisfaction, which could inform future improvements in service delivery." The sentence states missing data, and the clause only says why that data matters; it tells no one to act, so it is allowed. Failed under the earlier strict rule. |
| 4 Footnotes | Pass | P09 and P10 are cited and footnoted. |

The different-respondents limitation (P10) is stated correctly and its meaning explained. Minor, not violations: "promising" is an evaluative word in a cited sentence; "a shift towards more patients" reads P09's consultation count as a patient count.

---

## Sensitivity and consistency

| Reading | Baseline failures | Team failures | Ratio |
|---|---|---|---|
| Strict (first review): any uncited "could/may" consequence fails | 5 / 5 | 4 / 5 (T01, T02, T03, T05) | 0.8 |
| **Claim-based (reported)** | 5 / 5 | 2 / 5 (T01, T03) | **0.4** |
| Lenient: only citation, figure and attribution errors fail | 5 / 5 | 1 / 5 (T01) | 0.2 |

- The baseline fails under every reading, because each memo has at least one clear error: an uncited figure (T01, T03, T04, T05), a footnote mismatch (T02, T04), a wrong figure (T03) or a misattributed limitation (T01, T04, T05).
- Each rule was applied the same way to both systems. The "could affect the total cost of ownership" sentence appears in both T02 memos and is treated the same in each.
- The conclusion, a ratio of 1.0 or less, holds under all three readings.

## Limits of this review

- One reviewer, who knew which system wrote each memo (not blind).
- One run per topic at temperature 0; run-to-run variation is not measured.
- With five topics, one memo changes a failure rate by 0.2.
