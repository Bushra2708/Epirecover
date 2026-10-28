# EpiRecover — Quantitative Evaluation Protocol

**Project**: EpiRecover  
**Document**: Research Evaluation Protocol & Experimental Methodology  
**Target Metrics**: Character Error Rate (CER), Word Error Rate (WER), Candidate Accuracy, Ablation Impact

---

## 1. Evaluation Objectives

The objective of quantitative evaluation in EpiRecover is to measure:
1. **Preprocessing Efficacy**: Does adaptive OpenCV preprocessing (CLAHE, bilateral filtering, adaptive binarization) reduce character error rate over raw monument photographs?
2. **Reconstruction Accuracy**: Does multimodal evidence retrieval and heuristic ranking generate candidates that more closely match verified ground-truth epigraphic readings than raw OCR alone?
3. **Robustness Under Damage**: How do error rates scale across differing levels of physical damage (surface erosion vs. structural fracture)?
4. **Component Contribution (Ablations)**: What is the isolated contribution of each pipeline component (preprocessing, damage masking, corpus retrieval, candidate ranking)?

---

## 2. Formal Metric Formulations

### 2.1 Character Error Rate (CER)
Character Error Rate evaluates edit distance at the individual glyph/grapheme level:
$$\text{CER} = \frac{S + D + I}{N} = \frac{\text{Levenshtein}(H, R)}{|R|}$$
where:
- $H$ is the reconstructed hypothesis string.
- $R$ is the verified ground-truth reference string.
- $S$ is the number of character substitutions.
- $D$ is the number of character deletions.
- $I$ is the number of character insertions.
- $N = |R|$ is the total number of characters in the reference.

*Properties*:
- $\text{CER} = 0.0$: Perfect reconstruction.
- $\text{CER} > 1.0$: Possible when insertions exceed the length of the reference.
- If $R$ is empty or unavailable, $\text{CER}$ is mathematically undefined and reported as `unavailable`.

### 2.2 Word Error Rate (WER)
Word Error Rate evaluates token-level edit distance:
$$\text{WER} = \frac{S_w + D_w + I_w}{N_w}$$
where operations and lengths are computed across whitespace-delimited word tokens.

---

## 3. Experimental Protocols & Baselines

### 3.1 Standard Baselines
Every evaluation run benchmarks against two standardized baselines:
1. **Raw OCR Baseline ($B_{\text{raw}}$)**: OCR executed directly on the unprocessed monument photograph without filtering, CLAHE, or morphological cleaning.
2. **Dictionary-Only Baseline ($B_{\text{dict}}$)**: OCR followed by direct greedy dictionary lookup (`data/script_mappings.json`), without corpus retrieval or multimodal ranking.

### 3.2 Controlled Ablation Framework
The evaluator supports isolating individual pipeline stages:

| Configuration ID | Preprocessing | Damage Mask | Evidence Retrieval | Candidate Ranking | Target Measurement |
|---|---|---|---|---|---|
| `ABL-01` (Baseline) | Off | Off | Off | Off | Raw OCR performance |
| `ABL-02` (CV Only) | **On** | Off | Off | Off | Pure image enhancement gain |
| `ABL-03` (No Corpus)| **On** | **On** | Off | Heuristic | Local visual + OCR impact |
| `ABL-04` (Full System)| **On** | **On** | **On** | **On** | End-to-end EpiRecover pipeline |

---

## 4. Stratified Analysis

Results must be reported across two orthogonal axes:
1. **By Script Type**:
   - Brahmi (Mauryan, Kushana, Gupta)
   - Tamil-Brahmi
   - Early Devanagari / Nagari
   - Grantha / Early Tamil
2. **By Degradation Severity**:
   - `Low`: Minor superficial surface abrasion; glyph strokes contiguous.
   - `Medium`: Significant weathering; 20%–50% stroke erosion; local illegibility.
   - `High`: Severe fractures, spalling, or $>50\%$ missing stroke area.

---

## 5. Research Integrity & Scientific Rigor

1. **Strict Reference Requirement**: Metrics are only calculated when an explicit ground-truth transcription file exists in `data/annotations/`. Under no circumstances will benchmark scores or error rates be estimated or simulated.
2. **Missing Data Handling**: When reference data is missing, the evaluator outputs:
   ```json
   {
     "cer": null,
     "wer": null,
     "metric_unavailable_reason": "No verified ground-truth transcription available. Provide reference in data/annotations/"
   }
   ```
3. **Data Leakage Immunity**: The `exclude_ids` mechanism ensures that inscriptions undergoing testing are blocked from retrieval indexing.
4. **Reproducibility**: Every evaluation run exports:
   - `<experiment_id>_summary.json`: High-level aggregate statistics, ablation config, and timestamps.
   - `<experiment_id>_details.csv`: Per-inscription row-level CER, WER, reference lengths, and error breakdowns.
