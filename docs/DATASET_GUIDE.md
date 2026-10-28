# EpiRecover — Dataset Preparation & Corpus Ingestion Guide

**Project**: EpiRecover  
**Document**: Epigraphic Corpus & Dataset Guide  
**Target Ingestion**: BharatSHRI (Bharat Shared Repository of Inscriptions), ASI Inscriptions, and Custom Lithic Datasets

---

## 1. Overview of Data Sources

EpiRecover requires two distinct types of datasets:
1. **Historical Inscription Corpus (`data/corpus/`)**: Reference texts, transcriptions, translations, and epigraphic metadata used by the retrieval engine (Stage F) to ground reconstruction candidates in documented historical records.
2. **Ground Truth Evaluation Sets (`data/annotations/`)**: Paired images and expert-verified reference transcriptions used exclusively for benchmarking CER, WER, and ablation studies.

> [!IMPORTANT]
> **Data Leakage Prohibition**: Ground-truth records in `data/annotations/` must be strictly excluded from the retrieval corpus during quantitative evaluation. Including evaluation targets in the retrieval index results in artificial 100% match scores that invalidate research findings.

---

## 2. BharatSHRI Schema & Ingestion

BharatSHRI (Bharat Shared Repository of Inscriptions) by the Archaeological Survey of India (ASI) digitizes ancient lithic epigraphs across India. EpiRecover's `CorpusAdapter` natively ingests records formatted as JSONL, JSON, or CSV.

### Standard Field Schema

Every corpus entry is mapped into the following canonical schema:

| Field Name | Type | Required | Description | Example |
|---|---|---|---|---|
| `id` | `str` | Yes | Unique epigraphic accession identifier | `"BHARATSHRI-ASHO-001"` |
| `script` | `str` | Yes | Script of the inscription | `"brahmi"`, `"tamil-brahmi"`, `"devanagari"` |
| `language` | `str` | Yes | Language of the inscription | `"Prakrit"`, `"Sanskrit"`, `"Old Tamil"` |
| `transcription` | `str` | Yes | Standard Romanized / Devanagari transcription | `"devanampiyasa asokasa raño..."` |
| `iast` | `str` | No | International Alphabet of Sanskrit Transliteration | `"devānaṃpiyasa asokasa rāño..."` |
| `historical_period` | `str` | No | Epoch or dynastic period | `"3rd century BCE (Mauryan)"` |
| `region` | `str` | No | Modern state / geographic region | `"Madhya Pradesh"` |
| `monument` | `str` | No | Specific archaeological site or monument | `"Sanchi Stupa 1, South Gateway"` |
| `source_citation` | `str` | No | Bibliographic reference (Epigraphia Indica, CII) | `"Epigraphia Indica Vol. II, p. 87"` |
| `image_path` | `str` | No | Path to high-resolution stone panel image | `"data/raw/sanchi_01.jpg"` |
| `is_synthetic` | `bool` | Auto | Tagged `True` for synthetic test fixtures | `false` |

### Supported File Formats

#### 1. JSONL (Recommended for large corpora)
```jsonl
{"id":"BH-01","script":"brahmi","language":"Prakrit","transcription":"devanampiye piyadasi raja hevam aha","historical_period":"3rd c. BCE","region":"Bihar","monument":"Barabar Caves"}
{"id":"BH-02","script":"brahmi","language":"Prakrit","transcription":"dhammam caratha","historical_period":"3rd c. BCE","region":"Madhya Pradesh","monument":"Sanchi"}
```

#### 2. JSON Array
```json
[
  {
    "id": "BH-01",
    "script": "brahmi",
    "language": "Prakrit",
    "transcription": "devanampiye piyadasi raja hevam aha",
    "region": "Bihar"
  }
]
```

#### 3. CSV
```csv
id,script,language,transcription,historical_period,region,monument
BH-01,brahmi,Prakrit,devanampiye piyadasi raja hevam aha,3rd c. BCE,Bihar,Barabar Caves
```

---

## 3. Bundled Synthetic Fixtures

To ensure reproducibility without depending on external download servers or non-redistributable proprietary data, EpiRecover includes two synthetic fixtures:

1. **`data/corpus/bharatshri_sample.jsonl`**:
   - Contains 5 synthetic records spanning Brahmi, Tamil-Brahmi, and early Devanagari.
   - Each record contains `"is_synthetic": true` and explicit notice: `"SYNTHETIC FIXTURE for testing the evidence retrieval pipeline. Not a real inscription."`
2. **`data/annotations/sample_eval_set.json`**:
   - Contains 3 ground-truth test pairs with varying damage levels (`low`, `medium`, `high`) and verified reference strings for testing CER/WER calculation and ablation flows.

---

## 4. How to Import Real BharatSHRI / ASI Data

1. Place your exported data file into `data/corpus/`:
   ```bash
   cp my_bharatshri_export.jsonl data/corpus/bharatshri_live.jsonl
   ```
2. In `config/epirecover.yaml`, point `corpus_path` to your new file:
   ```yaml
   evidence_retrieval:
     corpus_path: "data/corpus/bharatshri_live.jsonl"
     similarity_threshold: 0.60
   ```
3. Or load dynamically via the Streamlit Dashboard under **Section 10: Dataset and Experiment Management**.

---

## 5. Monument Image Acquisition Guidelines

When photographing stone inscriptions for EpiRecover processing:
- **Lighting**: Oblique raking light (grazing light at 15°–30° from the stone surface) accentuates carved stroke relief and produces superior OCR binarization compared to direct flat flash.
- **Resolution**: Minimum recommended resolution is 1200×800 pixels with characters spanning at least 25 pixels in height.
- **Angle**: Position camera plane as parallel as possible to the stone panel; if oblique, enable the `--perspective` flag in EpiRecover.
