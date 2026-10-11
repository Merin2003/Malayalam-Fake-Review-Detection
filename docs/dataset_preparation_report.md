# Malayalam Fake Review Detection System — Dataset Preparation & AI Quality Control Report

**Prepared by:** Member 1 (Dataset & Data Pipeline Lead)  
**Date:** October 10, 2026  
**Project:** Malayalam Fake Review Detection System  
**Dataset Version:** `v1.0-final-3class`

---

## 1. Executive Summary

Member 1 has successfully completed the full dataset preparation, quality assurance, contradiction correction, template artifact removal, and stratified leakage-free split generation for the Malayalam Fake Review Detection System.

### Key Milestones Achieved:
1. **Original Human Review Dataset Cleaned:** Cleaned 4,080 raw records down to **3,083 unique human-authored Malayalam reviews** with zero missing values and mapped labels to:
   - `0` — **Human Real** (1,540 reviews, 49.00%)
   - `1` — **Human Fake** (1,543 reviews, 49.09%)
2. **AI Review Quality Scan & Artifact Removal:**
   - Identified and stripped parenthetical suffix artifacts (`(റിവ്യൂ 1)`, `(ഓപ്ഷൻ 1)`, etc.) across 140 generated reviews.
   - Discovered that the initial 200 AI rows were comprised of 60 distinct review templates.
   - Identified and resolved 14 logical contradictions (e.g., negative experience descriptions coupled with positive ending phrases like `"അടിപൊളി എക്സ്പീരിയൻസ്"` or `"വീണ്ടും വാങ്ങും"`).
3. **Final 3-Class Dataset Constructed:** Merged human reviews and cleaned AI reviews into [`final_dataset.csv`](file:///c:/Users/HP/Documents/Malayalam_Fake_Review_Detection/data/processed/final_dataset.csv) totaling **3,143 rows**:
   - `0` — **Human Real**: 1,540 (49.00%)
   - `1` — **Human Fake**: 1,543 (49.09%)
   - `2` — **AI Generated**: 60 (1.91%)
4. **Stratified & Leakage-Free Data Splits Created:** Generated `train_final.csv` (2,011 rows), `validation_final.csv` (503 rows), and `test_final.csv` (629 rows) with zero text overlap between splits.

---

## 2. Dataset Preparation Lifecycle

### 2.1 Original Raw Datasets Cleaning
- **Raw Files Processed:** `20kto3k (1).csv` (3,000 rows), `mal20kto1000 (1).csv` (1,000 rows), and `malayalam sample aug 27 (1).csv` (80 rows).
- **Whitespace Normalization:** Trimmed leading/trailing whitespace and collapsed multiple internal spaces.
- **Deduplication:** Deduplicated exact review texts to prevent data leakage and bias.
- **Master File:** Saved clean human reviews to [`master_dataset.csv`](file:///c:/Users/HP/Documents/Malayalam_Fake_Review_Detection/data/cleaned/master_dataset.csv) (3,083 rows).

### 2.2 AI Review Generation & Quality Inspection
- **Raw AI Input Processing:** Raw generated AI reviews (`native_100.csv` and `translation_100.csv`, 200 rows) were scanned for template artifacts and contradictions.
- **Artifact Discovery:** Suffix tags like `(റിവ്യൂ 1)`, `(ഓപ്ഷൻ 2)` were detected in 140 rows, created by prompt template iterations.
- **Template Consolidation:** Stripping these suffixes revealed 60 unique underlying review texts (30 native style, 30 translation style).
- **Contradiction Resolution:** Script [`clean_ai_reviews.py`](file:///c:/Users/HP/Documents/Malayalam_Fake_Review_Detection/src/clean_ai_reviews.py) corrected logical clashes:
  - *Before:* `"ഈ Home Entertainment ഉൽപ്പന്നം വിചാരിച്ചത്ര മികച്ചതല്ല. ഞാൻ ശുപാർശ ചെയ്യുന്നു."`
  - *After:* `"ഈ Home Entertainment ഉൽപ്പന്നം വിചാരിച്ചത്ര മികച്ചതല്ല. ഞാൻ ശുപാർശ ചെയ്യുന്നില്ല."`
- **Output Reviewed AI File:** [`ai_generated_reviews_reviewed.csv`](file:///c:/Users/HP/Documents/Malayalam_Fake_Review_Detection/data/augmentation/ai_generated_reviews_reviewed.csv) (60 rows, label `2`).

---

## 3. Final Dataset Composition & Class Distribution

The final combined dataset is saved in [`data/processed/final_dataset.csv`](file:///c:/Users/HP/Documents/Malayalam_Fake_Review_Detection/data/processed/final_dataset.csv).

| Label | Class Name | Source Type | Count | Percentage |
| :---: | :---: | :---: | :---: | :---: |
| **0** | **Human Real** | Human-authored genuine | 1,540 | 49.00% |
| **1** | **Human Fake** | Human-authored fake | 1,543 | 49.09% |
| **2** | **AI Generated** | AI-generated review | 60 | 1.91% |
| **Total** | | | **3,143** | **100.00%** |

---

## 4. Final Stratified Leakage-Free Splits

Generated via [`src/split_final_dataset.py`](file:///c:/Users/HP/Documents/Malayalam_Fake_Review_Detection/src/split_final_dataset.py) with seed `42`.

| Split File | Total Rows | Human Real (0) | Human Fake (1) | AI Generated (2) | Split Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: |
| [`train_final.csv`](file:///c:/Users/HP/Documents/Malayalam_Fake_Review_Detection/data/processed/train_final.csv) | **2,011** | 986 | 987 | 38 | ~64% |
| [`validation_final.csv`](file:///c:/Users/HP/Documents/Malayalam_Fake_Review_Detection/data/processed/validation_final.csv) | **503** | 246 | 247 | 10 | ~16% |
| [`test_final.csv`](file:///c:/Users/HP/Documents/Malayalam_Fake_Review_Detection/data/processed/test_final.csv) | **629** | 308 | 309 | 12 | ~20% |
| **Total** | **3,143** | **1,540** | **1,543** | **60** | **100%** |

### Leakage Verification Results:
- **Train vs Validation text overlap:** 0 reviews
- **Train vs Test text overlap:** 0 reviews
- **Validation vs Test text overlap:** 0 reviews

---

## 5. Member 1 Deliverables Summary

All Member 1 deliverables have been generated and validated in the workspace:

| Deliverable | File Path | Status |
| :--- | :--- | :---: |
| **Original Cleaned Dataset** | [`data/cleaned/master_dataset.csv`](file:///c:/Users/HP/Documents/Malayalam_Fake_Review_Detection/data/cleaned/master_dataset.csv) | Completed |
| **AI Cleaning & Correction Script** | [`src/clean_ai_reviews.py`](file:///c:/Users/HP/Documents/Malayalam_Fake_Review_Detection/src/clean_ai_reviews.py) | Completed |
| **Reviewed AI Candidate Dataset** | [`data/augmentation/ai_generated_reviews_reviewed.csv`](file:///c:/Users/HP/Documents/Malayalam_Fake_Review_Detection/data/augmentation/ai_generated_reviews_reviewed.csv) | Completed |
| **Updated AI Quality Scan** | [`results/ai_review_quality_check_reviewed.csv`](file:///c:/Users/HP/Documents/Malayalam_Fake_Review_Detection/results/ai_review_quality_check_reviewed.csv) | Completed |
| **Dataset Build Script** | [`src/build_final_dataset.py`](file:///c:/Users/HP/Documents/Malayalam_Fake_Review_Detection/src/build_final_dataset.py) | Completed |
| **Final 3-Class Dataset** | [`data/processed/final_dataset.csv`](file:///c:/Users/HP/Documents/Malayalam_Fake_Review_Detection/data/processed/final_dataset.csv) | Completed |
| **Final Split Script** | [`src/split_final_dataset.py`](file:///c:/Users/HP/Documents/Malayalam_Fake_Review_Detection/src/split_final_dataset.py) | Completed |
| **Final Data Splits** | [`data/processed/train_final.csv`](file:///c:/Users/HP/Documents/Malayalam_Fake_Review_Detection/data/processed/train_final.csv)<br>[`data/processed/validation_final.csv`](file:///c:/Users/HP/Documents/Malayalam_Fake_Review_Detection/data/processed/validation_final.csv)<br>[`data/processed/test_final.csv`](file:///c:/Users/HP/Documents/Malayalam_Fake_Review_Detection/data/processed/test_final.csv) | Completed |
| **Final Class Breakdown Report** | [`results/final_split_summary.csv`](file:///c:/Users/HP/Documents/Malayalam_Fake_Review_Detection/results/final_split_summary.csv) | Completed |
| **Dataset Documentation** | [`docs/dataset_preparation_report.md`](file:///c:/Users/HP/Documents/Malayalam_Fake_Review_Detection/docs/dataset_preparation_report.md) | Completed |

---

## 6. Handover Instructions for Model Training Lead (Member 2 / Member 3)

1. **Target Columns:** Use `review_text` as feature input and `label` (0, 1, 2) as target integer label.
2. **Class Imbalance Note:** Class 2 (AI Generated) has 60 instances due to deduplication of prompt suffix variants. When training classifiers (e.g. TF-IDF + Logistic Regression, mBERT, IndicBERT, XLM-RoBERTa), consider using class weighting (`class_weight='balanced'`) or SMOTE / resampling techniques.
3. **Split Integrity:** Always evaluate models on `validation_final.csv` for hyperparameter tuning and keep `test_final.csv` strictly for final evaluation.
