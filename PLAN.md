# Research Paper Topic Modelling — Refinement Plan

## 1. Project Summary
This project aims to classify research papers into multiple subject categories based on their abstracts using Natural Language Processing (NLP).

- **Dataset:** Kaggle — "Topic Modeling for Research Articles"
- **Current Assets:**
  - `topic-modelling-for-research-articles (1).ipynb` — main notebook
  - `archive.zip` — raw dataset (`Train.csv`, `Test.csv`, `Tags.csv`, `sample_sub.csv`)
  - `README.md` — brief description

## 2. Current Pipeline (As-Is)
1. **Load Data** — `Train.csv`, `Test.csv`, `Tags.csv`
2. **Preprocessing**
   - Strip non-alphabetic characters.
   - spaCy lemmatization + English stopword removal.
   - Store tokens in `Words` column.
3. **Feature Extraction**
   - Train Word2Vec (CBOW, 100-d, `min_count=2`) on training tokens.
   - Average word vectors per document → 100-D embedding.
   - Concatenate 4 main binary labels → 104-D feature vector.
4. **Modeling & Evaluation**
   - 80/20 `train_test_split`.
   - Models: `OneVsRest(KNN)`, `OneVsRest(SVC)`, `OneVsRest(LogisticRegression)`, and a custom XGBoost setup (argmax-based, not true multi-label).
   - Metric: micro-F1 only.
5. **Inference**
   - `pipeline()` function preprocesses `Test.csv` and predicts with KNN.

## 3. Identified Issues
| Area | Issue | Impact |
|---|---|---|
| **Preprocessing** | Aggressive cleaning strips digits/hyphens (e.g., "COVID-19", "3D"). | Loss of discriminative tokens. |
| | No lowercasing before stopword filtering. | Uppercase stopwords retained. |
| | spaCy model reloaded inside `preprocess()` on every call. | Severe inefficiency. |
| | Word2Vec trained on small corpus with `min_count=2`. | Rare scientific terms discarded. |
| **Features** | Simple mean of word vectors ignores TF-IDF and word order. | Weak document representations. |
| | 4 main labels concatenated as input features. | Potential target leakage if unavailable at test time. |
| | No feature scaling for KNN/SVM. | Distance metrics skewed. |
| **Modeling** | No random seed / no cross-validation. | Unreliable, non-reproducible scores. |
| | XGBoost uses argmax sigmoid hack instead of binary relevance. | Incorrect multi-label formulation. |
| | SVC with RBF on 100-D dense embeddings. | Very slow training. |
| | No class-imbalance handling. | Under-represented sub-categories poorly predicted. |
| **Evaluation** | Only micro-F1 reported. | No insight into per-label or rare-class performance. |
| | No per-label error analysis. | Hard to identify failure modes. |
| **Engineering** | Slow Python loops in `summarizer` and `combiner`. | Scalability bottleneck. |
| | No model serialization or submission generation. | Cannot reproduce or submit predictions easily. |

## 4. Refinement Roadmap

### Phase A — Stability & Baselines
1. **Reproducibility**
   - Set `random_state` in `train_test_split` and all model constructors.
   - Add a `config.py` / top-of-notebook config cell for seeds, paths, and hyperparameters.
2. **Proper Evaluation**
   - Report **micro-F1**, **macro-F1**, **weighted-F1**, **Hamming loss**, and **subset accuracy**.
   - Generate a full `classification_report` per label.
   - Introduce **Stratified K-Fold** (e.g., iterative stratification for multi-label via `scikit-multilearn`).
3. **Strong Baselines**
   - Add **TF-IDF + Logistic Regression** baseline.
   - Add **TF-IDF + Linear SVM** (`LinearSVC`) baseline.
   - Compare these against the current Word2Vec pipeline.

### Phase B — Text Preprocessing & Features
1. **Cleaner Preprocessing**
   - Use a regex that preserves intra-word hyphens and alphanumerics where useful.
   - Lowercase consistently before stopword removal.
   - Load `en_core_web_sm` **once** and reuse the `nlp` object.
2. **Better Embeddings**
   - Evaluate **TF-IDF weighted Word2Vec** averaging.
   - Try **pre-trained embeddings** (e.g., GloVe, BioWordVec) or domain-specific transformers.
3. **Feature Hygiene**
   - Confirm whether the 4 main binary labels exist in `Test.csv`. If not, remove them from the feature vector.
   - Add `StandardScaler` before distance-based classifiers (KNN, SVM).

### Phase C — Robust Multi-Label Modeling
1. **Fix XGBoost**
   - Use `XGBClassifier(objective='binary:logistic')` wrapped in `MultiOutputClassifier` for true binary-relevance multi-label learning.
2. **Class Imbalance**
   - Use `class_weight='balanced'` in Logistic Regression / Linear SVC.
   - For tree-based models, use `scale_pos_weight` per label.
3. **Threshold Tuning**
   - Optimize per-label decision thresholds on validation probabilities (maximize F1) rather than default 0.5.

### Phase D — Hierarchical & Advanced Models
1. **Hierarchical Approach**
   - Train a first-level classifier for the 4 main fields.
   - Train second-level classifiers for sub-categories conditioned on the predicted main field.
   - Compare hierarchical vs. flat multi-label performance.
2. **Transformer Upgrade (High Impact)**
   - Fine-tune **SciBERT** or **SPECTER** for multi-label classification.
   - Use the `transformers` library with a multi-label classification head.
3. **Ensembling**
   - Stack TF-IDF + Linear SVM + Transformer embeddings with a meta-classifier (e.g., Logistic Regression) to capture complementary signals.

### Phase E — Production & Submission
1. **Code Refactoring**
   - Replace slow loops in `summarizer` / `combiner` with vectorized NumPy/Pandas operations.
   - Build modular functions: `preprocess()`, `extract_features()`, `train_model()`, `evaluate()`, `predict_and_submit()`.
2. **Model Serialization**
   - Save trained models / vectorizers with `joblib` or `pickle`.
3. **Submission Pipeline**
   - Generate `submission.csv` in the exact Kaggle format (`Id` + one-hot label columns).
   - Validate column names and value ranges before saving.

## 5. Success Criteria
- Reproducible validation scores (fixed random seed, cross-validation).
- At least one strong baseline (TF-IDF + Linear model) outperforming the current Word2Vec + KNN setup.
- Correct multi-label formulation for XGBoost (binary relevance).
- Full multi-metric evaluation (micro, macro, weighted F1, Hamming loss).
- Working end-to-end inference pipeline that writes a valid `submission.csv`.

## 6. Next Steps (Immediate Actions)
- [ ] Unzip `archive.zip` and inspect `Test.csv` columns (check for main-label availability).
- [ ] Run the notebook end-to-end with a fixed `random_state` and record current micro-F1 baseline.
- [ ] Implement TF-IDF + Logistic Regression baseline with proper multi-label metrics.
- [ ] Refactor preprocessing to reuse spaCy model and use a more lenient cleaner.
