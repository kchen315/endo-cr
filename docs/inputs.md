# Prepared inputs and additional analyses

Keep the private inputs and outputs outside the source tree. Use a consistent pseudonym for each patient across modalities. Development and external patients must be disjoint.

## Patient tables and folds

Supply one row per patient, in the original fitting order, and preserve the exact 63 column names in `config/clinical_features.json`, including the leading space in the first name. Preserve clinical category spellings and missing values. Training partitions determine feature typing, filtering, numeric medians and calibration.

| Input | Fields |
|---|---|
| Development clinical CSV | `patient_id`, all predictors, binary `outcome`, `outer_fold` (1–5), `human_endoscopy`, `human_mri` |
| External clinical CSV | Same identifiers, predictors and clinician scores, with outcome and fold omitted |
| External label CSV | Exactly `patient_id,outcome` |
| Inner fold CSV | `patient_id,outer_context_fold,branch,inner_fold`. For each outer fold 1–5, list each outer-training patient exactly once per branch, with inner fold 1–3. The clinical branch uses the original joint pathway/outcome splits. The vision branch supplies the outcome-stratified memberships for training the ungated fuser. Context 0 uses the full development cohort and the clinical branch for external refitting |
| Evaluation-only pathway CSV | `patient_id,pathway,recurrence_within_2y`, covering the patients in the report. Pathways are `immediate_surgery`, `watch_and_wait`, or `other`. Code documented recurrence/regrowth within two years as 1 and reviewed absence of that event as 0. Supply the adjudicated indicator for every development WW patient with a negative endpoint. Leave unavailable or inapplicable values blank elsewhere |

Code complete, near-complete, partial/incomplete, and no response/progression as 1, 2/3, 1/3 and 0 respectively. Code mrTRG 1–5 as 1, .75, .50, .25 and 0. Use blank entries for unavailable clinician assessments. The benchmark standardizes both observed scores within the training partition and fits logistic regression with C = 1. The fixed binary rule requires complete endoscopic response and mrTRG 1 or 2.

The clinical model uses 100 trees, depth 5, learning rate .03, L2 leaf regularization 3 and a positive-class weight equal to the training negative-to-positive count ratio. Candidate calibration methods are identity, temperature scaling and Platt scaling. The no-endoscopy analysis cross-fits raw clinical margins on the same inner memberships as the ViT scores. Its nonnegative logistic slopes use ridge C = 1 and an unpenalized intercept.

## Frozen image features

Use one NumPy NPZ per cohort. Store `patient_ids` as Unicode strings. For each prefix `dino` (ViT-B/14, 768 components) and `rn50` (ResNet-50, 2,048 components), supply `<prefix>_identity` with shape `(patients, components)` and `<prefix>_application` with shape `(6, 16, patients, components)`. The first array contains each patient's full-bag mean and the second contains the mean for each view/subset. The readout fits a scaler, 16 principal components and L2 logistic regression with C = .1 on full-bag identity means. The paired encoders use identical patient and frame memberships.

For exact historical numerical analysis, use the original frozen features. To generate features locally from prepared frames, supply a JSON list with one object per patient. Each object contains a string `patient_id`, an ordered list of RGB image paths `frames`, and `subsets`, 16 lists of sorted, zero-based frame indices of length ceil(0.75 × frame count). Each draw samples without replacement. Paths may be absolute or relative to the manifest. Optional `extraction_indices` specify original cohort-wide frame order for batched encoding. Together these indices must form a permutation of 0 through the total frame count minus one. Preserve the original frame order and draws when changing pseudonyms.

The six views are identity, brightness ×0.95 and ×1.05, contrast ×0.95 and ×1.05, and a centered 95% crop. Each view is resized with preserved aspect ratio, centered on black padding and normalized by the encoder's channel statistics. Supply the local GastroNet-5M pretrained checkpoints. The loader checks the SHA-256 digest of each file before loading. [GastroNet ViT model card](https://huggingface.co/BONS-AI-TUE-AMC/GastroNetDinov2)

```bash
python -m endo_cr_analysis.images encode --manifest "$P/dev_frames.json" \
  --dino "$P/dinov2.pth" --rn50 "$P/RN50_GastroNet-5M_DINOv1.pth" --output "$O/dev_encoded"
```

Use the resulting `embeddings.npz` as the development vision input. Repeat for the external cohort. The algorithm also accepts original analysis-ready embedding files directly. Hardware, batch shape and numerical libraries can affect recomputed features.

For Figure 6, supply a private panel CSV with `panel,patient_id,frame_number`. Panel and frame numbers start at 1. Use the final reviewed display selection, including the four replacement true-response frames. The code fits the PCA/logistic head using only the selected patient's outer-training partition and computes a signed map and a positive normalized map. Text blurring for the historical displays took place after model inference.

```bash
python -m endo_cr_analysis.images maps --manifest "$P/dev_frames.json" --patients "$P/development.csv" \
  --vision "$P/dev_embeddings.npz" --panels "$P/panels.csv" --dino "$P/dinov2.pth" --output "$O/maps"
```

## Table 1

`python -m endo_cr_analysis.cohort --patients "$P/descriptive.csv" --output "$O/cohort"` takes a prepared CSV with `patient_id,cohort,outcome` and the following descriptive fields.

| Fields | Representation |
|---|---|
| age, bmi, distance, cea, cycles, images, days | Numeric values on the recorded scale. Distance is centimeters from the anal verge and days is the interval from the treatment anchor to endoscopy |
| sex, ct, cn, location, mmr, emvi, pathway, sequence, fractions, endo, mrtrg | Display categories listed in `cohort.CHARACTERISTICS`. Use missing for unknown/unrecorded. The descriptive pathway labels are Immediate surgery, Watch-and-wait and Other, as recorded |

The table reports median and interquartile range of observed numeric values and category counts divided by the full outcome-group denominator. Source cleaning and outcome adjudication precede these inputs.

## Development-only comparisons

For the supplementary fine-tuning comparison, use the separate development-only runner. It evaluates two GastroNet-5M encoders, three freezing schedules and three patient pooling heads per outer fold. Each recipe has three inner fits and an outer-training refit. All schedules run 25 epochs. Each encoder has a direct head, a logistic readout of mean features and, for gated pooling, a logistic readout of pooled features. The lowest inner Brier score, with log loss as tie-breaker, selects one prediction per encoder for each outer fold. Fixed recipes and equal-probability ensembles are reported separately.

```bash
python -m endo_cr_analysis.finetuning --patients "$P/development.csv" --inner-folds "$P/inner_folds.csv" \
  --manifest "$P/dev_frames.json" --dino "$P/dinov2.pth" --rn50 "$P/RN50_GastroNet-5M_DINOv1.pth" \
  --seed 42 --output "$O/finetuning_seed42"
```

Use the original development row order, joint pathway/outcome inner memberships (`branch=clinical`) and original within-bag path sort order for this experiment. Run seeds 42, 2026 and 2027 on the canonical folds, then seed 42 on the second supplied outer partition with its matching inner memberships. Each run comprises 360 fits. The primary manuscript models use the frozen encoders regardless of these exploratory comparisons.

The historical clinical, backbone and pooling screens used earlier feature sets, folds and in some cases different cohorts. Their reported metrics can be regenerated from those experiments' held-out probabilities with `python -m endo_cr_analysis.comparisons --predictions "$P/historical_oof.csv" --output "$O/historical"`. Supply a long CSV with `experiment,model,patient_id,outcome,probability`. Each experiment must contain a common set of patients for every compared model. The script scores the archived predictions within each experiment. The original preprocessing and historical parameter searches are outside the release.

## Numerical reproduction

The reference for the manuscript is the original frozen embedding bank. Re-extracting features from the prepared RGB images reproduced the displayed image AUROCs but slightly changed the development paired P value because the historical ResNet-50 identity features and recomputed features differed. The original frozen features reproduce the reported P value.
