# Predicting response to total neoadjuvant therapy for locally advanced rectal cancer

Modeling code for the clinical and endoscopic image study. The patient-level outcome is pathologic complete response after immediate surgery or sustained clinical complete response at two years during watch-and-wait. All inputs, including the GastroNet-5M pretrained weights, are supplied locally by the investigator. This repository contains source, a clinical feature dictionary, documentation and synthetic tests.

| Analysis | Implementation |
|---|---|
| Structured clinical model | One CatBoost classifier per training partition, with training-derived class weights, numeric imputation, native categorical predictors and inner-fold probability calibration |
| Endoscopic image model, Figure 2 | Frozen GastroNet-5M ViT-B/14 (DINOv2, 336 pixels) and ResNet-50 (DINOv1, 224 pixels). Mean frame embeddings, training-only standardization and PCA, then L2 logistic regression. Average probabilities over six views and 16 fixed frame subsets, then equally across encoders |
| Multimodal model, Figure 3 | The calibrated CatBoost logit receives a bounded adjustment from the ViT probability. The weight increases with clinical uncertainty and disagreement. The mrTRG–endoscopy benchmark is a separate two-predictor ridge logistic regression |
| Calibration and classification, Figure 4 | Compare calibrated probabilities on patients with both restaging assessments. Select one Youden cutoff on all development out-of-fold predictions and apply it at full precision to every pathway, subgroup and external patient |
| Contributions, Figure 5 | Held-out CatBoost TreeSHAP terms on the calibrated log-odds scale plus the exact image adjustment. Panel B removes nine clinician-endoscopy fields and fits an ungated combination on inner-fold predictions. Terms are summed within patient and clinical group before absolute values and bootstrap intervals |
| Image sensitivity, Figure 6 | Compute observed-class gradient-times-activation in the last ViT block for the privately selected development frames, register all six maps to the source frame and sum them. Output consists of numerical maps in source-image coordinates |
| Descriptive and development-only analyses | Table 1 summary from prepared descriptive fields, pooled scores for historical model comparisons, and the supplementary frozen versus fine-tuned encoder comparison |

## Install

Use Python 3.11. Study package versions are pinned in the project metadata. From the repository root, install the core model and reporting code with the test dependencies.

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

For image encoding, sensitivity maps or the supplementary training experiment, install the image dependencies in the same environment with `python -m pip install -e '.[images,test]'`. The clinical and embedding-based analysis runs on CPU. The image analyses were run with PyTorch 2.10 and CUDA.

## Fit and evaluate the manuscript models

The commands use shell variables for private input and output directories. Use new output directories for each run. The development fold file and original image subsamples are inputs. [Input schemas and optional analyses](docs/inputs.md)

```bash
P=../private_inputs
O=../private_results
F=config/clinical_features.json

endo-cr development --patients "$P/development.csv" --features "$F" \
  --inner-folds "$P/inner_folds.csv" --vision "$P/dev_embeddings.npz" \
  --noendo-sensitivity --output "$O/development"

endo-cr predict-external --patients "$P/development.csv" --features "$F" \
  --inner-folds "$P/inner_folds.csv" --vision "$P/dev_embeddings.npz" \
  --external-patients "$P/external_masked.csv" --external-vision "$P/ext_embeddings.npz" \
  --output "$O/masked"

endo-cr evaluate-external --predictions "$O/masked/external_predictions_masked.csv" \
  --lock "$O/masked/prediction_lock.json" --labels "$P/external_labels.csv" \
  --output "$O/evaluated"

endo-cr report --predictions "$O/development/development_oof_predictions.csv" \
  "$O/evaluated/external_predictions_evaluated.csv" --pathways "$P/pathways.csv" \
  --contributions "$O/development/patient_feature_contributions.csv" --features "$F" \
  --output "$O/report"
```

The report writes the AUROCs and their DeLong intervals, the two-sided paired tests, average precision, Brier score, log loss, calibration, Wilson classification intervals, pathway counts and Figure 5 contributions. Figures 2–5 are rendered from those calculations. Figure 1 is a conceptual schematic. The sensitivity-map command is documented separately.

## Interpretation

The image comparison requires recorded endoscopy. The multimodal comparison requires both endoscopy and mrTRG. The classification analysis includes patients with model predictions even when a restaging assessment is missing. The near-complete subgroup is the union of near-complete endoscopic response and mrTRG 2. The report labels the fixed clinician rule and the probabilistic benchmark separately.

The primary paired comparisons condition on the fitted predictions. The external comparisons and the development-selected decision cutoff were evaluated retrospectively. Figure 5 changes both the clinical inputs and the fusion equation between panels. Figure 6 describes sensitivity of the held-out ViT component. Patient images, display redactions and the clinical interpretation of individual examples remain with the investigators.
