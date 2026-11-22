# Results to add to the IEEE paper

The draft is `ieee_research_paper.tex`. Bracketed text (`[ ... ]`) marks information you must supply or verify. Do not invent or estimate scientific results.

## Before collecting final results

1. **Fix FedAvg client weighting.** In `client.py`, `len(self.train_ds)` counts batches because the dataset is already batched. Return the number of images/examples instead (and use an example count consistently for aggregation). Re-run the experiment after this correction.
2. **Replace the current “before FL” baseline.** `server.py` evaluates a newly initialized model before training. It is not a trained baseline. Train a local-only model on the held-out site's training split, and preferably a centralized model on the same pooled training data; compare them with FL on exactly the same test set and protocol.
3. **Make an independent test split.** The current code repeatedly evaluates the Aravid validation split, so it is not an untouched test set. If image files from the same patient can occur more than once, split by patient rather than by image. Keep the test set out of training, tuning, and threshold selection.
4. **Use reproducible runs.** Record software/library versions, hardware, random seeds, exact split procedure, training duration, and whether runs were repeated. Prefer multiple seeds and report mean and variability or confidence intervals.

## Populate the manuscript

- **Author block:** names, department/institution, location, and contact email.
- **Abstract:** final held-out test AUC, accuracy, sensitivity, specificity, and the uncertainty measure used. State the test-set size and number of patients where available.
- **Table I:** the current image counts were counted from supported image file extensions in the workspace. Verify them against the final curated dataset and report patient counts separately if available.
- **Table II (per-round):** exact logged loss, accuracy, AUC, and one additional metric (for example sensitivity) for the initial model and each of rounds 1–5. Clearly label this as validation if it is not test data. The starter project does not save all these values to a structured results file.
- **Table III (baselines):** accuracy, AUC, sensitivity, specificity for local-only, centralized, and federated models, all evaluated on the same untouched test examples. Include confidence intervals in the table or in the text.
- **Figure:** regenerate a plot from exact experiment logs. Show the metric by round and label validation/test data and the baselines. Do not digitize the existing `fl_results.png` to obtain exact values.
- **Results narrative:** describe exact final metrics and uncertainty, compare methods fairly, report the selected operating threshold, and state the confusion-matrix counts. Mention per-site outcomes if available.
- **Discussion:** explain observed performance, class/site imbalance, failures, and limits. Do not claim privacy protection, clinical readiness, or generalization beyond the evaluated sites without supporting evidence.

## Important interpretation of the existing plot

The plot appears to show AUC increasing from approximately 0.91 to 0.96 and accuracy around 0.81--0.83 across rounds, while the dashed baseline is around AUC 0.67 and accuracy 0.35. These are only visual approximations, not authoritative values. More importantly, the code's dashed baseline is an **untrained** model, so it cannot support a claim that FL outperformed a trained baseline. Re-run and report exact values only after the baseline and evaluation protocol have been corrected.
