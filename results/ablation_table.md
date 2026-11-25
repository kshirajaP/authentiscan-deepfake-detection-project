| Model | Accuracy (95% CI) | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| XceptionNet (fine-tuned) | 95.9% (95.6-96.1) | 96.4% | 95.3% | 0.958 | 0.993 |
| MesoNet-4 (pretrained) | 49.1% (48.5-49.8) | 45.4% | 8.6% | 0.145 | 0.495 |
| Ensemble (w_X=1.00, chosen on val) | 95.9% (95.6-96.1) | 96.4% | 95.3% | 0.958 | 0.993 |
| Ensemble (fixed 0.6/0.4) | 95.4% (95.1-95.7) | 97.4% | 93.3% | 0.953 | 0.979 |