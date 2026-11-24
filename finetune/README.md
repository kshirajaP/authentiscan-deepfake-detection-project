# Fine-tuning XceptionNet for AuthentiScan — exact steps

This fixes the XceptionNet ~50% confidence issue: its classifier head was
never trained on deepfake data. These scripts fine-tune it, then produce
real accuracy/F1/AUC numbers.

**Folder layout required** — place the `finetune` folder *inside* your
project, as a sibling of `backend`:

```
authentiscan_copy/
  backend/          <- already exists
  frontend/         <- already exists
  finetune/         <- put dataset.py, train_xception.py, evaluate.py, README.md here
```

This layout is what makes the scripts able to find and reuse
`backend/models/detector.py` automatically — no path editing needed if you
follow it exactly.

---

## Step 0 — Get the four files into place

Download `dataset.py`, `train_xception.py`, `evaluate.py`, and this
`README.md`, and put all four inside a new folder:
`C:\Projects\authentiscan_copy\finetune\`

## Step 1 — Activate your existing venv

Same one you already set up and used to run `app.py`:

```bash
cd C:\Projects\authentiscan_copy\backend
venv\Scripts\activate
```

Confirm `(venv)` shows in your prompt and `python --version` says 3.10.x.

## Step 2 — Move into the finetune folder and install extra packages

```bash
cd C:\Projects\authentiscan_copy\finetune
python -m pip install scikit-learn matplotlib
```

(torch/torchvision/timm are already installed from the backend setup — no
need to reinstall.)

## Step 3 — Get a dataset

Fastest option, no approval wait: Kaggle **"140k Real and Fake Faces"**
(`xhlulu/140k-real-and-fake-faces`) — needs only a free Kaggle account.

Download it, then arrange the images into this exact structure inside
`finetune\data\`:

```
finetune/
  data/
    train/real/*.jpg   train/fake/*.jpg
    val/real/*.jpg     val/fake/*.jpg
    test/real/*.jpg    test/fake/*.jpg
```

A few thousand images per class is enough for a first pass — you're only
training a small classifier head, not the whole network from scratch.

## Step 4 — Train

Still inside `finetune\` with the venv active:

```bash
python train_xception.py --data_dir ./data --epochs 10 --unfreeze_last_block
```

If you don't have a GPU on this machine, drop `--unfreeze_last_block`
first to confirm the pipeline runs end-to-end quickly, or run this step on
Google Colab instead (free GPU) — just upload the `finetune` folder and
your `backend/models/detector.py` there, keeping the same relative layout.

This produces two files in `finetune\`:
- `xception_finetuned.pth` — the trained weights
- `training_log.json` — per-epoch accuracy, useful for a training-curve
  figure in the paper

## Step 5 — Evaluate on the held-out test set

```bash
python evaluate.py --data_dir ./data --weights xception_finetuned.pth
```

Produces:
- `metrics.json` — accuracy, precision, recall, F1, AUC
- `confusion_matrix.png` — figure for the paper

## Step 6 — Copy the weights into the app

```bash
copy xception_finetuned.pth ..\backend\models\pretrained\xception_finetuned.pth
```

`detector.py` (already patched) automatically looks for this exact file
and loads it if present — no code changes needed.

## Step 7 — Re-run the app and confirm

```bash
cd ..\backend
python app.py
```

You should now see in the startup log:
```
✅ XceptionNet loaded (fine-tuned on deepfake data: xception_finetuned.pth)
```
instead of the "classifier head is UNTRAINED" warning — and its confidence
on real predictions should move well above ~50%.

## Step 8 — Re-derive the ensemble weights (optional but recommended)

Once you have `metrics.json` accuracy for XceptionNet and can get the same
for MesoNet, open `backend/models/detector.py`, find:

```python
mesonet_weight = 0.4
xception_weight = 0.6
```

and set these proportionally to each model's validation accuracy instead
of the placeholder values — then note this in the paper's Methodology
section.

## Step 9 — Update the paper

Send me `metrics.json` and `confusion_matrix.png` and I'll drop the real
numbers into Section 5.1 of the paper, replacing the "in progress" note.
