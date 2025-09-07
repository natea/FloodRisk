# Accelerating Pluvial Flood Modeling with Existing CNN Architectures

## Purpose
Evaluate two open‑source repositories for components we can reuse to speed up a **DEM + rainfall → flood depth/extent** model at city scale, focusing on architectures and pretrained weights (not data pipelines).

- Repos:  
  1) `UNITAR-UNOSAT/UNOSAT-AI-Based-Rapid-Mapping-Service` (rapid flood segmentation from satellite)  
  2) `SRKabir/Rapid_FloodModelling_CNN` (CNN surrogate for flood depths)

---

## Quick Take
- **Start from a U‑Net** (Repo 1) with a pretrained encoder; adapt inputs to `[DEM, rainfall]`, output either **depth (regression)** or **extent (segmentation)**.  
- **Borrow the simulation-first philosophy** from Repo 2 (train on physics-model outputs) and its emphasis on hyperparameter tuning.  
- **Do not** port Repo 2’s exact network (1D/time‑series → fixed grid); it is tightly coupled to fluvial boundary-condition inputs and a fixed domain.

---

## Repo 1 — UNOSAT Rapid Mapping (U‑Net Segmentation)
**What it has**
- Encoder–decoder **U‑Net** built with Fastai/PyTorch.  
- **Pretrained ResNet encoder** (ImageNet) + skip connections.  
- Training extras that matter: class‑imbalance handling (weighted loss), extensive spatial augmentations, tiling/inference utilities.

**What we can reuse directly**
- **Architecture skeleton**: U‑Net with a ResNet‑34 (or 50) backbone.  
- **Transfer learning**: initialize encoder with ImageNet weights.  
- **Training recipe**: tiling into 256×256 (or 512×512) patches; augmentations; one‑cycle LR schedule; checkpointing.

**Minimal changes needed**
- **Inputs**: change first conv to accept **2 channels** `[DEM, rainfall]`.  
- **Outputs**:  
  - *Extent*: 1‑channel + sigmoid; weighted BCE/Dice/IoU loss.  
  - *Depth*: 1‑channel **non‑negative** (linear + ReLU) and **MSE/MAE** loss; treat zeros as non‑flood.  
- **Normalization**: per‑patch scaling (e.g., DEM z‑score within AOI; rainfall scaled to design depth) rather than ImageNet stats.

---

## Repo 2 — Rapid Flood Modelling CNN (Depth Surrogate)
**What it has**
- Keras **CNN + dense** network trained to reproduce a 2D model’s **depth map** for a **fixed basin**, using **time‑series boundary inputs** (e.g., inflow hydrographs).  
- Bayesian hyperparameter tuning; regression loss.

**What we can borrow**
- **Surrogate strategy**: generate large synthetic training sets from a hydrodynamic engine (e.g., LISFLOOD‑FP/WCA2D) for many storms; train the NN to emulate them.  
- **Emphasis on tuning**: optimizer (Adam), batch size, layer widths, early stopping.

**What not to port verbatim**
- The **1D time‑series → full‑domain vector** architecture; it doesn’t ingest 2D DEM/rainfall and is tied to a fixed grid. We instead want a **2D fully‑convolutional** model (U‑Net).

---

## Recommended Architecture for DEM + Rainfall
**U‑Net (ResNet‑34 encoder)**
- **Inputs** (2 ch): DEM (m); rainfall (e.g., total mm for design storm or peak‑intensity map). Optional extra channels: slope, curvature, flow‑accumulation, HAND.  
- **Body**: ResNet‑34 encoder (pretrained), decoder with skip connections, multi‑scale context (consider a pyramid or provide a down‑sampled wider‑area DEM as an auxiliary input).  
- **Output**:  
  - Depth (preferred): 1 ch, ReLU; MSE/MAE loss (optionally weighted to emphasize flooded pixels).  
  - Extent (optional auxiliary head): 1 ch, sigmoid; BCE/Dice loss.  
- **Training**: Adam/AdamW; one‑cycle or cosine LR; strong spatial augmentations; tile‑based batches; class‑balance sampling.  
- **Generalization**: train on **diverse storm scenarios** (intensity/duration, hyetographs); leave‑one‑area‑out validation; consider **transfer learning** to new cities with 1–2 local events.

---

## How these repos accelerate us
- **Days → hours to prototype**: Reuse UNOSAT U‑Net codepath; swap input channels; change final head.  
- **Pretrained features**: leverage ResNet weights for faster convergence.  
- **Data strategy**: adopt Repo 2’s surrogate‑training approach: spin up batch simulations to create (DEM + rainfall) → depth labels.  
- **Tunable knobs**: replicate Repo 2’s small HPO loop for LR, weight‑decay, loss weights.

---

## Concrete “lift‑and‑shift” items
- From **Repo 1**:  
  - U‑Net definition + checkpointing/inference scripts.  
  - Weighted loss wrapper; tiling & augmentation utilities.  
- From **Repo 2**:  
  - HPO scaffold (Bayesian/Optuna/Hyperopt); training loop patterns for regression; error distribution reporting.

---

## Minimal POC plan
1. **Data**: DEM (30 m or finer) + Atlas‑14 design storms (total depth + 2–3 hyetographs).  
2. **Labels**: run a fast 2D model to produce depth maps for each storm; polygonize extent if needed.  
3. **Model**: U‑Net (ResNet‑34), 2‑channel input, 1‑channel depth output (ReLU).  
4. **Train**: 256–512 tiles, Adam, 30–60 epochs, weighted MSE (emphasize >0 depth).  
5. **Validate**: hold‑out storms and **unseen neighborhoods**; report RMSE/MAE (depth) + IoU (extent).  
6. **Transfer**: fine‑tune on a new city with one known event; re‑evaluate.

---

## Nice‑to‑have extensions
- **Temporal encoding**: add channels for cumulative rainfall at multiple time slices or a small temporal CNN branch.  
- **Physics‑aware loss**: add a soft **mass‑balance** regularizer (volume in ≈ volume stored + outflow).  
- **Multi‑scale input**: second encoder for a coarser, wider‑area DEM to capture upstream contributions.

---

## Risks & Mitigations
- *Imbalanced data (few flooded pixels)* → weighted losses; oversample flooded tiles.  
- *Overfitting to one city* → cross‑city training; regularization; per‑patch normalization.  
- *Non‑physical speckle* → ReLU output, connected‑component filtering, optional physics penalty.

---

## TL;DR
- **Use Repo 1’s U‑Net with a pretrained ResNet encoder**, adapted to 2‑channel `[DEM, rainfall]`, and predict **depth** (plus extent if desired).  
- **Use Repo 2’s idea of training on physics‑generated labels** and its hyperparameter‑tuning discipline.  
- This combo gives a fast, transferable, and physically‑plausible pluvial flood mapper for city‑scale risk scenarios.
