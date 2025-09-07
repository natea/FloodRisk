# AI Model for City-Scale Pluvial **Flood Extent** Mapping: Proposed Approach

*(Incorporates lessons from UNOSAT U-Net segmentation and SRKabir simulation-first surrogate modeling; labels from physics-based simulations; designed to generalize to unseen cities/watersheds.)*

---

## 1) Objective & Scope
Build a fast, high-resolution AI model that predicts **binary flood extent** (flooded vs not) for city-scale domains under design-storm rainfall. Train on **simulation-generated** flood maps (starting with **Nashville**, expand to additional cities as simulations are produced). The model must **generalize to unseen watersheds**.

---

## 2) Data & Labeling

### 2.1 Inputs
- **Digital Elevation Model (DEM):** USGS **3DEP** (~10 m). Reproject to a metric CRS; align to project grid.
- **Rainfall:** NOAA **Atlas 14** 24‑hour total rainfall depth for target return periods (e.g., **100‑yr** and **500‑yr**). Represent initially as a **uniform raster** (constant value per tile); optional mild spatial variability later.

**Optional derived inputs (recommended for transferability):**
- **Slope**, **curvature** (plan/profile), **flow accumulation** (log), **HAND** (height above nearest drainage). These encode hydrologic controls beyond raw elevation and improve cross‑city generalization.

### 2.2 Labels (from simulations)
- Run a fast 2D pluvial inundation model (e.g., WCA2D/LISFLOOD‑FP/your solver) on the DEM with saturated/limited drainage assumptions suited to pluvial risk.
- Convert simulated depth → **extent** using a small, conservative threshold (e.g., **≥ 0.05 m**).
- Clean with light morphology: remove speckles (< N connected cells), close 1‑cell gaps.

### 2.3 Gridding & Tiling
- Common grid at ≈ **10 m** resolution; aligned rasters for all channels.
- Create **512×512** (or 256×256) tiles with **~64 px overlap** for training and inference.

---

## 3) Model Architecture (UNet‑ResNet)

### 3.1 Core network
- **UNet** encoder–decoder with skip connections (as in UNOSAT rapid flood segmentation).
- **Encoder:** **ResNet‑34** (or 50) **pretrained** on ImageNet; adapt first conv to **C_in** input channels.
- **Decoder:** standard UNet upsampling + skip fusions; output logits at input resolution.

### 3.2 Inputs/Outputs
- **Inputs:** `[DEM, Rain]` → **C_in = 2**. If adding derived features: **C_in = 2–6**.
- **Output:** 1‑channel **logit** map; apply **sigmoid** at inference to obtain **flood probability**.

### 3.3 Multi‑scale context (crucial for unseen cities)
- Option A: UNet with an **FPN** head for pyramid features.
- Option B: **Dual‑stream**: high‑res tile + a **coarser, wider‑area DEM crop** (2–4× extent, downsampled), fused mid/late decoder.

### 3.4 Uncertainty (optional)
- **MC Dropout** at inference or an auxiliary **confidence head** for calibrated probabilities and threshold selection.

---

## 4) Losses, Class Imbalance & Metrics

### 4.1 Loss
- Start with **BCEWithLogits + Dice** (or **Tversky** α=0.5, β=0.7):  
  `Loss = 0.5 * BCE + 0.5 * Dice`
- If heavy imbalance (few flooded pixels): consider **Focal Loss** (γ=2) or **class weights** (w_flood ≈ 2–4× w_dry).

### 4.2 Metrics
- **IoU (Jaccard)** and **F1** at threshold p=0.5 (tune later).
- **AUCPR** (robust with imbalance), **Brier score** & reliability curves (probability calibration).
- Report **precision/recall** vs threshold; select operating point for desired risk tolerance.

---

## 5) Training Procedure

### 5.1 Sampling
- **Balanced tiling:** ~70% tiles contain ≥ 2–5% flooded pixels; ~30% random tiles (some all‑dry) to maintain specificity.
- **Return‑period mix:** Train jointly on **100‑yr** and **500‑yr** events; optionally include a few **sub‑design negatives** (e.g., 10–25‑yr) to suppress false positives.

### 5.2 Augmentations (hydrology‑safe)
- 90° rotations, flips; small translations.
- **Rainfall domain randomization:** ±10–15% scaling; small spatial gradients/noise only on the rainfall layer.
- Avoid elastic/affine warps that distort flow pathways.

### 5.3 Optimization
- **AdamW**, weight decay 1e‑5; **One‑cycle** or cosine schedule.
- **Freeze→unfreeze:** train decoder/new layers 5–10 epochs, then fine‑tune full net with lower LR.
- **Hyperparams (start):** batch **8** (FP16), epochs **60**, LR **1e‑3** (decoder/new) & **1e‑4** (encoder after unfreeze), **grad clip** (norm 1–5).

### 5.4 Implementation
- PyTorch + `segmentation_models.pytorch` (UNet, `encoder_name='resnet34'`, `in_channels=C_in`, `classes=1`).
- Config via Hydra; logs via Weights & Biases; deterministic seeds; checkpoint top‑k models.

---

## 6) Generalization to Unseen Watersheds

- **Data strategy (SRKabir surrogate philosophy):** generate simulations for **diverse cities** (flat/hilly, coastal/inland). For each city, simulate multiple **hyetograph shapes** for a given 24‑hr depth (front/center/back‑loaded).
- **Feature strategy:** add **slope/HAND/log‑accum** channels; **per‑tile normalization** so absolute elevation offsets don’t leak location.
- **Architecture strategy:** keep **multi‑scale context** (FPN or dual‑stream).
- **Adaptation strategy:** **few‑shot fine‑tune** on 1 simulated event for a new city (2–5 epochs, low LR), or unsupervised **entropy minimization** on unlabeled tiles.

---

## 7) Validation Protocol

### 7.1 Within‑city (Nashville POC)
- Train on 80% tiles; validate on 20% non‑overlapping tiles including at least one **held‑out neighborhood**.
- Report IoU/F1/AUCPR; choose threshold via PR curve (e.g., Youden J or fixed precision target).

### 7.2 Cross‑city (as more sims arrive)
- **Leave‑one‑city‑out (LOCO):** train on N−1 cities, test on the held‑out city; rotate.
- Summarize median/IQR of IoU & F1 across cities; analyze failure modes (e.g., flat floodplains, culverted corridors).

### 7.3 External checks (optional)
- Sanity‑check overlays with **FEMA NFHL** or any claims/high‑water marks available (not for training).

---

## 8) Inference & Deliverables

- **GeoTIFFs:**  
  - `flood_prob.tif` (float32, 0–1)  
  - `flood_extent.tif` (uint8; thresholded & morphology‑cleaned)
- **Vectors:** `flood_extent.gpkg` (polygonized, dissolved).  
- **QGIS styles (QML):** blue fill for extent; probability ramp for `flood_prob.tif`.  
- **Metadata:** CRS, storm RP/depth, tile size/overlap, model commit, threshold, metrics.

---

## 9) Phased Build Plan

**Phase 0 — Data assembly (1–2 days)**  
- 3DEP DEM + Atlas‑14 depths for **Nashville**; run simulator → depth → extent (≥ 0.05 m); tile; derive slope/HAND/log‑accum.

**Phase 1 — Baseline (3–5 days)**  
- UNet‑ResNet34, inputs `[DEM, Rain]`, loss `BCE + Dice`; balanced sampling; within‑city validation; target **IoU ≥ 0.70** on held‑out neighborhoods.

**Phase 2 — Multi‑scale & features (3–5 days)**  
- Add FPN or dual‑stream context; include slope/HAND/log‑accum; tune losses; calibrate threshold.

**Phase 3 — Generalization (1–2 weeks as sims expand)**  
- Add **5–10** diverse cities × (100‑yr & 500‑yr) × **2–3** hyetographs; LOCO evaluation; few‑shot fine‑tune recipe for new cities.

**Phase 4 — Packaging (2–3 days)**  
- Batch inference CLI; Docker image; QGIS styles; reproducible Hydra configs; model card.

---

## 10) Risks & Mitigations
- **Class imbalance (few flooded pixels):** focal/Tversky losses; oversample flooded tiles.  
- **Infrastructure not in DEM (culverts/drains):** expect some local errors; consider adding OSM roads/land‑use later as auxiliary inputs.  
- **Overfitting to one city:** multi‑scale context + physical features + rainfall randomization; expand training cities ASAP.  
- **Threshold sensitivity:** provide `flood_prob.tif`; calibrate operating point via PR; ship reliability curves.

---

## 11) Next Steps
1. Implement the UNet‑ResNet34 baseline with `[DEM, Rain]` inputs and `BCE+Dice` loss on **Nashville**.  
2. Add multi‑scale context and physical feature channels; re‑evaluate within‑city performance and calibration.  
3. Generate simulations for additional cities; move to **LOCO** testing; finalize packaging for QGIS delivery.
