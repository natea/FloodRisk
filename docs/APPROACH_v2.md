# AI Model for City-Scale Pluvial Flood Risk Assessment: Comprehensive Approach v2.0

*(Integrates UNOSAT U-Net segmentation, SRKabir surrogate modeling, and comprehensive validation with NFIP claims data. Designed for both flood extent mapping and depth prediction with generalization to unseen watersheds.)*

---

## Executive Summary

This document presents a comprehensive AI-driven approach to city-scale pluvial flood mapping that combines:
- **Binary flood extent mapping** using proven U-Net architectures
- **Continuous depth prediction** with physics-informed constraints
- **Multi-scale context processing** for improved generalization
- **Real-world validation** using NFIP insurance claims and historical events
- **Simulation-based training** using LISFLOOD-FP and other 2D models

The approach is designed to predict both flood extent and depth rapidly (orders of magnitude faster than traditional models) while maintaining high accuracy and physical realism.

---

## 1. Objective & Scope

### Primary Objectives
1. **Fast Prediction**: Generate city-scale flood maps in seconds vs. hours for traditional models
2. **Dual Output**: Predict both binary flood extent and continuous depth values
3. **High Resolution**: 10m spatial resolution suitable for property-level risk assessment
4. **Generalization**: Transfer to new cities with minimal additional training data
5. **Real-world Validation**: Validate against insurance claims and historical flood events

### Target Use Cases
- **Insurance Risk Assessment**: Property-level flood risk for underwriting
- **Urban Planning**: Infrastructure vulnerability and mitigation planning  
- **Emergency Management**: Rapid scenario assessment for design storms
- **Climate Adaptation**: Future flood risk under changing precipitation patterns

---

## 2. Data Sources & Preprocessing

### 2.1 Primary Inputs

#### Digital Elevation Model (DEM)
- **Source**: USGS 3D Elevation Program (3DEP) at ~10m resolution
- **Processing**: Hydrologically conditioned (sink filling, flow direction)
- **Coordinate System**: Metric CRS aligned to project grid
- **Coverage**: Complete Nashville metro area as proof-of-concept

#### Rainfall Scenarios
- **Source**: NOAA Atlas 14 precipitation frequency estimates
- **Return Periods**: 2, 5, 10, 25, 50, 100, 500-year events
- **Duration**: 24-hour design storms with multiple hyetograph shapes
- **Representation**: Initially uniform raster, progressing to spatially variable
- **Temporal Dynamics**: Front-loaded, center-loaded, and back-loaded distributions

### 2.2 Derived Features (Enhanced Generalization)

**Essential Terrain Derivatives:**
- **Slope**: Magnitude and direction of terrain gradient
- **Curvature**: Planform, profile, and mean curvature for flow convergence
- **Flow Accumulation**: Log-transformed drainage area (upstream contributing area)
- **HAND**: Height Above Nearest Drainage for relative elevation context
- **Topographic Wetness Index (TWI)**: Slope-area relationship for saturation potential
- **Stream Power Index (SPI)**: Erosive power indicator

**Physics-Informed Features:**
- **Dimensionless ratios**: Slope-rainfall ratios, scaled catchment area
- **Storage capacity indicators**: Depression depth, sink volume
- **Flow path length**: Distance to outlet, time of concentration proxies

### 2.3 Training Labels from Physics Simulations

#### Simulation Framework
- **Primary Model**: LISFLOOD-FP 2D hydrodynamic solver
- **Backup Models**: WCA2D cellular automata, simplified 2D solvers
- **Assumptions**: Saturated soil conditions, limited storm drainage capacity
- **Domain**: Urban areas with surface-topography-dominated flooding

#### Label Generation Process
1. **Depth Simulation**: Run 2D model for each rainfall scenario
2. **Extent Conversion**: Apply conservative threshold (≥0.05m for extent, ≥0.3m for analysis)
3. **Morphological Cleaning**: Remove speckles (<N connected cells), close gaps
4. **Quality Control**: Validate mass balance, check physical plausibility
5. **Multi-threshold Analysis**: Generate extent maps at multiple depth thresholds

### 2.4 Data Organization & Tiling

#### Spatial Tiling Strategy
- **Tile Sizes**: 512×512 primary, 256×256 for memory constraints
- **Overlap**: ~64 pixels (640m) for edge effect mitigation
- **Multi-scale Context**: 
  - High-res: 512×512 at 10m (5.12km coverage)
  - Medium-res: 1024×1024 at 20m (20.48km coverage)  
  - Low-res: 2048×2048 at 40m (81.92km coverage)

#### Balanced Sampling Strategy
- **Flooded tiles**: 70% contain ≥2-5% flooded pixels
- **Non-flooded tiles**: 30% random tiles (maintaining specificity)
- **Return period mixing**: Joint training on multiple return periods
- **Negative examples**: Include sub-design events (10-25yr) to reduce false positives

---

## 3. Model Architecture (Hybrid U-Net Design)

### 3.1 Core Architecture: Enhanced U-Net

#### Encoder (Feature Extraction)
- **Backbone**: ResNet-34/50 pretrained on ImageNet
- **Input Adaptation**: First conv layer modified for C_in channels (4-8 channels)
- **Multi-scale Processing**: Parallel streams for different spatial contexts
- **Feature Pyramid**: FPN head for pyramid features at multiple scales

#### Decoder (Reconstruction)
- **Skip Connections**: Standard U-Net skip connections with attention mechanisms
- **Upsampling**: Transposed convolution + bilinear interpolation
- **Feature Fusion**: Adaptive fusion of multi-scale context information
- **Physics Constraints**: Mass conservation regularization in decoder

### 3.2 Input/Output Specification

#### Input Channels (C_in = 4-8)
1. **DEM** (elevation, normalized locally)
2. **Rainfall** (total depth or intensity)
3. **Slope** (terrain gradient magnitude)
4. **Flow Accumulation** (log-transformed)
5. **HAND** (optional, height above drainage)
6. **Land Use** (optional, imperviousness proxy)

#### Output Options
- **Option A**: Single-channel logits → sigmoid → flood probability (0-1)
- **Option B**: Dual-head → flood extent (binary) + depth (continuous) 
- **Option C**: Multi-threshold → probability maps at multiple depth thresholds

### 3.3 Multi-Scale Context Integration

#### Dual-Stream Architecture
```
High-Resolution Stream (512×512 @ 10m):
├── ResNet Encoder → Feature Maps → Decoder
└── Skip Connections with Attention

Context Stream (1024×1024 @ 20m): 
├── Simplified Encoder → Context Features
└── Fusion → Combined Features → Output

Regional Stream (2048×2048 @ 40m):
├── Lightweight Encoder → Regional Context  
└── Global Features → Attention Weights
```

### 3.4 Physics-Informed Components

#### Mass Conservation Loss
```python
def physics_loss(pred_depth, rainfall, dem):
    # Water volume conservation
    total_input = rainfall.sum()
    total_output = pred_depth.sum() 
    conservation_error = abs(total_input - total_output) / total_input
    return conservation_error
```

#### Flow Consistency Regularization
- **Downhill Flow**: Penalize water accumulation on higher elevations
- **Gradient Alignment**: Encourage flow direction consistency with terrain
- **Volume Preservation**: Maintain water balance across tile boundaries

---

## 4. Training Strategy & Optimization

### 4.1 Loss Function Design

#### Combined Multi-Task Loss
```
Total_Loss = α × Extent_Loss + β × Depth_Loss + γ × Physics_Loss

Where:
- Extent_Loss = 0.5 × BCE + 0.5 × Dice (for binary classification)
- Depth_Loss = MSE + MAE (for continuous regression)  
- Physics_Loss = Mass_Conservation + Flow_Consistency
- α=0.4, β=0.5, γ=0.1 (tunable weights)
```

#### Class Imbalance Handling
- **Focal Loss**: γ=2 for difficult examples emphasis
- **Class Weights**: w_flood ≈ 3-5× w_dry based on pixel distribution
- **Tversky Loss**: α=0.5, β=0.7 for precision/recall balance

### 4.2 Optimization Configuration

#### Training Hyperparameters
- **Optimizer**: AdamW with weight decay 1e-5
- **Learning Rate**: One-cycle schedule, max_lr=1e-2
- **Batch Size**: 8 (FP16), 4 (FP32) depending on GPU memory
- **Epochs**: 60-100 with early stopping
- **Gradient Clipping**: Norm clipping at 1-5

#### Transfer Learning Strategy  
1. **Phase 1**: Freeze encoder, train decoder (5-10 epochs)
2. **Phase 2**: Unfreeze encoder, full network fine-tuning (lower LR)
3. **Phase 3**: Domain adaptation for new cities (few-shot learning)

### 4.3 Data Augmentation (Hydrology-Safe)

#### Geometric Augmentations
- **Rotations**: 90°, 180°, 270° (preserves flow directions)
- **Flips**: Horizontal/vertical (terrain-appropriate)
- **Small translations**: <10% tile size

#### Intensity Augmentations  
- **Rainfall scaling**: ±10-15% to simulate measurement uncertainty
- **DEM noise**: Small elevation perturbations (±0.1m) 
- **Lighting normalization**: Histogram equalization on DEM

---

## 5. Generalization to Unseen Watersheds

### 5.1 Multi-City Training Strategy

#### Diverse Training Dataset  
- **Topographic Diversity**: Flat coastal, hilly inland, mountainous regions
- **Climate Diversity**: Different precipitation regimes and intensities
- **Urban Morphology**: Dense urban, suburban, mixed development patterns
- **Synthetic Augmentation**: DEM warping, slope modification within physical bounds

#### Cross-City Validation Protocol
- **Leave-One-City-Out (LOCO)**: Train on N-1 cities, test on held-out city
- **Performance Metrics**: Median/IQR of IoU, F1, RMSE across test cities
- **Failure Mode Analysis**: Identify systematic errors (flat areas, culverted zones)

### 5.2 Transfer Learning for New Locations

#### Few-Shot Adaptation
- **Data Requirements**: Single simulated event per new city
- **Fine-tuning Protocol**: 2-5 epochs with reduced learning rate (1e-5)
- **Layer Selection**: Fine-tune decoder + final encoder layers only
- **Validation**: Robust performance across different rainfall scenarios

#### Unsupervised Domain Adaptation
- **Entropy Minimization**: Reduce prediction uncertainty on unlabeled tiles
- **Feature Alignment**: Match feature distributions between source and target cities
- **Self-Training**: Use high-confidence predictions as pseudo-labels

---

## 6. Validation Framework

### 6.1 Simulation-Based Validation

#### Within-City Validation (Nashville POC)
- **Spatial Split**: 80% train, 20% validation (non-overlapping tiles)
- **Neighborhood Holdout**: At least one complete neighborhood reserved for testing
- **Metrics**: IoU ≥ 0.70, F1 ≥ 0.75, RMSE < 0.5m for flood depths
- **Threshold Optimization**: ROC/PR curve analysis for optimal operating point

#### Cross-Model Validation
- **LISFLOOD-FP Comparison**: Primary validation against high-fidelity simulations
- **Multi-Model Ensemble**: Compare with WCA2D, other 2D models where available
- **Uncertainty Quantification**: Monte Carlo dropout for prediction confidence

### 6.2 Real-World Validation with NFIP Claims

#### Historical Claims Analysis
- **Dataset**: NFIP flood insurance claims (1978-2021)
- **Pluvial Focus**: 87.1% of claims outside FEMA floodplains (pluvial flooding)
- **Nashville Coverage**: ~3,500 pluvial flood claims in Davidson County
- **Damage Correlation**: Median damage 9.4% of property value

#### Validation Methodologies

**Spatial Correlation Analysis:**
- **Hit Rate**: Fraction of claim locations correctly predicted as flooded
- **False Alarm Rate**: Predicted flooding where no claims occurred
- **Clustering Analysis**: Spatial correlation between predicted hotspots and claim clusters
- **ROC Analysis**: Receiver Operating Characteristic for various depth thresholds

**Event-Specific Validation:**
- **Historical Storm Matching**: Simulate known flood events (e.g., 2010 Nashville flood)
- **Claim Timing**: Match predicted flooding with claim filing dates
- **Damage Severity**: Correlate predicted depths with claim payout amounts
- **Spatial Precision**: Validate property-level flood risk predictions

**Economic Validation:**
- **Damage Functions**: Relate predicted depths to expected damage percentages
- **Portfolio Analysis**: Aggregate risk assessment for insurance portfolios  
- **Cost-Benefit**: Economic validation of mitigation recommendations

### 6.3 External Validation Sources

#### Observational Data
- **High-Water Marks**: Post-event flood elevation surveys
- **Remote Sensing**: Sentinel-1 SAR flood extent mapping (UNOSAT approach)
- **Crowdsourced Reports**: Citizen science flood observations
- **Infrastructure Logs**: Storm drain capacity exceedances

#### Regulatory Compliance
- **FEMA NFHL**: Comparison with National Flood Hazard Layer
- **Local Studies**: Municipal flood studies and drainage master plans
- **Climate Projections**: Consistency with projected future conditions

---

## 7. Performance Metrics & Success Criteria

### 7.1 Accuracy Metrics

#### Spatial Accuracy (Extent Prediction)
- **IoU (Jaccard Index)**: Target ≥ 0.75 for within-city, ≥ 0.65 for cross-city
- **F1 Score**: Target ≥ 0.80 for within-city, ≥ 0.70 for cross-city  
- **Precision/Recall**: Balanced performance, adjustable for risk tolerance
- **Critical Success Index (CSI)**: Meteorological standard for binary prediction

#### Depth Accuracy (Continuous Prediction)  
- **RMSE**: Target < 0.4m for flooded areas
- **MAE**: Target < 0.25m for flooded areas
- **R² Correlation**: Target > 0.7 with ground truth depths
- **Percentage within ±0.3m**: Target > 90% (following Rapid CNN benchmark)

#### Real-World Validation
- **Claims Correlation**: R² > 0.6 with NFIP claim spatial distribution
- **Hit Rate**: >75% of historical claim locations correctly identified  
- **Economic Accuracy**: <25% error in aggregate portfolio risk estimates

### 7.2 Computational Performance

#### Speed Requirements
- **Inference Time**: < 10 seconds for 100 km² area (city-scale)
- **Throughput**: > 100 concurrent predictions (API scalability)
- **Memory Usage**: < 8GB GPU memory per prediction
- **Speedup Factor**: >100× faster than traditional 2D hydraulic models

#### Scalability Metrics
- **Geographic Coverage**: Scalable to metro areas >1000 km²
- **Scenario Processing**: Batch processing of multiple return periods
- **Real-time Capability**: Support for operational flood forecasting workflows

---

## 8. Implementation Roadmap

### Phase 0: Foundation (Days 1-3)
- **Data Assembly**: Nashville DEM + NOAA Atlas 14 rainfall scenarios
- **LISFLOOD-FP Integration**: Simulation pipeline for training data generation
- **Preprocessing Pipeline**: DEM conditioning, feature extraction, tiling
- **Baseline Metrics**: Establish performance benchmarks

### Phase 1: U-Net Baseline (Days 4-7)  
- **Architecture Implementation**: ResNet-34 U-Net with dual outputs
- **Training Pipeline**: Multi-task loss, balanced sampling, basic augmentation
- **Initial Validation**: Within-city performance on Nashville
- **Target**: IoU ≥ 0.70, RMSE < 0.6m

### Phase 2: Multi-Scale Enhancement (Days 8-12)
- **Context Integration**: Multi-scale input processing
- **Physics Constraints**: Mass conservation, flow consistency regularization  
- **Advanced Augmentation**: Hydrology-safe data augmentation
- **Threshold Calibration**: Optimize operating point via PR curves
- **Target**: IoU ≥ 0.75, RMSE < 0.5m

### Phase 3: Real-World Validation (Days 13-17)
- **NFIP Integration**: Claims data processing and spatial analysis
- **Historical Events**: 2010 Nashville flood case study validation
- **Cross-Validation**: Spatial and temporal validation protocols
- **Uncertainty Quantification**: Confidence intervals, reliability curves
- **Target**: Claims correlation R² > 0.6

### Phase 4: Generalization (Weeks 3-4)
- **Multi-City Expansion**: Additional cities for LOCO validation
- **Transfer Learning**: Few-shot adaptation protocols
- **Domain Analysis**: Systematic evaluation of failure modes
- **Production Optimization**: Model compression, inference acceleration

### Phase 5: Deployment (Week 5)
- **API Development**: FastAPI service with batch processing
- **Containerization**: Docker deployment with GPU support
- **Documentation**: API docs, user guides, model cards
- **Quality Assurance**: End-to-end testing, performance validation

---

## 9. Risk Management & Mitigation

### 9.1 Technical Risks

#### Model Performance Risks
- **Class Imbalance**: Few flooded pixels relative to dry areas
  - *Mitigation*: Focal loss, balanced sampling, synthetic augmentation
- **Generalization Failure**: Overfitting to Nashville-specific features
  - *Mitigation*: Multi-scale context, physical features, diverse training data
- **Physics Violation**: Unrealistic flood patterns (water uphill)
  - *Mitigation*: Physics-informed loss terms, post-processing constraints

#### Data Quality Risks
- **DEM Limitations**: Missing infrastructure (culverts, storm drains)
  - *Mitigation*: Expect local errors, consider infrastructure augmentation
- **Simulation Bias**: LISFLOOD-FP model limitations
  - *Mitigation*: Multi-model validation, real-world calibration with claims
- **Rainfall Uncertainty**: Atlas 14 statistical limitations
  - *Mitigation*: Multiple hyetograph shapes, uncertainty propagation

### 9.2 Operational Risks

#### Deployment Challenges
- **Computational Resources**: GPU availability, memory constraints
  - *Mitigation*: Model optimization, cloud scaling, batch processing
- **Data Pipeline**: Real-time DEM updates, rainfall integration
  - *Mitigation*: Automated workflows, data validation, fallback procedures
- **User Expectations**: Overconfidence in model predictions
  - *Mitigation*: Uncertainty communication, confidence intervals, user training

#### Regulatory & Compliance
- **Insurance Applications**: Model approval for rate-making
  - *Mitigation*: Comprehensive validation, actuarial review, regulatory engagement
- **Legal Liability**: Incorrect predictions leading to damages
  - *Mitigation*: Clear disclaimers, appropriate use documentation, insurance coverage

---

## 10. Deliverables & Outputs

### 10.1 Model Outputs

#### Spatial Products
- **Flood Probability Maps**: `flood_prob.tif` (float32, 0-1 probabilities)
- **Binary Extent Maps**: `flood_extent.tif` (uint8, thresholded and cleaned)
- **Depth Prediction Maps**: `flood_depth.tif` (float32, meters)
- **Confidence Maps**: `flood_confidence.tif` (prediction uncertainty)

#### Vector Products
- **Flood Polygons**: `flood_extent.gpkg` (dissolved, attributed with return period)
- **Risk Zones**: Property-level risk classifications (Low/Medium/High/Extreme)
- **Critical Infrastructure**: Roads, buildings at risk by scenario

#### Metadata & Documentation
- **Model Cards**: Performance metrics, limitations, appropriate use
- **Technical Documentation**: Architecture, training procedure, validation results
- **User Guides**: QGIS integration, interpretation guidelines
- **API Documentation**: Endpoint specifications, example workflows

### 10.2 Software Components

#### Core Model
- **PyTorch Implementation**: Trained model weights, inference code
- **ONNX Export**: Cross-platform deployment capability
- **TensorRT Optimization**: GPU-accelerated inference (NVIDIA)
- **Model Versioning**: Git-based version control, MLflow tracking

#### API Service
- **FastAPI Application**: RESTful API with async processing
- **Docker Containers**: Reproducible deployment environment  
- **Kubernetes Manifests**: Scalable cloud deployment
- **Monitoring**: Prometheus metrics, Grafana dashboards

#### Analysis Tools
- **QGIS Plugins**: Custom tools for flood risk analysis
- **Python SDK**: Programmatic access to model predictions
- **Jupyter Notebooks**: Analysis examples, validation workflows
- **Command-Line Tools**: Batch processing utilities

---

## 11. Future Enhancements

### 11.1 Technical Improvements

#### Architecture Evolution
- **Graph Neural Networks**: Explicitly model flow connectivity
- **Attention Mechanisms**: Spatial and temporal attention for improved context
- **Multi-Modal Integration**: Incorporate satellite imagery, social media data
- **Ensemble Methods**: Combine multiple model architectures

#### Data Integration
- **Real-Time Sensors**: Stream gauge, rain gauge integration
- **Infrastructure Modeling**: Storm drain capacity, green infrastructure
- **Climate Projections**: Future rainfall scenarios under climate change
- **Dynamic Land Use**: Incorporate development and land use changes

### 11.2 Application Extensions

#### Broader Geographic Coverage
- **National Scaling**: Extend to all major US metropolitan areas
- **International Applications**: Adapt to different climates and urban forms
- **Rural Applications**: Extend beyond urban areas to rural flood risk

#### Temporal Extensions
- **Real-Time Forecasting**: Integration with weather prediction models
- **Climate Change Scenarios**: Future risk under different emissions pathways
- **Infrastructure Planning**: Long-term risk assessment for capital investments

#### Risk Applications
- **Multi-Hazard Modeling**: Integration with other natural hazards
- **Economic Impact**: Detailed damage and disruption modeling
- **Social Vulnerability**: Integration with demographic and social data

---

## 12. Conclusion

This comprehensive approach combines the best elements of proven flood modeling techniques:

1. **UNOSAT U-Net Architecture**: Leverages successful SAR-based flood segmentation with 0.92 Dice coefficient
2. **SRKabir Surrogate Philosophy**: Uses physics-based simulation data for training with high accuracy (<0.3m error)
3. **Multi-Scale Context**: Captures both local details and regional drainage patterns
4. **Real-World Validation**: Validates against insurance claims and historical events
5. **Production-Ready Design**: Scalable, fast, and suitable for operational deployment

The resulting system will provide unprecedented capability for rapid, accurate flood risk assessment at city scale, enabling better decision-making for insurance, planning, and emergency management applications. By combining cutting-edge AI techniques with solid physical understanding and real-world validation, this approach bridges the gap between research innovation and practical application.

The Nashville proof-of-concept will demonstrate the feasibility and accuracy of the approach, providing a foundation for scaling to additional cities and broader applications in flood risk management and climate adaptation.

---

## References & Citations

*Key literature informing this approach includes:*

- **UNOSAT Methodology**: Nemni et al. (2020), "Fully Convolutional Neural Network for Rapid Flood Segmentation in SAR Imagery" 
- **Surrogate Modeling**: Kabir et al. (2020), "Deep CNN for Rapid Prediction of Fluvial Flood Inundation"
- **Generalization Research**: Cache et al. (2024), Guo et al. (2022), Bartlett et al. (2024)
- **Physics-Informed Methods**: Fraehr et al. (2023), Bentivoglio et al. (2025)
- **Validation Studies**: Nelson-Mercer et al. (2025), Samadi et al. (2024)
- **Multi-Scale Context**: Recent advances in computer vision and geospatial AI

*Each methodology component has been adapted and enhanced for the specific challenges of urban pluvial flood modeling while maintaining scientific rigor and practical applicability.*