# 🌱 Seeing the Unseen, A Novel Approach to Extract Latent Plant Root Traits from Digital Images

*A novel approach to extract latent plant root traits from digital images through advanced machine learning and a custom algorithm*

## 🎯 Project Overview

ART (Algorithmic Root Trait) is designed to uncover hidden root traits using state-of-the-art computer vision and machine learning and a custom algorithm. This project implements a novel methodology that bridges conventional visually-derived traits with autonomous computational analyses, enabling the discovery of previously undetected plant  root characteristics.

### Core Capabilities
- Extraction of 27 algorithmic root traits using multiple unsupervised ML algorithms
- Integration with traditional root trait analysis
- Automated dense root cluster detection and spatial analysis
- Advanced statistical validation framework
- Cross-environment (glasshouse/field) compatibility


## 🔑 Keywords
`root-phenotyping` `machine-learning` `computer-vision` `plant-science` `image-analysis` `agriculture` `phenomics`  `root-traits` `drought-tolerance` `plant-breeding` `Algorithmic Root Traits (ART)`  `wheat` `latent trait` `root` `Image analysis`

## 🏗 System Architecture

```
ART/
├── scripts/
│   ├── Python/
│   │   ├── Algorithms_for_ART_Generation/
│   │   │   └── SLIC.py                    # Superpixel segmentation
│   │   ├── Classification/
│   │   │   ├── CatBoost.py                # Gradient boosting implementation
│   │   │   ├── RandomForest.py            # RF classifier
│   │   │   └── Classification_Algorithms.py# Core classification engine
│   │   └── Rests/
│   │       ├── Feature_Importance_Plot.py  # Feature analysis tools
│   │       ├── Heatmap_PCA_MDS.py         # Dimensionality reduction
│   │       ├── Image_Processing.py         # Image preprocessing pipeline
│   │       └── Performance_Comparison.py   # Model evaluation metrics  
├── docker/
│   ├── Dockerfile                         # Main environment config
│   ├── Dockerfile.hdbscan                 # Specialized clustering container
│   └── docker-compose.yml                 # Container orchestration
└── requirements/
    ├── requirements.txt                   # Python dependencies
    └── r_requirements.txt                 # R statistical packages
```

## 🔧 Requirements

### Hardware
- RAM: 16GB minimum, 32GB recommended
- CPU: 4+ cores recommended
- GPU: Optional, but recommended for large-scale analysis
- Storage: 20GB minimum for base installation

### Software
- Docker Desktop 24.0+
- Git
- WSL2 (for Windows users)
- CUDA drivers (optional, for GPU support)



## 🚀 Technical Stack

### Core Analysis Pipeline
- **Image Processing**: 
  - OpenCV (image preprocessing)
  - RootPainter (segmentation)
  - Custom algorithms for trait extraction
- **Machine Learning Framework**: 
  - Unsupervised Clustering:
    - DBSCAN
    - HDBSCAN
    - K-means
    - Gaussian Mixture Models
    - Mean-shift
    - OPTICS
    - Fuzzy C-means
    - SLIC
  - Classification:
    - Random Forest
    - CatBoost
    - XGBoost
    - Support Vector Machines
- **Statistical Analysis**: 
  - R tidyverse ecosystem
  - scikit-learn
  - Custom statistical validation tools

### Infrastructure
- **Containerization**: Docker 24.0+
- **Primary Environment**: Python 3.11
- **Statistical Environment**: R 4.4.2
- **Specialized Clustering**: Python 3.9 (HDBSCAN-optimized)

## 📊 Performance Metrics

| Algorithm Type | Metric | Score |
|---------------|---------|--------|
| ART Classification | Accuracy | 0.92 |
| ART Classification | ROC AUC | 0.97 |
| ART Classification | Specificity | 0.92 |
| Combined TRT+ART | Accuracy | 0.97 |
| Combined TRT+ART | ROC AUC | 0.97 |
| Combined TRT+ART | Specificity | 0.98 |

## 🛠 Installation & Setup

1. **Clone Repository**
   ```bash
   git clone https://github.com/shoaibms/ART.git
   cd ART
   ```

2. **Build Environments**
   ```bash
   docker-compose build
   ```

3. **Launch Services**
   ```bash
   docker-compose up -d
   ```

4. **Verify Installation**
   ```bash
   # Main container validation
   docker-compose exec main bash -c "python3 test_main.py"
   
   # HDBSCAN validation
   docker-compose exec hdbscan bash -c "python3 test_hdbscan.py"
   ```

## 🔬 Validation Framework

The system implements a comprehensive validation framework:
- 10-fold cross-validation
- Independent validation sets for glasshouse and field data
- Bootstrapped augmentation for improved generalization
- Multiple performance metrics (accuracy, precision, ROC AUC)
- Environment-specific validation protocols

## 📈 Data Processing Pipeline

1. **Image Acquisition**
   - Glasshouse rhizotron imaging
   - Field minirhizotron scanning
   - QR code tracking system

2. **Preprocessing**
   - Automated orientation correction
   - Region of interest extraction
   - Resolution standardization

3. **Trait Extraction**
   - Traditional trait calculation
   - Algorithmic trait computation
   - Spatial coordinate analysis

4. **Analysis**
   - Statistical validation
   - Performance evaluation
   - Cross-environment testing

## 🌟 Related Projects

- [RootPainter](https://github.com/Abe404/root_painter)
- [PlantCV](https://github.com/danforthcenter/plantcv)
- [RhizoVision Explorer](https://github.com/rootphenomics/RhizoVisionExplorer)


🔎 FAQ
<details>
<summary>What image formats are supported?</summary>
ART supports common image formats including JPEG, PNG, and TIFF.
</details>
<details>
<summary>Can I use ART without Docker?</summary>
Yes, but Docker is recommended for consistent environments.
</details>
<details>
<summary>Is GPU support required?</summary>
No, but GPU acceleration can significantly improve processing speed.
</details>


## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## 📚 Citation

If you use ART in your research, please cite:

```
yet to come
```

## 📜 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---
*Developed at Agriculture Victoria and La Trobe University for advanced plant phenotyping research*