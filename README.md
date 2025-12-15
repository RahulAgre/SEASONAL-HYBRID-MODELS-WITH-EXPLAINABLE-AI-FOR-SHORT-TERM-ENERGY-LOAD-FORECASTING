# ⚡ SEASONAL-HYBRID-MODELS-WITH-EXPLAINABLE-AI-FOR-SHORT-TERM-ENERGY-LOAD-FORECASTING

This repository contains the complete code, analysis, and final results for the thesis submitted for the **M.Sc. Data Science Programme at Liverpool John Moores University (LJMU)**.

The project validates an **STL + 1D-CNN Hybrid Model** as the optimal predictor for Very Short-Term Forecasting (VSTF) of electricity load. It achieves superior accuracy by decomposing the time series and modeling the complex residual component with a deep learning architecture. A key aspect of this research is the integration of **Explainable AI (XAI)** techniques to ensure the models are not only accurate but also interpretable.

---

## 🏆 Key Performance Results (ROCV Averages)

The models were rigorously evaluated using a 9-fold Rolling Origin Cross-Validation (ROCV). The STL + 1D-CNN achieved an unprecedented sub-1% error rate.

| Model | MAE (kW) | RMSE (kW) | sMAPE (%) | Insight |
| :--- | :--- | :--- | :--- | :--- | 
| **STL + 1D-CNN** | **498.16** | **686.09** | **0.92** | **Definitive Optimal Model (< 1%)** | 
| STL + XGBoost | 1003.88 | 1432.04 | 1.82 | Strong Tree Model Benchmark |
| STL + LightGBM | 997.37 | 1438.61 | 1.80 | Statistically identical to XGBoost |
| STL + Random Forest | 1052.79 | 1571.95 | 1.91 | Least accurate ML model |
| Seasonal Naive (sNaive) | 2149.96 | 3622.23 | 4.37 | Zero-Skill Baseline |
| ETS (Classical) | 6419.60 | 7702.59 | 11.03 | Structural Failure (Classical Benchmark) |

### Visual Comparison
The combined forecast plot clearly illustrates the superior fit of the STL + 1D-CNN model across a 24-hour cycle.

![Comprehensive Forecast Comparison: Optimal vs. Benchmarks (5 Models)](https://github.com/RahulAgre/SEASONAL-HYBRID-MODELS-WITH-EXPLAINABLE-AI-FOR-SHORT-TERM-ENERGY-LOAD-FORECASTING/blob/main/reference%20plots/all_model_comparison.png)

---

## ⚙️ Methodology and Architecture

### 1. STL Decomposition
The raw load series ($Y_t$) is first separated into its predictable components (Trend $T_t$ and Seasonal $S_t$) and the unpredictable **Residual ($R_t$)** using the STL algorithm.

$$Y_t = T_t + S_t + R_t$$

### 2. Residual Modeling (1D-CNN)
The complex, non-linear Residual component is forecasted using the following customized 1D-CNN architecture:

| Layer Type | Parameters | Output Function | Key Role |
| :--- | :--- | :--- | :--- |
| **Input** | Sequence Length: **672** | N/A | The scaled 1-week residual time series. |
| **Conv1D (1)** | filters=64, kernel=3 | ReLU | Feature extraction for pattern recognition. |
| **MaxPooling1D (1)** | pool_size=2 | N/A | Sequence downsampling and noise reduction. |
| **Conv1D (2)** | filters=32, kernel=3 | ReLU | Secondary feature extraction. |
| **MaxPooling1D (2)** | pool_size=2 | N/A | Further compression of the feature map. |
| **Flatten** | | N/A | Prepares data for the dense classification head. |
| **Dense (Output)** | units=1 | Linear | Final regression output: the predicted residual value. |

![1D Convolutional Neural Network architecture diagram](https://github.com/RahulAgre/SEASONAL-HYBRID-MODELS-WITH-EXPLAINABLE-AI-FOR-SHORT-TERM-ENERGY-LOAD-FORECASTING/blob/main/1d-cnn_arch.png)

### 3. Explainable AI (XAI)
Interpretation techniques, such as Feature Importance and potentially SHAP values, are applied to both the tree-based models and the CNN to explain feature influence and model predictions, enhancing trust and practical usability in a production environment.

![1D-CNN SHAP Temporal Importance diagram](https://github.com/RahulAgre/SEASONAL-HYBRID-MODELS-WITH-EXPLAINABLE-AI-FOR-SHORT-TERM-ENERGY-LOAD-FORECASTING/blob/main/cnn_shap_temporal_importance.png)
![STL+LightGBM Feature Importance diagram](https://github.com/RahulAgre/SEASONAL-HYBRID-MODELS-WITH-EXPLAINABLE-AI-FOR-SHORT-TERM-ENERGY-LOAD-FORECASTING/blob/main/lgbm_feature_importance.png)
![STL+XGBoost Feature Importance diagram](https://github.com/RahulAgre/SEASONAL-HYBRID-MODELS-WITH-EXPLAINABLE-AI-FOR-SHORT-TERM-ENERGY-LOAD-FORECASTING/blob/main/xgboost_feature_importance.png)
![STL+Random Forest Feature Importance diagram](https://github.com/RahulAgre/SEASONAL-HYBRID-MODELS-WITH-EXPLAINABLE-AI-FOR-SHORT-TERM-ENERGY-LOAD-FORECASTING/blob/main/rf_feature_importance.png)


---

## 📂 Repository Structure

| File/Folder | Description |
| :--- | :--- |
| **`LJMU_Research.ipynb`** | **Primary content.** The single Jupyter Notebook containing the end-to-end code for data processing, $\text{STL}$, $\text{ROCV}$, model training, and plotting. |
| `data/` | Placeholder for data. The dataset is **not included** in the repository due to file size constraints. |
| | **Data Source:** [Open Power System Data - 15min Single Index](https://data.open-power-system-data.org/time_series/2020-10-06/time_series_15min_singleindex.csv) |
| `reference plots/` | Contains the final visual outputs used in the thesis, including the forecast comparison plots and architecture diagrams. |
| `requirements.txt` | Lists all necessary Python dependencies. |

---

## 🚀 Getting Started

### Prerequisites (Key Libraries)

You will need the following libraries, as used in the analysis:

* `tensorflow` (with GPU support highly recommended)
* `xgboost`
* `lightgbm`
* `scikit-learn`
* `statsmodels`
* `pandas`, `numpy`, `matplotlib`

### Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/RahulAgre/SEASONAL-HYBRID-MODELS-WITH-EXPLAINABLE-AI-FOR-SHORT-TERM-ENERGY-LOAD-FORECASTING.git
cd SEASONAL-HYBRID-MODELS-WITH-EXPLAINABLE-AI-FOR-SHORT-TERM-ENERGY-LOAD-FORECASTING
# Assuming you will create and populate requirements.txt
pip install -r requirements.txt

---
```

### 🎓 Author & Thesis Details

* **Author:** Rahul Bhikaji Agre
* **Submission Date:** December 2025
* **Thesis Title:** SEASONAL HYBRID MODELS WITH EXPLAINABLE AI FOR SHORT-TERM ENERGY LOAD FORECASTING
* **University:** Liverpool John Moores University (LJMU)
* **Supervisor:** Dr. Anukriti Bansal


For any questions or collaboration inquiries, please contact the repository author.
