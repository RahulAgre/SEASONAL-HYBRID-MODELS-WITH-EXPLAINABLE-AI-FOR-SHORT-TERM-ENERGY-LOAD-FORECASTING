# ⚡ SEASONAL-HYBRID-MODELS-WITH-EXPLAINABLE-AI-FOR-SHORT-TERM-ENERGY-LOAD-FORECASTING

This repository contains the complete code, analysis, and final results for the thesis submitted for the **M.Sc. Data Science Programme at Liverpool John Moores University (LJMU)**.

The project validates an **STL + 1D-CNN Hybrid Model** as the optimal predictor for Very Short-Term Forecasting (VSTF) of electricity load. It achieves superior accuracy by decomposing the time series and modeling the complex residual component with a deep learning architecture. A key aspect of this research is the integration of **Explainable AI (XAI)** techniques to ensure the models are not only accurate but also interpretable.

---

## 🏆 Key Performance Results (ROCV Averages)

The models were rigorously evaluated using a 9-fold Rolling Origin Cross-Validation (ROCV). The STL + 1D-CNN achieved an unprecedented sub-1% error rate.

| Model | MAE (kW) | RMSE (kW) | sMAPE (%) | Insight | Rank |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **STL + 1D-CNN** | **498.16** | **686.09** | **0.92** | **Definitive Optimal Model (< 1%)** | **1** |
| STL + XGBoost | 1003.88 | 1432.04 | 1.82 | Strong Tree Model Benchmark | 2 |
| STL + LightGBM | 997.37 | 1438.61 | 1.80 | Statistically identical to XGBoost | 3 |
| STL + Random Forest | 1052.79 | 1571.95 | 1.91 | Least accurate ML model | 4 |
| Seasonal Naive (sNaive) | 2149.96 | 3622.23 | 4.37 | Zero-Skill Baseline | 5 |
| ETS (Classical) | 6419.60 | 7702.59 | 11.03 | Structural Failure (Classical Benchmark) | 6 |

### Visual Comparison
The combined forecast plot clearly illustrates the superior fit of the STL + 1D-CNN model across a 24-hour cycle.

![Comprehensive Forecast Comparison: Optimal vs. Benchmarks (5 Models)](https://raw.githubusercontent.com/RahulAgre/SEASONAL-HYBRID-MODELS-WITH-EXPLAINABLE-AI-FOR-SHORT-TERM-ENERGY-LOAD-FORECASTING/)

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

![1D Convolutional Neural Network architecture diagram](https://raw.githubusercontent.com/RahulAgre/SEASONAL-HYBRID-MODELS-WITH-EXPLAINABLE-AI-FOR-SHORT-TERM-ENERGY-LOAD-FORECASTING/)

### 3. Explainable AI (XAI)
Interpretation techniques, such as Feature Importance and potentially SHAP values, are applied to both the tree-based models and the CNN to explain feature influence and model predictions, enhancing trust and practical usability in a production environment.

---

## 📂 Repository Structure

| File/Folder | Description |
| :--- | :--- |
| `notebooks/` | **Primary content.** Contains the main Jupyter Notebooks (e.g., `Forecasting_Pipeline.ipynb`) with the end-to-end code for data processing, STL, ROCV, model training, and plotting. |
| `data/` | Placeholder for the raw and processed time series data. |
| `models/` | Stores the final trained model weights (e.g., `cnn_weights.h5`). |
| `XAI_Analysis/` | Contains specific notebooks or scripts for generating feature importance and interpretability plots. |
| `images/` | Stores all visual outputs, including forecast plots and architecture diagrams. **(Note: Update image links once uploaded to this directory)** |
| `requirements.txt` | Lists all necessary Python dependencies (TensorFlow, XGBoost, LightGBM, Pandas, etc.). |

---

## 🎓 Author & Thesis Details

* **Author:** Rahul Bhikaji Agre
* **Submission Date:** December 2025
* **Thesis Title:** SEASONAL HYBRID MODELS WITH EXPLAINABLE AI FOR SHORT-TERM ENERGY LOAD FORECASTING
* **University:** Liverpool John Moores University (LJMU)
* **Supervisor:** Dr. Anukriti Bansal


For any questions or collaboration inquiries, please contact the repository author.
