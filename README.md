# 🩺 Patient Outcome Prediction System

A machine learning-based classification system that predicts patient outcomes using electronic healthcare records (EHR). This project demonstrates the complete ML pipeline—from data preprocessing and feature engineering to model optimization and evaluation—using a mix of classical and neural network models.

---

## 📌 Project Overview

This project aims to predict patient outcomes such as recovery or deterioration based on features extracted from hospital datasets. By leveraging classification algorithms and dimensionality reduction techniques, the system helps in early risk detection and healthcare decision-making.

Key highlights:
- Multi-model approach: SVM, XGBoost, MLP
- Hyperparameter optimization using PSO and RandomizedSearchCV
- Dimensionality reduction using PCA
- Accuracy range: **79% – 89%** on unseen data

---

## 🧠 Technologies & Tools Used

- **Languages & Libraries**: Python, NumPy, Pandas
- **Machine Learning**: Scikit-learn, XGBoost
- **Deep Learning**: TensorFlow / Keras
- **Dimensionality Reduction**: PCA
- **Model Tuning**: RandomizedSearchCV, Particle Swarm Optimization (PSO)
- **Visualization**: Matplotlib, Seaborn

---

## 📊 Workflow & Methodology

### 1. 🔄 Data Preprocessing
- Handled **missing values** with imputation techniques.
- **Encoded categorical features** using one-hot encoding.
- Engineered new features, such as **length of hospital stay**.
- Normalized/standardized numerical features where required.

### 2. 📉 Dimensionality Reduction
- Applied **Principal Component Analysis (PCA)** to reduce feature space dimensionality while retaining **95% of the variance**.
- Reduced computational cost and improved model performance.

### 3. 🧪 Model Training & Optimization
- Trained multiple classification models:
  - **SVM** (optimized with Particle Swarm Optimization)
  - **XGBoost** (optimized using `RandomizedSearchCV`)
  - **MLP Neural Network** (with hyperparameter tuning via `RandomizedSearchCV`)

### 4. 🧮 Evaluation Metrics
- Evaluated each model using:
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1-score**
- Validation and test set accuracy ranged from **79% to 89%**, showing strong generalization across models.

---

## 🧪 Results

| Model       | Validation Accuracy | Test Accuracy | Notes                            |
|-------------|---------------------|----------------|----------------------------------|
| SVM (PSO)   | ~84%                | ~82%           | Tuned with Particle Swarm        |
| XGBoost     | ~89%                | ~87%           | Best-performing overall          |
| MLP         | ~83%                | ~79%           | Slightly lower due to overfitting|

---

## 🚀 Getting Started

### Clone the repository:
```bash
git clone https://github.com/yourusername/patient-outcome-prediction.git
cd patient-outcome-prediction
