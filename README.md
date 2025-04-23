# ML_Final

This project explores multiple machine learning approaches to predict outcomes from insurance complaint data. The tasks include classification of **Disposition** and **Conclusion**, and regression on **Recovery Amount**.

## 🔍 Models Evaluated

### 📈 XGBoost
- **Disposition Accuracy:** 55%
- **Conclusion Accuracy:** 35%
- **Recovery Regression:** R² Score = **1.10**

### 🐈 CatBoost
- **Disposition Accuracy:** 52%
- **Conclusion Accuracy:** 31%
- **Recovery Regression:** R² Score = **0.42**

### 🔤 Token Embedding Neural Network
- **Disposition Accuracy:** 54.5%
- **Conclusion Accuracy:** 32.5%
- **Recovery Regression:**
  - R² Score = **0.16**
  - MSE = **9.78**

## 📦 Dataset
Insurance complaint records including categorical fields (coverage, reason, etc.) and target fields (disposition, conclusion, recovery).

## 🚧 Work in Progress
- Hyperparameter tuning and feature engineering
- TabTransformer integration
- Performance improvements for recovery prediction

## 📁 Structure
- `xgboost_model.py`: XGBoost implementation
- `catboost_model.py`: CatBoost implementation
- `pytorch_token_embedding.py`: Token embedding neural network
- `data/`: Preprocessed dataset

---

**Author**: Ryan Courtney  

