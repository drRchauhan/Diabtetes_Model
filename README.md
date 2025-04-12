# 🧠 Diabetes Risk Prediction Tool using Machine Learning

**Fighting Diabetes with Data-Driven Precision**

This project presents a comprehensive machine learning pipeline for classifying diabetes risk using real-world health data. We developed and evaluated four models—**XGBoost, AdaBoost, Random Forest, and a Decision Tree**—on the UCI-BRFSS dataset. The key focus is not just accuracy, but **detecting prediabetes**, a critical intervention window often missed due to class imbalance.

![Model Comparison](plots/model_comparison_all4.png)

---
## 🚀 Key Highlights

- ✅ Ensemble learning with **XGBoost, AdaBoost, Random Forest**
- ⚖️ Addressed extreme **class imbalance** (1.83% prediabetes) using **SMOTE**
- 📈 Evaluated with metrics: **Accuracy, Recall, AUC, Log Loss, MAE, F1**
- 🔍 Performed **feature importance analysis** across models
- 🔬 Trained on 253,000+ entries from UCI’s BRFSS dataset

---

## 📊 Dataset

- **Source**: UCI Behavioral Risk Factor Surveillance System (BRFSS)
- **Classes**:
  - `0`: No diabetes (84.24%)
  - `1`: Prediabetes (1.83%)
  - `2`: Diabetes (13.93%)

- **Features**: 22 total, including BMI, general health, age, blood pressure, physical activity, alcohol consumption, and income.

---

## 🧪 Models Compared

| Model               | Accuracy | Recall | AUC  | F1 Score | Log Loss |
|--------------------|----------|--------|------|----------|-----------|
| XGBoost            | 71.3%    | **73.3%** | **0.790** | 0.416    | 0.523     |
| AdaBoost           | 72.1%    | 71.0% | 0.788 | 0.415    | 0.624     |
| Random Forest      | **78.6%**| 48.4% | 0.769 | 0.387    | 0.472     |
| Classification Tree| 73.7%    | 44.6% | 0.676 | 0.321    | **4.282** |

---

## 📁 Project Structure

```
📦 diabetes_prediction_tool/
├── diabetes_prediction_notebook.ipynb
├── predict_diabetes.py
├── train_model.py
├── preprocess_data.py
├── evaluate_model.py
├── model_outputs.zip
├── README.md
```

---

## 📌 Key Findings

- SMOTE **significantly improved** detection of prediabetic cases
- **XGBoost** provided the best balance between recall and AUC
- **Random Forest** gave highest accuracy but was biased towards majority class
- Top Predictors: **BMI**, **General Health**, **High Blood Pressure**, **Age**
- **Heavy alcohol consumption** surprisingly important in AdaBoost model

---

## 📷 Sample Visualizations

### 📉 ROC Curves
![ROC](plots/auroc_curve_all_models.png)

### 🔥 Feature Importance
![Importance](plots/feature_importances_all.png)

---

## 🛠 Technologies Used

- **Python 3.10**
- **Scikit-learn**
- **XGBoost**
- **imbalanced-learn**
- **Matplotlib, Seaborn**
- **Google Colab** + **Azure VMs**

---

## 💡 Future Work

- Integrate **SHAP** values for explainability
- Explore **deep learning** methods
- Build **multi-stage prediabetes-focused models**
- Add **longitudinal validation**

---

## 👨‍⚕️ Author

**Dr. Rewant Chauhan**  
MSc Dental Sciences, McGill University  
🔗 [LinkedIn](https://www.linkedin.com/in/dr_rewantchauhan) | ✉️ rewant.chauhan@gmail.com

---

## 🌍 Slogan

> **"Fighting Diabetes with Data-Driven Precision"**
