# Diabetes-ML-Model


## Overview

This project leverages machine learning to predict the likelihood of diabetes based on health metrics such as glucose levels, BMI, age, and more. By exploring different machine learning models, feature engineering techniques, and data balancing strategies, this project aims to create interpretable and effective models for early diabetes detection. Early identification of at-risk individuals can significantly improve intervention outcomes, making this an impactful application of machine learning in healthcare.

---

## Features and Dataset

The dataset used for this project is the [Diabetes Dataset](https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset?resource=download), sourced from Kaggle. It includes the following features:

- **Glucose**: Blood sugar levels, a primary indicator of diabetes.
- **BMI (Body Mass Index)**: A measure of body fat based on weight and height.
- **Age**: The age of the individual, which often correlates with diabetes risk.
- **Pregnancies**: Number of pregnancies, a potential indicator of gestational diabetes.
- **Blood Pressure**: Measures of systolic blood pressure.
- **Skin Thickness and Insulin**: Indicators of metabolic processes and potential insulin resistance.
- **Diabetes Pedigree Function**: Represents genetic predisposition to diabetes.

---

## Machine Learning Pipeline

### 1. **Data Preprocessing**
   - Replaced missing or zero values with mean/median values depending on the feature distribution.
   - Applied SMOTE (Synthetic Minority Oversampling Technique) to handle class imbalance.

### 2. **Feature Engineering**
   - Analyzed feature importance using Random Forest.
   - Reduced the feature set to focus on the most significant predictors, simplifying the models and improving interpretability.

### 3. **Model Training and Evaluation**
   - **Models Tested**:
     - Random Forest with all features.
     - Random Forest with reduced features.
     - Tuned Random Forest with hyperparameter optimization.
     - XGBoost.
   - **Metrics Used**:
     - Accuracy, ROC-AUC score, Precision, Recall, and F1-Score.
   - **Best Model**:
     - The tuned Random Forest model, achieving an accuracy of 78% and a recall of 82% for diabetic cases, was selected for its balance between true positive identification and false positive reduction.

---

## Results

| Model                          | Accuracy | ROC-AUC | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) |
|--------------------------------|----------|---------|----------------------|-------------------|--------------------|
| Random Forest (All Features)   | 77%      | 0.82    | 0.65                 | 0.76              | 0.70               |
| Random Forest (Reduced)        | 78%      | 0.85    | 0.67                 | 0.76              | 0.71               |
| Tuned Random Forest            | 78%      | 0.84    | 0.65                 | 0.82              | 0.73               |
| XGBoost                        | 75%      | 0.83    | 0.61                 | 0.80              | 0.69               |

---

## Next Steps and Areas for Improvement

1. **Feature Expansion**: Include lifestyle and biomarker features for improved predictions.
2. **Advanced Algorithms**: Test deep learning models for capturing complex relationships.
3. **Real-World Validation**: Evaluate on larger and more diverse datasets to ensure robustness.
4. **Interactive Deployment**: Develop a web application or API for easy accessibility.
5. **Explainability**: Integrate explainable AI techniques to foster trust in model predictions.

---

## Getting Started

### Prerequisites
- Python 3.12.8
- Required Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `imblearn`.

### Installation
Clone this repository:
```
git clone https://github.com/yourusername/Diabetes-ML-Model.git
cd Diabetes-ML-Model
```

### Running the Project
1. Open the Jupyter Notebook file: `diabetes_model.ipynb`.
2. Execute the cells sequentially to preprocess the data, train models, and view results.

---

## Author

**Justin Ho**  
LinkedIn: https://www.linkedin.com/in/justin-ho-4a6157285/   
GitHub: https://github.com/justinh128

---

## Acknowledgments

- **Dataset**: [Diabetes Dataset by Akshay Dattatray Khare on Kaggle](https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset?resource=download).
- Special thanks to the open-source community for providing tools and resources used in this project.
