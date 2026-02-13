# 🫘 Dry Bean Classification — ML Assignment 2

## Problem Statement
The goal of this project is to classify dry bean seeds into one of 7 distinct
varieties (BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER, SIRA) using machine
learning classification models. The classification is based on 16 geometric
features extracted from images of the beans, such as area, perimeter,
compactness, eccentricity, and extent.

This is a **multi-class classification** problem.

---

## Dataset Description
- **Source:** UCI Machine Learning Repository
- **URL:** https://archive.ics.uci.edu/dataset/602/dry+bean+dataset
- **Reference:** Koklu, M. and Ozkan, I.A., (2020). Multiclass Classification of
  Dry Beans Using Computer Vision and Machine Learning Techniques.
  Computers and Electronics in Agriculture, 174, 105507.
- **Total Instances:** 13,611
- **Total Features:** 16 (all numeric)
- **Target Classes:** 7 (BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER, SIRA)
- **Missing Values:** None

| Feature | Description |
|---|---|
| Area | Area of the bean zone |
| Perimeter | Perimeter of the bean |
| MajorAxisLength | Length of the major axis |
| MinorAxisLength | Length of the minor axis |
| AspectRation | Ratio of major to minor axis |
| Eccentricity | Eccentricity of the ellipse |
| ConvexArea | Area of the convex hull |
| EquivDiameter | Diameter of a circle with same area |
| Extent | Ratio of pixels in bounding box |
| Solidity | Ratio of area to convex area |
| roundness | 4π×Area / Perimeter² |
| Compactness | Ratio of EquivDiameter to MajorAxisLength |
| ShapeFactor1 | MajorAxisLength / (Area^(1/2)) |
| ShapeFactor2 | MinorAxisLength / (Area^(1/2)) |
| ShapeFactor3 | Area / (MajorAxisLength)^2 |
| ShapeFactor4 | Area / (MajorAxisLength × MinorAxisLength) |

---

## Models Used

### Comparison Table

| ML Model Name | Accuracy (%) | AUC (%) | Precision (%) | Recall (%) | F1 (%) | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 92.14 | 99.48 | 92.22 | 92.14 | 92.16 | 0.905 |
| Decision Tree | 90.97 | 96.9 | 90.95 | 90.97 | 90.93 | 0.8908 |
| KNN | 91.81 | 98.84 | 91.92 | 91.81 | 91.83 | 0.901 |
| Naive Bayes | 89.79 | 99.16 | 90.07 | 89.79 | 89.81 | 0.8773 |
| Random Forest (Ensemble) | 91.92 | 99.38 | 91.95 | 91.92 | 91.92 | 0.9023 |
| XGBoost (Ensemble) | 92.29 | 99.52 | 92.31 | 92.29 | 92.29 | 0.9067 |

---

### Observations

| ML Model Name | Observation |
|---|---|
| Logistic Regression | Logistic Regression performs well as a linear baseline model. It achieves good accuracy given the dataset is not perfectly linearly separable. It is the fastest model to train and provides interpretable coefficients. Its relatively lower performance compared to ensemble models suggests non-linear boundaries exist between bean classes. |
| Decision Tree | Decision Tree captures non-linear patterns well but shows slightly lower accuracy than ensemble methods, indicating some overfitting even with depth constraints. It is fully interpretable and visual. The shallow depth limit (max_depth=15) and minimum leaf size help generalization, but it cannot match the variance-reduction power of ensemble approaches. |
| KNN | KNN performs very competitively on this dataset because bean shape features are continuous and proximity in feature space is meaningful. With distance-weighted voting and the optimal k, it captures local structure well. Its main limitation is high inference time on large datasets since it is a lazy learner that stores all training examples. |
| Naive Bayes | Gaussian Naive Bayes achieves the lowest performance among all models because the feature independence assumption is clearly violated — many geometric features like Area, ConvexArea, and Perimeter are highly correlated. Despite this, it trains extremely fast and provides reasonable class probability estimates. It serves as a useful probabilistic baseline. |
| Random Forest (Ensemble) | Random Forest is one of the top performers, significantly outperforming single decision trees. By averaging 200 decorrelated trees, it reduces variance and avoids overfitting. It handles highly correlated features better than individual trees and provides reliable feature importance scores. Strong across all metrics including MCC. |
| XGBoost (Ensemble) | XGBoost achieves the highest or near-highest scores across all metrics. Its gradient boosting framework sequentially corrects errors from previous trees, making it extremely effective. Regularization parameters prevent overfitting. The high MCC score confirms it is the most balanced and reliable model for the Dry Bean classification task. |

---

## Repository Structure
```
project-folder/
├── app.py                   # Streamlit web application
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── ML_Assignment_2_DryBean.ipynb  # Complete Jupyter Notebook
├── test_data_sample.csv     # Sample test data for Streamlit upload
└── model/
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    ├── xgboost.pkl
    ├── scaler.pkl
    ├── label_encoder.pkl
    ├── feature_cols.pkl
    └── metrics_results.csv
```

## How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit App Features
1. Upload test CSV file
2. Select classification model from dropdown
3. View evaluation metrics
4. View confusion matrix

## Tech Stack
Python 3.10 | scikit-learn | XGBoost | Streamlit | Pandas | Matplotlib | Seaborn
