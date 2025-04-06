# HOUSE-PRICE-PREDICTION

# 🏠 Real Estate Price Prediction using Machine Learning

## 📌 Project Overview
This project aims to build a regression model to predict **house prices per unit area** based on various real estate features such as house age, distance to MRT, number of nearby convenience stores, and geographical location (latitude & longitude).

---

## 📂 Dataset Details
- **Source:** Real estate valuation data set (Taiwan housing data)
- **Total Rows:** 414
- **Total Features (after cleaning):** 6 predictors + 1 target variable

### 📊 Variables Description:

| Variable              | Description |
|-----------------------|-------------|
| `House_age`           | Age of the house (in years) |
| `Distance_to_MRT`     | Distance to the nearest MRT station (in meters) |
| `Convenience_stores`  | Number of convenience stores nearby |
| `Latitude`            | Geographical latitude |
| `Longitude`           | Geographical longitude |
| `Price` (Target)      | House price per unit area (NTD/m²) |

---

## 🎯 Objective
To build a robust regression model that can predict real estate prices based on the available features.

---

## 🛠️ Steps Performed

1. **Data Cleaning**  
   - Removed irrelevant columns: `No`, `Transaction_date`
   - Checked for null values and outliers

2. **Exploratory Data Analysis (EDA)**  
   - Visualized distributions using `sns.displot`
   - Detected outliers using `boxplot`
   - Analyzed skewness and applied `log1p` transformation where needed
   - Studied correlation matrix to identify multicollinearity

3. **Modeling**  
   - Applied `AdaBoostRegressor` with base model as `DecisionTreeRegressor`
   - Performed `train_test_split` (80:20)
   - Evaluated using R² Score, MAE, RMSE

---

## 🤖 Model Used

```python
AdaBoostRegressor(
    base_estimator=DecisionTreeRegressor(max_depth=4),
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)
📈 Model Performance
Metric	Value
R² Score	0.889 ✅
MAE	~3.5
RMSE	~4.6
Interpretation:
Model explains ~89% of the variance in housing prices — indicating a very good fit.

🔍 Insights
Distance to MRT and Latitude had strong influence on price.

Latitude and Distance_to_MRT were highly correlated (r = 0.81), but both were kept for better model performance.

Outliers were handled carefully, and skewness was corrected using log1p.

✅ Conclusion
A strong predictive model for real estate prices was built using ensemble learning. The AdaBoost Regressor provided a solid R² score of 88.9%, demonstrating reliable performance on this dataset.

📌 Tools Used
Python

Pandas, NumPy

Seaborn, Matplotlib

Scikit-learn

🙌 Author
Pragya maurya – Data Science Enthusiast 
