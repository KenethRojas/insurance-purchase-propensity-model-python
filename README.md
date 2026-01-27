# Insurance Purchase Propensity Model: A Predictive Analysis Using Python

## 📌 Business Context
An insurance company wants to identify which customer base they can offer one of its most important products, vehicle insurance.
Marketing campaigns can be inefficient if they target customers with low purchase intent.

This projects analyze historical customer data to create a predictive model about probability of customers' purchase using Machine Learning with Python.

## 🎯 Objective
- Identify patterns that allows differentiate between buying and non-buying customers.
- Estimate the probability of a customer purchasing vehicle insurance.
- Rank customers according to their likelihood of purchase
- Support the prioritization of sales efforts, focusing on customers with the highest likelihood of purchasing insurance.

## 📊 Dataset
- Source: Insurance Company Dataset - Confidential data; identifiers and some features anonymized and scaled for this repo.
- Records: 30K+
- Features: `Variable1` to `Variable10`: Customer demographic, transaction history, and interaction data (anonymized and scaled).
- Target: `Flag_Vehicular`, variable that indicates whether the customer purchased vehicle insurance. Where 1 indicates yes and 0 indicates no.

## 🛠️ Methodology
- Data cleaning and preprocessing: Validation of nulls and duplicates.
- Exploratory Data Analysis (EDA): Identification of target imbalance (~6% of buyers), which can introduce challenges for the model.
- Feature engineering: Application of logarithmic transformation to reduce asymmetries. Normalization of variables through standardization.
- Selection of a predictive technique: Logistic regression due to its stability and interpretability. Stratified train/test split (70/30). Class weighting to address imbalance.

## 💡 Key Insights
- Customers were segmented into propensity deciles based on the score assigned by the model.
- The area under the curve (AUC) metric was 0.758, which validates the model's discrimination capacity.
- The higher deciles, ergo, the groups with the highest scores, represent a greater proportion of buyers.
- Decile 10 has an approximate purchase rate of 19%, more than three times the overall average (~6%). Decile 9 has 11% rate.

## 🧰 Tools & Technologies
- Python
- pandas, numpy, scikit learn
- PyCharm IDE

## 🚀 Next Steps
- Prioritize commercial efforts initially on the 10th decile and secondarily on the 9th decile.
- Integrate propensity scores into marketing campaigns to prioritize high-potential customers, minimizing marketing spend inefficiency.
- Retrain the model periodically with new data to avoid data drift and keep the model up to date.
- Explore threshold adjustments based on business objectives (conversion vs. reach).
