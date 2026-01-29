#########################################################################################
######## Insurance Purchase Propensity Model: A Predictive Analysis Using Python ########
#########################################################################################

## Import neccesary libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, confusion_matrix, classification_report

## Read dataset
df = pd.read_excel('dataset_insurance_purchase_propensity_model.xlsx')

## Initial EDA
print('\nDataset dimensions:')
print(df.shape)
print('\nQuick view of records:')
print(df.head().to_string()) # to_string to see all columns
print('\nNulls per column:')
print(df.isnull().sum())
print('\nDuplicate records:')
print(df.duplicated().sum())
print('\nUnique values for the customer identifier:')
print(df['cliente'].nunique()) # To validate duplicates

## Target review
print('\nTarget Distribution')
print(df['Vehicular_Flag'].value_counts(normalize=True).mul(100).round(2)) # Imbalance identified

## Variable preparation
x = df.drop(columns=['customer', 'Vehicular_Flag'])
y = df['Vehicular_Flag']

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

## Logarithmic transformation and standard scaling of variables
x_train_log = np.log1p(x_train)
x_test_log = np.log1p(x_test)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train_log)
x_test_scaled = scaler.transform(x_test_log)

## Logistic regression model
logit = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    random_state=42
)

logit.fit(x_train_scaled, y_train)

## Metrics
y_pred_proba = logit.predict_proba(x_test_scaled)[:, 1]
y_pred = logit.predict(x_test_scaled)

auc = roc_auc_score(y_test, y_pred_proba)
print('\nAUC:')
print(auc)

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
ks = max(tpr - fpr)
print('\nKS:')
print(ks)

gini = 2 * auc - 1
print('\nGINI:')
print(gini)

acc = accuracy_score(y_test, y_pred)
print('\nAccuracy (referential only due to imbalance):')
print(acc)

## Confusion matrix and classification report
print('\nConfusion matrix:')
print(confusion_matrix(y_test, y_pred))

print('\nClassification report:')
print(classification_report(y_test, y_pred, digits=4))

## Influential variables
coef_df = pd.DataFrame({
    'Variable': x.columns,
    'Coefficient': logit.coef_[0]
}).sort_values(by='Coefficient', ascending=False)

print('\nVariables that increase or decrease the probability of purchase')
print(coef_df)

## Final customer scores
df_score = pd.DataFrame({
    'score': y_pred_proba,
    'real': y_test.values
})
print('\nCustomer score and actual purchase:')
print(df_score.sort_values('score', ascending=False).head(10))

## Segmentation by deciles
df_score['decile'] = pd.qcut(df_score['score'], 10, labels=False) + 1
print('\nScore deciles:')
print(df_score.groupby('decile')['real'].mean().mul(100).round(2))

## Customers with higher scores have a higher purchase rate
## Prioritize campaigns and efforts for customers in the 10th decile, as they have a purchase rate of 19%, more than 3 times the initial average (6%)

## Secondly, also work with clients in the 9th decile
