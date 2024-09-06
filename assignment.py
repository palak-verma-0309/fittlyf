import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import numpy as np
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import classification_report, roc_auc_score

# Step 1: Data Import and Exploratory Analysis
file_path = 'AssignmentData.xlsx'
funnel_df = pd.read_excel(file_path, sheet_name='WorkerFunnel')

# Display the first few rows of the dataframe
print(funnel_df.head())

# Display information about the dataframe
print(funnel_df.info())

# Identify missing values and duplicates
print(funnel_df.isnull().sum())
print(funnel_df.duplicated().sum())

# Handle missing and duplicate values
# Remove duplicates
funnel_df = funnel_df.drop_duplicates()

# Convert columns to numeric where appropriate and handle missing values
numeric_cols = ['Targeted Productivity', 'Overtime', 'No. of Workers', 'Actual Productivity']
funnel_df[numeric_cols] = funnel_df[numeric_cols].apply(pd.to_numeric, errors='coerce')  # Convert to numeric if necessary

# Fill missing values with mean for numeric columns
funnel_df = funnel_df.fillna(funnel_df[numeric_cols].mean())

# Fill missing values in non-numeric columns with the mode
funnel_df['Department'] = funnel_df['Department'].fillna(funnel_df['Department'].mode()[0])

print(funnel_df.isnull().sum())  # Verify that there are no more missing values

# Step 2: PCA
features = ['Targeted Productivity', 'Overtime', 'No. of Workers', 'Actual Productivity']
x = funnel_df[features]
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
pca = PCA(n_components=4)
principal_components = pca.fit_transform(x_scaled)
explained_variance = pca.explained_variance_ratio_

plt.figure(figsize=(8,6))
plt.bar(range(1, len(explained_variance)+1), explained_variance, alpha=0.5, align='center')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.show()

cumulative_variance = pca.explained_variance_ratio_.cumsum()
print("Cumulative variance explained by components: ", cumulative_variance)
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2', 'PC3', 'PC4'])
print(pca_df.head())

# Step 3: Time Series Forecasting with ARIMA
funnel_df['Date'] = pd.to_datetime(funnel_df['Date'])
funnel_df.set_index('Date', inplace=True)
quarterly_productivity = funnel_df['Actual Productivity'].resample('Q').mean()

model = ARIMA(quarterly_productivity, order=(1,1,1))
model_fit = model.fit()
forecast = model_fit.forecast(steps=4)

mse = mean_squared_error(quarterly_productivity, model_fit.fittedvalues)
mape = mean_absolute_percentage_error(quarterly_productivity, model_fit.fittedvalues)
print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Percentage Error: {mape}")

plt.figure(figsize=(10,6))
plt.plot(quarterly_productivity, label='Actual')
plt.plot(forecast, label='Forecast', linestyle='--')
plt.legend()
plt.show()

# Step 4: K-Means Clustering
features = ['Actual Productivity', 'Overtime', 'No. of Workers']
x = funnel_df[features]
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

kmeans = KMeans(n_clusters=3, random_state=0)
clusters = kmeans.fit_predict(x_scaled)
funnel_df['Cluster'] = clusters

sns.scatterplot(x=funnel_df['Actual Productivity'], y=funnel_df['Overtime'], hue=funnel_df['Cluster'], palette='Set1')
plt.title('Clusters of Workers based on Productivity and Overtime')
plt.show()

# Step 5: Anomaly Detection (Credit Card Fraud)
creditcard_df = pd.read_excel(file_path, sheet_name='creditcard')
creditcard_test_df = pd.read_excel(file_path, sheet_name='creditcard_test')

creditcard_df = creditcard_df.dropna()
x = creditcard_df.drop(columns=['Class'])
y = creditcard_df['Class']
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Isolation Forest
iso_forest = IsolationForest(contamination=0.001)
iso_forest.fit(x_scaled)
y_pred_iso = iso_forest.predict(x_scaled)
y_pred_iso = [1 if x == -1 else 0 for x in y_pred_iso]

# Local Outlier Factor
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.001)
y_pred_lof = lof.fit_predict(x_scaled)
y_pred_lof = [1 if x == -1 else 0 for x in y_pred_lof]

# Evaluate models
print("Isolation Forest Classification Report:")
print(classification_report(y, y_pred_iso))
print(f"ROC-AUC Score: {roc_auc_score(y, y_pred_iso)}")

print("Local Outlier Factor Classification Report:")
print(classification_report(y, y_pred_lof))
print(f"ROC-AUC Score: {roc_auc_score(y, y_pred_lof)}")

