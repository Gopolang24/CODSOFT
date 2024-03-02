import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from mlxtend.plotting import plot_decision_regions

# Create a folder for visuals
visuals_folder = 'visuals'
os.makedirs(visuals_folder, exist_ok=True)

creditcard_df = pd.read_csv('creditcard.csv')

# Check the structure of the data
print(creditcard_df.head())

# Check for missing values in the dataset
missing_values = print(creditcard_df.head().isnull().sum())

#Describe data
print(creditcard_df.describe())

#Check correlation between variables
correlation_matrix = creditcard_df.corr()
plt.figure(figsize=(12, 8))
plt.imshow(correlation_matrix, cmap='hot', interpolation='nearest')
plt.title('Correlation Matrix')
plt.colorbar()
ticklabels = [item.strip("0") for item in correlation_matrix.columns]
plt.xticks(range(len(ticklabels)), ticklabels, rotation=45)
plt.yticks(range(len(ticklabels)), ticklabels)
plt.savefig(os.path.join(visuals_folder, 'correlation_matrix.png'))  # Save the plot
plt.show()

# Print columns with missing values
print("Columns with Missing Values:\n", missing_values)

# Explore the distribution of classes
class_distribution = creditcard_df['Class'].value_counts()
print("Class Distribution:\n", class_distribution.value_counts())

# Plotting class distribution using a pie chart
plt.figure(figsize=(6, 6))
plt.pie(class_distribution, labels=['Genuine', 'Fraudulent'], colors=['green', 'red'], autopct='%1.1f%%', startangle=90)
plt.title('Class Distribution')
plt.savefig(os.path.join(visuals_folder, 'class_distribution_pie_chart.png'))  # Save the plot
plt.show()

# Feature scaling
scaler = StandardScaler()
creditcard_df[['Amount', 'Time']] = scaler.fit_transform(creditcard_df[['Amount', 'Time']])
print(creditcard_df[['Amount', 'Time']].head())
print(creditcard_df[['Amount', 'Time']].describe())
creditcard_df[['Amount', 'Time']].hist(bins=20, figsize=(10, 5))
plt.savefig(os.path.join(visuals_folder, 'amount_time_histogram.png'))  # Save the plot
plt.show()
correlation_matrix = creditcard_df[['Amount', 'Time']].corr()
print(correlation_matrix)

# Separate features and target variable
X = creditcard_df.drop('Class', axis=1)
y = creditcard_df['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train[['Amount', 'Time']] = scaler.fit_transform(X_train[['Amount', 'Time']])
X_test[['Amount', 'Time']] = scaler.transform(X_test[['Amount', 'Time']])

# Train a Logistic Regression model
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)

# Evaluate the Logistic Regression model
lr_predictions = lr_model.predict(X_test)
lr_conf_matrix = confusion_matrix(y_test, lr_predictions)
lr_classification_report = classification_report(y_test, lr_predictions)

print("Logistic Regression Model Evaluation:")
print("Confusion Matrix:\n", lr_conf_matrix)
print("Classification Report:\n", lr_classification_report)

# Handle imbalanced classes by using class_weight='balanced'
rf_model = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
rf_model.fit(X_train, y_train)

# Feature Importance Analysis
feature_importances = pd.DataFrame(rf_model.feature_importances_, index=X_train.columns, columns=['importance']).sort_values('importance', ascending=False)

# Print the feature importance scores
print("Feature Importance Scores:\n", feature_importances)

# Plotting the top N important features
top_n = 10  # You can adjust this value based on your preference
top_features = feature_importances.head(top_n)
plt.figure(figsize=(10, 6))
plt.barh(top_features.index, top_features['importance'], color='skyblue')
plt.xlabel('Importance')
plt.title('Top {} Important Features'.format(top_n))
plt.savefig(os.path.join(visuals_folder, 'top_features_bar_chart.png'))  # Save the plot
plt.show()

# Display correlation heatmap
plt.figure(figsize=(12, 8))
correlation_matrix_all = creditcard_df.corr()
sns.heatmap(correlation_matrix_all[['Class']], annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Analysis')
plt.savefig(os.path.join(visuals_folder, 'class_correlation_heatmap.png'))  # Save the plot
plt.show()

# Identify features with weak correlation
weak_correlation_threshold = 0.05  # You can adjust this threshold based on your preference
weakly_correlated_features = correlation_matrix_all[abs(correlation_matrix_all['Class']) < weak_correlation_threshold].index

print("Features with Weak Correlation:\n", weakly_correlated_features)
