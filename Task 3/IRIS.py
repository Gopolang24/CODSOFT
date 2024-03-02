import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import load_iris
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from mlxtend.plotting import plot_decision_regions

# Create a folder for visuals
visuals_folder = 'visuals'
os.makedirs(visuals_folder, exist_ok=True)

# Load the Iris dataset
iris = load_iris()
# Create a DataFrame with features and target variable
data = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])

# Display the first few rows of the dataset
print(data.head())
# Display information about the dataset, including data types and null values
print(data.info())
# Display summary statistics of the dataset
print(data.describe())
# Check for missing values in the dataset
print(data.isnull().sum())

# Prepare the data
X = data.drop('target', axis=1)
y = data['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model (K-Nearest Neighbors in this case)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train_scaled, y_train)

# Evaluate the model on the test set
y_pred = model.predict(X_test_scaled)

# Calculate and print accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy)

# Generate and print classification report
classification_rep = classification_report(y_test, y_pred)
print("Classification Report:\n", classification_rep)

# Generate and print confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Visualize decision boundaries
plot_decision_regions(X_train_scaled, y_train.values.astype(np.int_), clf=model, legend=2, filler_feature_values={2: 0, 3: 0}, scatter_kwargs={'alpha': 0.3})
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('Decision Boundaries of KNN Model')

# Save the decision boundaries plot
decision_boundaries_path = os.path.join(visuals_folder, 'decision_boundaries.png')
plt.savefig(decision_boundaries_path)
plt.close()

# Additional feature engineering: create new ratio features
data['sepal_length_width_ratio'] = data['sepal length (cm)'] / data['sepal width (cm)']
data['petal_length_width_ratio'] = data['petal length (cm)'] / data['petal width (cm)']

# Select features including the newly created columns
X = data[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)', 'petal_length_width_ratio', 'sepal_length_width_ratio']]
# Split the data again after feature engineering
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Pair plot to visualize relationships between features
pair_plot = sns.pairplot(data, hue='target')
pair_plot.fig.suptitle('Pair Plot of Iris Dataset', y=1.02)  # Add a title to the pair plot

# Save the pair plot
pair_plot_path = os.path.join(visuals_folder, 'pair_plot.png')
plt.savefig(pair_plot_path)
plt.close()

# Box plot for each feature based on the target variable
for feature in iris['feature_names']:
    box_plot = sns.boxplot(x='target', y=feature, data=data)
    box_plot.set_title(f'Boxplot of {feature} by Species')

    # Save the box plot
    box_plot_path = os.path.join(visuals_folder, f'box_plot_{feature}.png')
    plt.savefig(box_plot_path)
    plt.close()

# Display the count of each target class
print(data['target'].value_counts())

# Calculate and display the correlation matrix
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')

# Save the correlation matrix plot
correlation_matrix_path = os.path.join(visuals_folder, 'correlation_matrix.png')
plt.savefig(correlation_matrix_path)
plt.close()

# Cross-validate the model
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
print("Cross-Validation Scores:", cv_scores)
print("Mean CV Score:", np.mean(cv_scores))

# Perform grid search for hyperparameter tuning
from sklearn.model_selection import GridSearchCV
param_grid = {'n_neighbors': [3, 5, 7, 9]}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)

# Get the best hyperparameters from grid search
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)
