import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# Create a folder for visuals
if not os.path.exists('visuals'):
    os.makedirs('visuals')

# Load your dataset 
sales_ds = pd.read_csv('advertising.csv')

# Explore the data
print('First few rows:', sales_ds.head())
print('\nData Information:')
print(sales_ds.info())
print('\nDescriptive Statistics:')
print(sales_ds.describe())

# Check for missing values
print('\nMissing Values:')
print(sales_ds.isnull().sum())

# Check for categorical variables
categorical_cols = sales_ds.select_dtypes(include=['object']).columns
print("\nCategorical Columns:", categorical_cols)

# Visualize the data
sns.pairplot(sales_ds)
plt.savefig('visuals/pairplot.png')  # Save the pairplot visualization
plt.close()

# Check for outliers using boxplots
plt.figure(figsize=(12, 6))

# Boxplot for 'TV' variable
plt.subplot(1, 2, 1)
sns.boxplot(x=sales_ds['TV'])
plt.title('Boxplot for TV')
plt.savefig('visuals/boxplot_tv.png')  # Save the boxplot for TV

# Boxplot for 'Sales' variable
plt.subplot(1, 2, 2)
sns.boxplot(x=sales_ds['Sales'])
plt.title('Boxplot for Sales')
plt.savefig('visuals/boxplot_sales.png')  # Save the boxplot for Sales

plt.close()

# Calculate the correlation matrix and create a heatmap
correlation_matrix = sales_ds.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap')
plt.savefig('visuals/correlation_heatmap.png')  # Save the correlation heatmap
plt.close()

# Set the style of seaborn
sns.set(style="whitegrid")

# Create a pair plot with a different color palette ('husl')
pair_plot = sns.pairplot(sales_ds, kind='scatter', aspect=1, height=4, palette='husl', hue='Sales')

# Set the title
pair_plot.fig.suptitle("Pair Plot for Sales Prediction", y=1.02)

# Show the plot
plt.savefig('visuals/pairplot_husl.png')  # Save the pairplot with husl palette
plt.close()

# Separate the feature and target variables
x = sales_ds['TV']
y = sales_ds['Sales']

# Separate the dataset into training and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, test_size=0.3, random_state=120)

# Initialize the Linear Regression model using scikit-learn
model = LinearRegression()

# Train the model using the training data
model.fit(x_train.values.reshape(-1, 1), y_train)

# Print the coefficients and intercept from scikit-learn model
print(f"\nScikit-learn Coefficient: {model.coef_[0]}")
print(f"Scikit-learn Intercept: {model.intercept_}")

# Make predictions on the test data
y_pred = model.predict(x_test.values.reshape(-1, 1))

# Evaluate the model using scikit-learn metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'\nScikit-learn Mean Squared Error: {mse}')
print(f'Scikit-learn R-squared: {r2}')

# Perform the linear regression using statsmodels.api
# Add constant to the independent variable
x_train_sm = sm.add_constant(x_train)

# Fit the regression model
lr = sm.OLS(y_train, x_train_sm).fit()

# Print the parameters
print('\nStatsmodels Parameters:')
print(lr.params)

# Print the summary of the regression
print('\nStatsmodels Regression Summary:')
print(lr.summary())

# Scatter plot with Regression Line using seaborn
plt.figure(figsize=(10, 6))
sns.scatterplot(x=x_train.values.flatten(), y=y_train, color='blue', label='Scatter Plot')
sns.lineplot(x=x_train.values.flatten(), y=6.57 + 0.058 * x_train.values.flatten(), color='red', label='Regression Line')
plt.xlabel('TV')
plt.ylabel('Sales')
plt.legend()
plt.title('Scatter Plot with Regression Line (Seaborn)')
plt.savefig('visuals/scatter_regression.png')  # Save the scatter plot with regression line
plt.close()

# Residual analysis using seaborn
residuals = y_test - y_pred
sns.residplot(x=y_pred, y=residuals, lowess=True, color="g")
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.savefig('visuals/residual_plot.png')  # Save the residual plot
plt.close()

# Perform 5-fold cross-validation
cross_val_scores = cross_val_score(model, x.values.reshape(-1, 1), y, cv=5)
print('\nCross-Validation Scores:')
print(cross_val_scores)
print(f'Mean Cross-Validation Score: {np.mean(cross_val_scores)}')
