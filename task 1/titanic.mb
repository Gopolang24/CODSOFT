#Titanic Dataset Analysis Summary
#Overview
The analysis focuses on the Titanic dataset, exploring various aspects such as data statistics, missing values, data preprocessing, and building a predictive model for survival prediction. The dataset contains information about passengers, including features like age, gender, class, and survival status.

#Data Exploration
###First Few Rows
The dataset begins with a glimpse of the first few rows, showcasing features like PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, and Embarked.

##Dataset Information
The dataset contains 891 entries with information about 12 columns. Data types include integers, floats, and objects. Key observations include missing values in the 'Age,' 'Cabin,' and 'Embarked' columns.

##Descriptive Statistics
Descriptive statistics reveal essential information about the dataset, including the count, mean, standard deviation, minimum, 25th percentile, median (50th percentile), 75th percentile, and maximum values for numerical features.

##Data Cleaning
###Handling Missing Values
The analysis addresses missing values in the dataset. Columns 'Cabin' and 'Name' were dropped due to a high number of missing values. The 'Age' column was imputed using the mean, and remaining missing values were handled appropriately.

##Exploratory Data Visualization
Visualizations explore the distribution of survivors, the impact of siblings/spouses on survival, gender vs. survival, age distribution, and class vs. survival.

##Feature Engineering
###One-Hot Encoding
One-hot encoding was applied to categorical features ('Sex' and 'Embarked') to transform them into a numerical format for model training.

##Data Scaling
Numerical features were scaled using StandardScaler to ensure uniformity in scale.

##Model Building and Evaluation
###Logistic Regression
A Logistic Regression model was chosen for survival prediction. The dataset was split into training and testing sets. The model achieved an accuracy of 81%, with a confusion matrix and classification report providing detailed performance metrics.

##Model Evaluation Results
Accuracy: 81%
Confusion Matrix:
[[90 15]
 [19 55]]
##Classification Report:

            precision    recall  f1-score   support

         0       0.83      0.86      0.84       105
         1       0.79      0.74      0.76        74

  accuracy                           0.81       179
 macro avg       0.81      0.80      0.80       179
 weighted avg    0.81      0.81      0.81       179
##Explanation of the Classification Report
Precision: Precision measures the accuracy of the positive predictions. In this context, it tells us the accuracy of predicting survival (1). A precision of 0.79 for survival indicates that 79% of the predicted survivals were correct.

Recall: Recall, also known as sensitivity or true positive rate, measures the ability of the model to capture all the positive instances. A recall of 0.74 for survival means that the model correctly identified 74% of the actual survivors.

F1-Score: The F1-score is the harmonic mean of precision and recall. It provides a balance between precision and recall. The weighted average F1-score of 0.81 indicates a good overall balance.

Support: Support is the number of actual occurrences of the class in the specified dataset. In this case, the support for survival (1) is 74, indicating the number of actual survivors in the test set.

##Conclusion
The analysis provides insights into the Titanic dataset, emphasizing data preprocessing, exploratory data analysis, feature engineering, and predictive modeling. The Logistic Regression model demonstrates reasonable predictive accuracy, offering a foundation for further refinement and analysis.

##Code
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import pdfkit
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px

#Load data set
titanic_ds = pd.read_csv('Titanic-Dataset.csv')

#View first few rows of the dataset
print('\n FIRST FEW ROWS: \n',titanic_ds.head())  

print('\n TATANIC DATASET INFO: \n', titanic_ds.info())

#Describe  the dataset statissic
print('\n STATISTIC DESCRIPTION:\n', titanic_ds.describe())

#Handle  missing values if any in the dataset
print("\n MISSING VALUES IN DATASET: \n", titanic_ds.isnull().sum()) 

'''
Drop column due to too many missing values as well 
as the name and ticket columns as they wont realy help us make the conclusion we need
'''
droped_titanic_ds = titanic_ds.drop(columns=['Cabin', 'Name',  'Ticket'])
#print("\n DROPPED COLUMN DATA SET: \n", droped_titanic_ds.head())

# Visualize the distribution of survivors/non-survivors
sns.countplot(x='Survived', data=droped_titanic_ds)
plt.title("Distribution of Survivors")

#Visualize the distribution of SibSp
sns.displot(data=droped_titanic_ds, x="SibSp", bins=[0,1,2,3,4,5])
plt.title("Number of siblings / spouses aboard the Titanic")

# Analyze the impact of SibSp on survival
sns.catplot(kind="bar", x="Survived", y="SibSp", data=droped_titanic_ds, sharex=False,sharey=False )
plt.title("Impact of Number of siblings / spouses on survival ")

# Analyze the impact of gender on survival
sns.catplot(x='Sex', hue='Survived', kind='count', data=droped_titanic_ds)
plt.title( "Gender vs Survival")

# Explore the distribution of age
plt.figure(figsize=(10, 6))
plt.title( "Age Distribution")
sns.histplot(droped_titanic_ds['Age'], bins=30, kde=True, color='skyblue')

# Analyze the impact of class on survival
sns.catplot(x='Pclass', hue='Survived', kind='count', data=droped_titanic_ds)
plt.title( "Class vs Survival")

plt.show()

#Split data into training and testing data 
split = StratifiedShuffleSplit(n_splits=1, test_size=.2, random_state=42)
for  train_index, val_index in split.split(droped_titanic_ds, droped_titanic_ds[["Survived", "Pclass", "Sex"]]):
     strat_train_set = droped_titanic_ds.loc[train_index]
     strat_val_set = droped_titanic_ds.loc[val_index]
     
#Check if the split worked
print ("\nTraining Set:\n",strat_train_set.shape)
print ("Validation Set:\n",strat_val_set.shape)

# Analyze the impact of class on survival
sns.catplot(x='Pclass', hue='Survived', kind='count', data=droped_titanic_ds)
plt.title( "Class vs Survival")
plt.show()

# Plotting the distribution of 'Survived' in the training set
plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.title("Training Set - 'Survived' Distribution")
strat_train_set["Survived"].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.xlabel('Survived')
plt.ylabel('Count')

# Plotting the distribution of 'Survived' in the validation set
plt.subplot(1, 2, 2)
plt.title("Validation Set - 'Survived' Distribution")
strat_val_set["Survived"].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.xlabel('Survived')
plt.ylabel('Count')

plt.tight_layout()
plt.show()

print("\n DROPPED COLUMN DATA SET: \n", droped_titanic_ds.head())

#Handle  missing values for dropeded titanic data set
print("\n MISSING VALUES IN DATASET: \n", droped_titanic_ds.isnull().sum())

# Create a heatmap for missing values
plt.figure(figsize=(12, 8))
sns.heatmap(droped_titanic_ds.isnull(), cmap='viridis', cbar=False, yticklabels=False)
plt.title('Missing Values in the Titanic Dataset')
plt.show()

# Create SimpleImputer instance
age_imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
# Impute age missing values with mean
droped_titanic_ds["Age"] = age_imputer.fit_transform(droped_titanic_ds[["Age"]]).ravel()

# Create a heatmap after imputing missing values
plt.figure(figsize=(12, 8))
sns.heatmap(droped_titanic_ds.isnull(), cmap='viridis', cbar=False, yticklabels=False)
plt.title('IMPUTED  Missing Values in the Titanic Dataset')
plt.show()

onehotencode = OneHotEncoder(handle_unknown='ignore')
feature_arry = onehotencode.fit_transform(droped_titanic_ds[["Sex", "Embarked"]]).toarray()
print(feature_arry)

feature_labels = onehotencode.categories_
print(feature_labels)

flattened_labels = np.concatenate(feature_labels).ravel()
print(flattened_labels)

feature_arry_ds = pd.DataFrame(feature_arry, columns=flattened_labels)
print(feature_arry_ds)

final_ds = pd.concat([droped_titanic_ds, feature_arry_ds], axis=1)
print(final_ds)

droped_final_ds = final_ds.drop(columns=['Sex', 'Embarked'])
print(droped_final_ds)

strat_train_set = droped_final_ds
print('train set: \n',strat_train_set)

#Check null values
print(strat_train_set.isnull().sum())

# Extract numerical features for scaling
numerical_features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']

# Separate numerical features from the dataset
numerical_data = strat_train_set[numerical_features]

# Create a StandardScaler instance
scaler = StandardScaler()

# Fit and transform the numerical data using StandardScaler
scaled_numerical_data = scaler.fit_transform(numerical_data)

# Create a DataFrame with the scaled numerical features
scaled_numerical_df = pd.DataFrame(scaled_numerical_data, columns=numerical_features)

# Concatenate the scaled numerical features with the one-hot encoded features
final_scaled_ds = pd.concat([scaled_numerical_df, strat_train_set[['Survived', 'female', 'male', 'C', 'Q', 'S']]], axis=1)

# Display the scaled and processed dataset
print(final_scaled_ds.head())

# Extract features and target variable
X = final_scaled_ds.drop('Survived', axis=1)
y = final_scaled_ds['Survived']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Logistic Regression model
model = LogisticRegression(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Display the results
print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", classification_rep)

final_scaled_ds.to_csv('final_dataset.csv', index=False)
