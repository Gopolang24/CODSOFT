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

# Set up subplots with increased spacing
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(20, 17), gridspec_kw={'hspace': 1, 'wspace': 1})

# Visualize the distribution of survivors/non-survivors
sns.countplot(x='Survived', data=droped_titanic_ds, ax=axes[0, 0])
axes[0, 0].set_title("Distribution of Survivors")

# Visualize the distribution of SibSp
sns.histplot(data=droped_titanic_ds, x="SibSp", bins=[0, 1, 2, 3, 4, 5], ax=axes[0, 2])
axes[0, 2].set_title("Number of siblings / spouses aboard the Titanic")

# Analyze the impact of SibSp on survival
sns.barplot(x="Survived", y="SibSp", data=droped_titanic_ds, ax=axes[0, 3])
axes[0, 3].set_title("Impact of Number of siblings / spouses on survival")

# Analyze the impact of gender on survival
sns.countplot(x='Sex', hue='Survived', data=droped_titanic_ds, ax=axes[2, 0])
axes[2, 0].set_title("Gender vs Survival")

# Explore the distribution of age
sns.histplot(droped_titanic_ds['Age'], bins=30, kde=True, color='skyblue', ax=axes[1, 2])
axes[1, 2].set_title("Age Distribution")

# Analyze the impact of class on survival
sns.barplot(x='Pclass', hue='Survived', data=droped_titanic_ds, ax=axes[2, 3])
axes[2, 3].set_title("Class vs Survival")

# Plotting the distribution of 'Survived' in the training set
strat_train_set["Survived"].value_counts().plot(kind='bar', color=['skyblue', 'salmon'], ax=axes[3, 1])
axes[3, 1].set_title("Training Set - 'Survived' Distribution")
axes[3, 1].set_xlabel('Survived')
axes[3, 1].set_ylabel('Count')

# Plotting the distribution of 'Survived' in the validation set
strat_val_set["Survived"].value_counts().plot(kind='bar', color=['skyblue', 'salmon'], ax=axes[3, 2])
axes[3, 2].set_title("Validation Set - 'Survived' Distribution")
axes[3, 2].set_xlabel('Survived')
axes[3, 2].set_ylabel('Count')

# Create a heatmap for missing values
sns.heatmap(droped_titanic_ds.isnull(), cmap='viridis', cbar=False, yticklabels=False, ax=axes[3, 3])
axes[3, 3].set_title('Missing Values in the Titanic Dataset')

# Adjust layout
plt.tight_layout()
plt.show()