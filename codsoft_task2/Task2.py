import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

# Create a visuals folder
visuals_folder = 'visuals'
os.makedirs(visuals_folder, exist_ok=True)

#Load data set
imdb_ds = pd.read_csv('IMDb_Movies_India.csv', encoding='latin1')

#Print the first 5 rows of the dataset
print('First 5 Rows: \n', imdb_ds.head())

#Print dataset info
print('\nDataset Info:\n', imdb_ds.info())

# Calculate the percentage of missing values for each column
missing_percentage = (imdb_ds.isnull().sum() / len(imdb_ds)) * 100

# Create a DataFrame to display missing percentage
missing_df = pd.DataFrame({'Column': imdb_ds.columns, 'Missing Percentage': missing_percentage})

# Sort the DataFrame by missing percentage in descending order
missing_df = missing_df.sort_values(by='Missing Percentage', ascending=False)

# Display the result
print(missing_df)

# Convert data types
imdb_ds['Year'] = pd.to_numeric(imdb_ds['Year'].astype(str).str.extract('(\d+)', expand=False), errors='coerce').astype('Int64')
imdb_ds['Duration'] = pd.to_numeric(imdb_ds['Duration'].str.extract('(\d+)', expand=False), errors='coerce')
imdb_ds['Votes'] = pd.to_numeric(imdb_ds['Votes'], errors='coerce')

# Display updated data types
print('\nUpdated Data Types:\n', imdb_ds.dtypes)

#Describe the dataset
print('\nData Description:\n', imdb_ds.describe())

# Visualize the distribution of movie ratings using a histogram
plt.figure(figsize=(10, 6))
sns.histplot(imdb_ds['Rating'], bins=20, kde=True, color='skyblue')
plt.title('Distribution of Movie Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.savefig(os.path.join(visuals_folder, 'movie_ratings_distribution.png'))
plt.show()

#Check and visualise missing values
missing_data = imdb_ds.isnull().sum()
print('\nMissing Data Counts:\n', missing_data)

#Plotting bar graph to show missing values in each column
plt.bar(missing_data.index, missing_data.values, width=0.5, align='center')  # Adjust the width as needed
plt.xlabel('Features')
plt.ylabel('Number of Missing Values')
plt.title('Missing Value Analysis')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust layout to prevent clipping of rotated labels
plt.savefig(os.path.join(visuals_folder,  'missing_value_analysis.png'))
plt.show()

def handle_director_missing_values(imdb_ds):
    # Handling missing values by filling with a placeholder value or dropping rows
    imdb_ds.fillna({'Director': 'Unknown'}, inplace=True)

    # Check if there are multiple directors
    imdb_ds['MultipleDirectors'] = imdb_ds['Director'].str.contains(',')

    # Display the movies with multiple directors
    movies_with_multiple_directors = imdb_ds[imdb_ds['MultipleDirectors']]
    print("\nMovies with Multiple Directors:\n", movies_with_multiple_directors[['Name', 'Director', 'MultipleDirectors']])

    # Display the count of movies with multiple directors
    print("\nNumber of Movies with Multiple Directors:", movies_with_multiple_directors.shape[0])

    # Group by 'Director' and count the number of movies directed by each director
    director_counts = imdb_ds.groupby('Director')['Name'].count().reset_index()

    # Display directors who have directed multiple movies
    directors_with_multiple_movies = director_counts[director_counts['Name'] > 1]
    print("\nDirectors with Multiple Movies:\n", directors_with_multiple_movies)

    # Display the count of directors with multiple movies
    print("\nNumber of Directors with Multiple Movies:", directors_with_multiple_movies.shape[0])

    return imdb_ds
imdb_ds = handle_director_missing_values(imdb_ds)

class DataFrameImputer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy='mean'):
        self.strategy = strategy
        self.imputer = SimpleImputer(strategy=self.strategy)

    def fit(self, X, y=None):
        self.imputer.fit(X)
        return self

    def transform(self, X):
        return pd.DataFrame(self.imputer.transform(X), columns=X.columns)

def impute_missing_feature(imdb_ds, feature):
    # Drop rows with missing values in the specified feature for training the model
    train_data = imdb_ds.dropna(subset=[feature])

    # Prepare data for training
    features = ['Duration', 'Votes', 'Rating']  # Replace with relevant features
    features.remove(feature)  # Exclude the target feature
    X_train = train_data[features]
    y_train = train_data[feature]

    # Create a pipeline with custom imputer and linear regression
    model = Pipeline([
        ('imputer', DataFrameImputer(strategy='mean')),
        ('linear_regression', LinearRegression())
    ])

    # Train the model
    model.fit(X_train, y_train)

    # Predict missing values
    X_missing = imdb_ds.loc[imdb_ds[feature].isnull(), features]
    X_missing_imputed = pd.DataFrame(model.named_steps['imputer'].transform(X_missing), columns=features)
    imdb_ds.loc[imdb_ds[feature].isnull(), feature] = model.predict(X_missing_imputed)

    return imdb_ds

imdb_ds = impute_missing_feature(imdb_ds, 'Votes')
imdb_ds = impute_missing_feature(imdb_ds, 'Duration')
imdb_ds = impute_missing_feature(imdb_ds, 'Rating')

#Drop year missing values
imdb_ds.dropna(subset=['Year'], inplace=True)
print('Year', imdb_ds['Year'].head())

def handle_genre_missing_values(imdb_ds):
    most_common_genre = imdb_ds['Genre'].mode().iloc[0]
    imdb_ds['Genre'].fillna(most_common_genre, inplace=True)
    return imdb_ds

imdb_ds = handle_genre_missing_values(imdb_ds)

def handle_actor_missing_values(imdb_ds):
    most_common_actor_1 = imdb_ds['Actor 1'].mode().iloc[0]
    most_common_actor_2 = imdb_ds['Actor 2'].mode().iloc[0]
    most_common_actor_3 = imdb_ds['Actor 3'].mode().iloc[0]

    imdb_ds['Actor 1'] = imdb_ds['Actor 1'].fillna(most_common_actor_1)
    imdb_ds['Actor 2'] = imdb_ds['Actor 2'].fillna(most_common_actor_2)
    imdb_ds['Actor 3'] = imdb_ds['Actor 3'].fillna(most_common_actor_3)

    return imdb_ds

imdb_ds = handle_actor_missing_values(imdb_ds)

#Check and visualise missing values
missing_data = imdb_ds.isnull().sum()
print('\nMissing Data Counts:\n', missing_data)

#Plotting bar graph to show missing values in each column
plt.bar(missing_data.index, missing_data.values, width=0.5, align='center')  # Adjust the width as needed
plt.xlabel('Features')
plt.ylabel('Number of Missing Values')
plt.title('Missing Value Analysis')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust layout to prevent clipping of rotated labels
plt.savefig(os.path.join(visuals_folder, 'Number_of_Missing_Values.png'))
plt.show()

def analyze_and_concat(imdb_ds):
    # Count occurrences of each unique genre
    genre_counts = imdb_ds['Genre'].str.split(', ', expand=True).stack().value_counts()
    # Print counts in sorted order by genre name
    sorted_genre_counts = genre_counts.sort_index()
    imdb_ds['Genre_Counts'] = imdb_ds['Genre'].apply(lambda x: sorted_genre_counts[x.split(', ')[0]])

    # Count number of movies per actor
    actors = pd.concat([imdb_ds['Actor 1'], imdb_ds['Actor 2'], imdb_ds['Actor 3']], ignore_index=True)
    actor_counts = actors.value_counts()
    imdb_ds['Movies_Per_Actor'] = imdb_ds['Actor 1'].map(actor_counts) + imdb_ds['Actor 2'].map(actor_counts) + imdb_ds['Actor 3'].map(actor_counts)

    # Total votes per actor
    actors_votes_df = pd.DataFrame({'Actor': actors, 'Votes': imdb_ds['Votes']})
    actor_votes_sum = actors_votes_df.groupby('Actor')['Votes'].sum()
    imdb_ds['Total_Votes_Per_Actor'] = imdb_ds['Actor 1'].map(actor_votes_sum) + imdb_ds['Actor 2'].map(actor_votes_sum) + imdb_ds['Actor 3'].map(actor_votes_sum)

    # Total votes per movie
    imdb_ds['Total_Votes_Per_Movie'] = imdb_ds['Votes']

    # Total movies per director
    director_movie_counts = imdb_ds['Director'].value_counts()
    imdb_ds['Movies_Per_Director'] = imdb_ds['Director'].map(director_movie_counts)
    
    # Count movies per year
    imdb_ds['Movies_Per_Year'] = imdb_ds.groupby('Year')['Name'].transform('count')

    return imdb_ds

imdb_ds = analyze_and_concat(imdb_ds)

print('First 5 Rows: \n', imdb_ds.columns)

# Line plot for the count of movies per year
plt.figure(figsize=(14, 8))
sns.lineplot(x='Year', y='Movies_Per_Year', data=imdb_ds, marker='o')
plt.title('Number of Movies per Year')
plt.xlabel('Year')
plt.ylabel('Number of Movies')
plt.savefig(os.path.join(visuals_folder, 'Number_of_movies_per_year.png'))
plt.show()

# Count plot for the total movies per director
plt.figure(figsize=(14, 8))
sns.countplot(x='Movies_Per_Director', data=imdb_ds)
plt.title('Total Movies per Director')
plt.xlabel('Total Movies')
plt.ylabel('Count of Directors')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.savefig(os.path.join(visuals_folder,'movies_per_director.png'))
plt.show()

# Count plot for genre counts
plt.figure(figsize=(14, 8))
sns.countplot(x='Genre_Counts', data=imdb_ds)
plt.title('Genre Counts')
plt.xlabel('Genre Counts')
plt.ylabel('Count of Movies')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.savefig(os.path.join(visuals_folder, 'genre_counts.png'))
plt.show()

# Count plot for movies per actor
plt.figure(figsize=(14, 8))
sns.countplot(x='Movies_Per_Actor', data=imdb_ds)
plt.title('Count of Movies per Actor')
plt.xlabel('Movies Per Actor')
plt.ylabel('Count of Actors')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.savefig(os.path.join(visuals_folder, 'movies_per_actor.png'))
plt.show()

# Box Plot to compare the distribution of movie ratings across different genres
# Split the 'Genre' column into multiple rows
genres = imdb_ds['Genre'].str.split(', ', expand=True).stack()

# Create a DataFrame with 'Name' and 'Rating' for each genre
genre_ratings = pd.DataFrame({'Name': imdb_ds['Name'].repeat(imdb_ds['Genre'].str.count(', ') + 1),
                               'Genre': genres.values,
                               'Rating': imdb_ds['Rating'].repeat(imdb_ds['Genre'].str.count(', ') + 1)})

# Box plot to compare the distribution of movie ratings across different genres
plt.figure(figsize=(14, 8))
sns.boxplot(x='Genre', y='Rating', data=genre_ratings)
plt.title('Distribution of Movie Ratings Across Different Genres')
plt.xlabel('Genre')
plt.ylabel('Rating')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.savefig(os.path.join(visuals_folder, 'distribution_of_movie_ratings.png'))
plt.show()

# Scatter plot for 'Votes' vs 'Rating'
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Votes', y='Rating', data=imdb_ds)
plt.title('Scatter Plot of Votes vs Rating')
plt.xlabel('Votes')
plt.ylabel('Rating')
plt.savefig(os.path.join(visuals_folder, 'votes_vs_rating.png'))
plt.show()

# Scatter plot for 'Duration' vs 'Rating'
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Duration', y='Rating', data=imdb_ds)
plt.title('Scatter Plot of Duration vs Rating')
plt.xlabel('Duration')
plt.ylabel('Rating')
plt.savefig(os.path.join(visuals_folder, 'Scatter_Plot_of_Duration_vs_Rating.png'))
plt.show()

sns.pairplot(imdb_ds[['Rating', 'Votes', 'Duration']], height=5)
plt.suptitle('Pair Plot of Rating, Votes, and Duration', y=1.02)
plt.savefig(os.path.join(visuals_folder, 'Pair_Plot_of_Rating_Votes_and_Durations.png'))
plt.show()

plt.figure(figsize=(14, 8))
sns.barplot(x='Genre', y='Rating', data=genre_ratings, ci=None)
plt.title('Average Rating for Each Genre')
plt.xlabel('Genre')
plt.ylabel('Average Rating')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.savefig(os.path.join(visuals_folder, 'Average Rating for Each Genre.png'))
plt.show()

top_10_movies = imdb_ds.nlargest(10, 'Rating')
plt.figure(figsize=(12, 8))
plt.barh(top_10_movies['Name'], top_10_movies['Rating'], color='skyblue')
plt.xlabel('Rating')
plt.ylabel('Movie Title')
plt.title('Top 10 Movies Based on Ratings')
plt.gca().invert_yaxis()  # To have the highest rating at the top
plt.savefig(os.path.join(visuals_folder, 'Top 10 Movies Based on Ratings.png'))
plt.show()

top_10_genres = imdb_ds['Genre'].value_counts().nlargest(10)
plt.figure(figsize=(12, 8))
top_10_genres.plot(kind='bar', color='skyblue')
plt.xlabel('Genre')
plt.ylabel('Frequency')
plt.title('Top 10 Genres')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.savefig(os.path.join(visuals_folder, 'Top 10 Genres.png'))
plt.show()

actors_columns = ['Actor 1', 'Actor 2', 'Actor 3']
all_actors = pd.concat([imdb_ds[col] for col in actors_columns])
top_10_actors = all_actors.value_counts().nlargest(10)
plt.figure(figsize=(12, 8))
top_10_actors.plot(kind='bar', color='skyblue')
plt.xlabel('Actor')
plt.ylabel('Frequency')
plt.title('Top 10 Actors')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.savefig(os.path.join(visuals_folder, 'Top 10 Actors.png'))
plt.show()

top_10_directors = imdb_ds['Director'].value_counts().nlargest(10)
plt.figure(figsize=(12, 8))
top_10_directors.plot(kind='bar', color='skyblue')
plt.xlabel('Director')
plt.ylabel('Frequency')
plt.title('Top 10 Directors')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.savefig(os.path.join(visuals_folder, 'Top 10 Directors.png'))
plt.show()

# Drop specified columns
Droped_imbd = imdb_ds.drop(['Name', 'Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3'], axis=1)

# Print dropped_imbd
print('\nDropped_imbd\n', Droped_imbd)

# Split the data into input (X) and output (y)
X = Droped_imbd.drop('Rating', axis=1)  # Features
y = Droped_imbd['Rating']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Initialize regression models
linear_regression_model = LinearRegression()
random_forest_model = RandomForestRegressor()
decision_tree_model = DecisionTreeRegressor()
xgb_model = XGBRegressor()
gradient_boosting_model = GradientBoostingRegressor()
lgbm_model = LGBMRegressor()
catboost_model = CatBoostRegressor()
knn_model = KNeighborsRegressor()
svr_model = SVR()

# List of models
models = [
    ("Linear Regression", linear_regression_model),
    ("Random Forest", random_forest_model),
    ("Decision Tree", decision_tree_model),
    ("XGBoost", xgb_model),
    ("Gradient Boosting", gradient_boosting_model),
    ("LightGBM", lgbm_model),
    ("CatBoost", catboost_model),
    ("K-Nearest Neighbors", knn_model),
    ("Support Vector Regression", svr_model),
]

def evaluate_model(model, X_test, y_test):
    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Print evaluation metrics
    print(f"\nEvaluation Metrics for {type(model).__name__}:")
    print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
    print(f"R-squared Score: {r2_score(y_test, y_pred)}")

    return mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred)

# Fit and evaluate each model
for model_name, model in models:
    # Fit the model to the training data
    model.fit(X_train, y_train)
    
    # Evaluate the model
    mse, r2 = evaluate_model(model, X_test, y_test)

from yellowbrick.regressor import ResidualsPlot

visualizer = ResidualsPlot(linear_regression_model)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()

from wordcloud import WordCloud

wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(imdb_ds['Director']))
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for Directors')
plt.savefig(os.path.join(visuals_folder, 'Word_Cloud_for_Movie_Titles.png'))
plt.show()

wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(imdb_ds['Genre']))
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for Movie Genres')
plt.savefig(os.path.join(visuals_folder, 'Word_Cloud_for_Directors.png'))
plt.show()

wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(imdb_ds['Name']))
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for Movie Titles')
plt.savefig(os.path.join(visuals_folder, 'Word_Cloud_for_Movie_Titles.png'))
plt.show()