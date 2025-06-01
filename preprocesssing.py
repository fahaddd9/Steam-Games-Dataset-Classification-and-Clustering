import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load the dataset
df = pd.read_csv('C:\\UNIVERSITY\\steamGames.csv')  # Adjust path to your dataset file

# Step 1: Explore the dataset
print("Dataset Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nMissing Values:\n", df.isnull().sum())
print("\nGenre Distribution:\n", df['genre'].value_counts())

# Step 2: Feature Engineering
# Extract primary genre (first genre in the list)
df['primary_genre'] = df['genre'].apply(lambda x: x.split(',')[0] if isinstance(x, str) else 'Unknown')

# Filter to top 4 genres (e.g., Action, Adventure, RPG, Simulation)
top_genres = ['Action', 'Adventure', 'RPG', 'Simulation']
df = df[df['primary_genre'].isin(top_genres)]
print("\nFiltered Genre Distribution:\n", df['primary_genre'].value_counts())

# Fix numerical columns: Convert original_price, discount_price, and achievements to numeric
# Remove '$' and handle special cases like 'Free to Play'
def convert_price_to_numeric(price):
    if pd.isna(price):
        return np.nan
    if isinstance(price, str):
        price = price.replace('$', '').strip()
        if price.lower() in ['free to play', 'free', 'play for free!']:
            return 0.0
        try:
            return float(price)
        except ValueError:
            return np.nan
    return float(price)

df['original_price'] = df['original_price'].apply(convert_price_to_numeric)
df['discount_price'] = df['discount_price'].apply(convert_price_to_numeric)

# Convert achievements to numeric, handle non-numeric values
df['achievements'] = pd.to_numeric(df['achievements'], errors='coerce')

# Derive numerical features
# Release year from release_date (standardize format)
df['release_date'] = df['release_date'].str.strip()
df['release_year'] = pd.to_datetime(df['release_date'], format='%d-%b-%y', errors='coerce').dt.year
# Handle dates that might be in a different format (e.g., '6-May-03')
df['release_year'] = df['release_year'].fillna(
    pd.to_datetime(df['release_date'], format='%d-%b-%Y', errors='coerce').dt.year
)
# Fill remaining NaNs with median
df['release_year'] = df['release_year'].fillna(df['release_year'].median())

# Number of languages
df['num_languages'] = df['languages'].apply(lambda x: len(x.split(',')) if isinstance(x, str) else 0)

# Number of tags
df['num_tags'] = df['popular_tags'].apply(lambda x: len(x.split(',')) if isinstance(x, str) else 0)

# Sentiment score from recent_reviews
def map_sentiment(review):
    if pd.isna(review):
        return 0
    if 'Very Positive' in review:
        return 1
    elif 'Mostly Positive' in review:
        return 0.5
    elif 'Mixed' in review:
        return 0
    elif 'Negative' in review or 'Mostly Negative' in review:
        return -1
    else:
        return 0

df['sentiment_score'] = df['recent_reviews'].apply(map_sentiment)

# Binary feature: requires GPU
df['requires_gpu'] = df['minimum_requirements'].apply(lambda x: 1 if isinstance(x, str) and 'GPU' in x else 0)

# Word count for game_description
df['desc_word_count'] = df['game_description'].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)

# Step 3: Handle Missing Values
# Numerical features
numerical_cols = ['original_price', 'discount_price', 'achievements', 'release_year', 'num_languages', 'num_tags', 'sentiment_score', 'desc_word_count']
for col in numerical_cols:
    df[col] = df[col].fillna(df[col].mean())

# Categorical features
categorical_cols = ['developer', 'publisher', 'popular_tags', 'game_details', 'mature_content']
for col in categorical_cols:
    df[col] = df[col].fillna('Unknown')

# Drop rows with excessive missing data (e.g., >50% missing)
df = df.dropna(thresh=len(df.columns) * 0.5)

# Step 4: Feature Selection
selected_features = numerical_cols + categorical_cols + ['requires_gpu']
X = df[selected_features]
y = df['primary_genre']

# Step 5: Encode Categorical Features and Normalize Numerical Features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols + ['requires_gpu']),
        ('cat', OneHotEncoder(handle_unknown='ignore', max_categories=10), categorical_cols)  # Limit categories to top 10
    ])

# Apply preprocessing
X_processed = preprocessor.fit_transform(X)

# Step 6: Train-Test Split for Classification
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Step 7: Prepare Data for Clustering (numerical features only)
X_clustering = df[numerical_cols + ['requires_gpu']]
scaler = StandardScaler()
X_clustering_scaled = scaler.fit_transform(X_clustering)

# Save preprocessed data
np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)
np.save('X_clustering_scaled.npy', X_clustering_scaled)
df.to_csv('preprocessed_steam_games.csv', index=False)

print("Preprocessing completed. Data saved for classification and clustering.")