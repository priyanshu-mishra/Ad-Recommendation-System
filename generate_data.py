import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Set random seed for reproducibility
np.random.seed(42)

# Generate user data
def generate_user_data(n_users):
    return pd.DataFrame({
        'user_id': range(n_users),
        'age': np.random.randint(13, 75, n_users),
        'gender': np.random.choice(['M', 'F', 'O'], n_users),
        'country': np.random.choice(['US', 'UK', 'CA', 'AU', 'DE', 'FR', 'JP', 'BR', 'IN'], n_users),
        'account_age': np.random.randint(1, 3650, n_users),
        'follower_count': np.random.exponential(scale=1000, size=n_users).astype(int),
        'avg_daily_usage': np.random.lognormal(mean=3, sigma=1, size=n_users)
    })

# Generate ad data
def generate_ad_data(n_ads):
    return pd.DataFrame({
        'ad_id': range(n_ads),
        'category': np.random.choice(['Fashion', 'Tech', 'Food', 'Travel', 'Beauty', 'Fitness', 'Entertainment'], n_ads),
        'duration': np.random.randint(5, 120, n_ads),
        'is_skippable': np.random.choice([0, 1], n_ads),
        'ad_quality_score': np.random.uniform(1, 10, n_ads),
        'advertiser_rating': np.random.uniform(1, 5, n_ads),
        'target_age_min': np.random.randint(13, 50, n_ads),
        'target_age_max': np.random.randint(20, 75, n_ads)
    })

# Generate interaction data
def generate_interactions(users, ads, n_interactions):
    user_ids = np.random.choice(users['user_id'], n_interactions)
    ad_ids = np.random.choice(ads['ad_id'], n_interactions)
    
    # Simulate more realistic watch behavior
    watched = []
    for user_id, ad_id in zip(user_ids, ad_ids):
        user = users.loc[users['user_id'] == user_id].iloc[0]
        ad = ads.loc[ads['ad_id'] == ad_id].iloc[0]
        
        # Calculate watch probability based on user and ad features
        watch_prob = 0.3  # Base probability
        watch_prob += 0.1 if user['age'] >= ad['target_age_min'] and user['age'] <= ad['target_age_max'] else -0.1
        watch_prob += 0.05 * (ad['ad_quality_score'] / 10)
        watch_prob += 0.03 * (ad['advertiser_rating'] / 5)
        watch_prob += 0.02 if ad['is_skippable'] else -0.02
        watch_prob = max(0, min(1, watch_prob))  # Ensure probability is between 0 and 1
        
        watched.append(np.random.choice([0, 1], p=[1-watch_prob, watch_prob]))
    
    return pd.DataFrame({
        'user_id': user_ids,
        'ad_id': ad_ids,
        'watched': watched,
        'timestamp': pd.date_range(start='2023-01-01', periods=n_interactions, freq='30S')
    })

# Generate data
n_users = 100000
n_ads = 10000
n_interactions = 1000000

users = generate_user_data(n_users)
ads = generate_ad_data(n_ads)
interactions = generate_interactions(users, ads, n_interactions)

# Merge all data
full_data = interactions.merge(users, on='user_id').merge(ads, on='ad_id')

# Create preprocessing pipeline
numeric_features = ['age', 'account_age', 'follower_count', 'avg_daily_usage', 'duration', 'ad_quality_score', 'advertiser_rating', 'target_age_min', 'target_age_max']
categorical_features = ['gender', 'country', 'category', 'is_skippable']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Fit the preprocessor and transform the data
X = preprocessor.fit_transform(full_data)
y = full_data['watched'].values

# Save preprocessed data
np.save('X_preprocessed.npy', X)
np.save('y_preprocessed.npy', y)

print("Data generation and preprocessing complete.")
print(f"Feature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")