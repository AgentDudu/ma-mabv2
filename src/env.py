import pandas as pd
import numpy as np

def load_movielens_arms(csv_path, min_ratings=100):
    print("Loading MovieLens 20M dataset...")
    df = pd.read_csv(csv_path, usecols=['userId', 'movieId', 'rating'])
    
    df['scaled_rating'] = (df['rating'] - 0.5) / 4.5
    
    print("Calculating user trust weights based on rating volume...")
    user_counts = df['userId'].value_counts().reset_index()
    user_counts.columns = ['userId', 'rating_count']
    user_counts['trust_weight'] = np.log1p(user_counts['rating_count'])
    df = df.merge(user_counts[['userId', 'trust_weight']], on='userId')
    
    print("Calculating weighted Gaussian distributions for each movie...")
    df['weighted_score'] = df['scaled_rating'] * df['trust_weight']
    stats = df.groupby('movieId').agg(
        count=('rating', 'count'),
        sum_weighted_score=('weighted_score', 'sum'),
        sum_weight=('trust_weight', 'sum'),
        std_dev=('scaled_rating', 'std')
    ).reset_index()
    
    stats['weighted_mean'] = stats['sum_weighted_score'] / stats['sum_weight']
    stats['std_dev'] = stats['std_dev'].fillna(0)
    
    valid_movies = stats[stats['count'] >= min_ratings].copy()
    valid_movies = valid_movies.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    means = valid_movies['weighted_mean'].values
    stds = valid_movies['std_dev'].values
    
    print(f"Extracted {len(means)} valid movies (Gaussian/Continuous arms).")
    
    # KITA TIDAK LAGI MENGGUNAKAN CLASS OBJEK! KITA KEMBALIKAN RAW ARRAY
    return means, stds