'''
Example: How to load and use preprocessed data for model training

This script demonstrates:
1. Loading the sparse interaction matrix
2. Loading label encoders
3. Loading catalog metadata
4. Basic data exploration
5. Preparing data for model training
'''

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.load_preprocessed import (
    load_interaction_data,
    load_catalog,
    load_events,
    get_data_summary
)

import numpy as np
import polars as pl


def main():
    print("="*70)
    print("EXAMPLE: Loading Preprocessed Data")
    print("="*70)
    
    # 1. Load interaction matrix and encoders
    print("\n📂 Step 1: Loading interaction matrix and encoders...")
    matrix, encoders = load_interaction_data()
    
    print(f"✅ Matrix shape: {matrix.shape}")
    print(f"✅ Non-zero entries: {matrix.nnz:,}")
    print(f"✅ Sparsity: {100 * (1 - matrix.nnz / np.prod(matrix.shape)):.4f}%")
    print(f"✅ Data type: {matrix.dtype}")
    
    # 2. Access encoders
    print("\n📂 Step 2: Accessing encoders...")
    user_encoder = encoders['user_encoder']
    track_encoder = encoders['track_encoder']
    user_decoder = encoders['user_decoder']
    track_decoder = encoders['track_decoder']
    
    print(f"✅ Users: {len(user_encoder):,}")
    print(f"✅ Tracks: {len(track_encoder):,}")
    
    # 3. Load catalog
    print("\n📂 Step 3: Loading catalog...")
    catalog = load_catalog()
    print(f"✅ Catalog shape: {catalog.shape}")
    print("\nSample tracks:")
    print(catalog.head(5))
    
    # 4. Get data summary
    print("\n📂 Step 4: Getting data summary...")
    summary = get_data_summary()
    print("\n📊 Summary Statistics:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # 5. Example: Get user's listening history
    print("\n📂 Step 5: Example - Get user's listening history...")
    
    # Get a random user from the first 100 users
    user_idx = 0
    user_id = user_decoder[user_idx]
    
    # Get user's interactions from sparse matrix
    user_interactions = matrix[user_idx].toarray().flatten()
    
    # Find tracks user listened to
    listened_track_indices = np.where(user_interactions > 0)[0]
    listen_counts = user_interactions[listened_track_indices]
    
    print(f"\n👤 User {user_id} (index {user_idx}):")
    print(f"  • Listened to {len(listened_track_indices)} unique tracks")
    print(f"  • Total listens: {int(listen_counts.sum())}")
    print(f"  • Average listens per track: {listen_counts.mean():.2f}")
    
    # Decode track IDs and get top tracks
    listened_track_ids = [track_decoder[idx] for idx in listened_track_indices[:10]]
    
    user_tracks = catalog.filter(pl.col('track_id').is_in(listened_track_ids))
    print(f"\n  Top tracks:")
    print(user_tracks.select(['track_clean', 'artist_id', 'track_group_id']))
    
    # 6. Ready for model training
    print("\n" + "="*70)
    print("✅ Data loaded successfully!")
    print("="*70)
    print("\n💡 Next steps:")
    print("  1. Split data into train/test sets")
    print("  2. Train recommendation model (ALS, SVD, Neural CF, etc.)")
    print("  3. Generate recommendations using track_decoder")
    print("  4. Evaluate using track_group_id for diversity")
    print("\n💡 Example model training code:")
    print("""
    from implicit.als import AlternatingLeastSquares
    
    # Train ALS model
    model = AlternatingLeastSquares(factors=64, regularization=0.01, iterations=15)
    model.fit(matrix.T)  # Transpose for implicit library (items × users)
    
    # Get recommendations
    user_idx = 0
    recommendations = model.recommend(user_idx, matrix[user_idx], N=10)
    
    # Decode track IDs
    rec_track_ids = [track_decoder[idx] for idx, score in recommendations]
    rec_tracks = catalog.filter(pl.col('track_id').is_in(rec_track_ids))
    print(rec_tracks)
    """)


if __name__ == '__main__':
    main()

