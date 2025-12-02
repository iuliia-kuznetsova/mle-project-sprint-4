'''
Recommendation System Demo

This script demonstrates the complete recommendation workflow:
1. Load preprocessed data
2. Get popular tracks baseline
3. Load trained ALS model
4. Generate personalized recommendations
5. Compare results
'''

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import polars as pl
from src.load_preprocessed import load_interaction_data, load_catalog, load_events
from src.popular_tracks import PopularityRecommender
from src.als_recommender import ALSRecommender


def demo_recommendations(
    user_id: int = 12345,
    n_recommendations: int = 10,
    preprocessed_dir: str = 'data/preprocessed'
):
    '''
    Demonstrate recommendation generation for a user.
    
    Args:
        user_id: User ID to generate recommendations for
        n_recommendations: Number of recommendations
        preprocessed_dir: Directory with preprocessed data
    '''
    print("="*70)
    print("RECOMMENDATION SYSTEM DEMO")
    print("="*70)
    
    # Load data
    print("\n📂 Loading data...")
    matrix, encoders = load_interaction_data(preprocessed_dir)
    catalog = load_catalog(preprocessed_dir)
    events = load_events(preprocessed_dir)
    
    user_encoder = encoders['user_encoder']
    track_decoder = encoders['track_decoder']
    
    print(f"✅ Loaded {len(user_encoder):,} users and {len(encoders['track_encoder']):,} tracks")
    
    # Check if user exists
    if user_id not in user_encoder:
        print(f"\n❌ User {user_id} not found. Selecting a random user...")
        user_id = list(user_encoder.keys())[1000]  # Pick user at index 1000
    
    user_idx = user_encoder[user_id]
    
    # Show user's listening history
    print(f"\n{'='*70}")
    print(f"USER {user_id} PROFILE")
    print(f"{'='*70}")
    
    user_history = events.filter(pl.col('user_id') == user_id)
    n_tracks = user_history.height
    total_listens = user_history['listen_count'].sum()
    
    print(f"  • Total tracks listened: {n_tracks:,}")
    print(f"  • Total listens: {total_listens:,}")
    print(f"  • Average listens per track: {total_listens/n_tracks:.2f}")
    
    # Get top tracks from history
    top_user_tracks = (
        user_history
        .sort('listen_count', descending=True)
        .head(5)
        .join(catalog.select(['track_id', 'track_clean']), on='track_id', how='left')
    )
    
    print(f"\n  Top 5 most listened tracks:")
    for row in top_user_tracks.iter_rows(named=True):
        print(f"    - {row['track_clean']} ({row['listen_count']} listens)")
    
    # Method 1: Popular Tracks Baseline
    print(f"\n{'='*70}")
    print("METHOD 1: POPULAR TRACKS BASELINE")
    print(f"{'='*70}")
    
    popularity_rec = PopularityRecommender(method='listen_count')
    popularity_rec.fit(events, catalog)
    
    popular_recs = popularity_rec.recommend(
        user_id, events, n=n_recommendations, filter_listened=True
    )
    
    popular_tracks = catalog.filter(pl.col('track_id').is_in(popular_recs))
    
    print(f"\nTop {n_recommendations} popular tracks (not yet listened by user):")
    for i, row in enumerate(popular_tracks.iter_rows(named=True), 1):
        print(f"  {i:2d}. {row['track_clean']}")
    
    # Method 2: ALS Personalized Recommendations
    print(f"\n{'='*70}")
    print("METHOD 2: ALS PERSONALIZED RECOMMENDATIONS")
    print(f"{'='*70}")
    
    # Check if model exists
    model_path = os.path.join(preprocessed_dir, 'als_model.pkl')
    
    if not os.path.exists(model_path):
        print(f"\n⚠️  ALS model not found at {model_path}")
        print("   Train model first: python3 -m src.als_recommender")
        return
    
    # Load model
    print("\n📂 Loading ALS model...")
    als_rec = ALSRecommender.load(model_path)
    
    # Generate recommendations
    als_recs = als_rec.recommend(
        user_id,
        matrix,
        n=n_recommendations,
        filter_already_liked_items=True
    )
    
    if als_recs:
        print(f"\nTop {n_recommendations} personalized recommendations:")
        for i, (track_id, score) in enumerate(als_recs, 1):
            track_info = catalog.filter(pl.col('track_id') == track_id)
            if track_info.height > 0:
                track_name = track_info['track_clean'][0]
                print(f"  {i:2d}. {track_name} (score: {score:.4f})")
    else:
        print("\n❌ No recommendations generated")
    
    # Comparison
    print(f"\n{'='*70}")
    print("COMPARISON")
    print(f"{'='*70}")
    
    popular_set = set(popular_recs)
    als_set = set([track_id for track_id, _ in als_recs])
    
    overlap = popular_set & als_set
    
    print(f"\nPopular tracks: {len(popular_set)} recommendations")
    print(f"ALS personalized: {len(als_set)} recommendations")
    print(f"Overlap: {len(overlap)} tracks ({len(overlap)/n_recommendations*100:.1f}%)")
    
    if overlap:
        print(f"\nTracks in both recommendations:")
        overlap_tracks = catalog.filter(pl.col('track_id').is_in(list(overlap)))
        for row in overlap_tracks.iter_rows(named=True):
            print(f"  - {row['track_clean']}")
    
    print(f"\n{'='*70}")
    print("✅ DEMO COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Recommendation system demo')
    parser.add_argument('--user-id', type=int, default=None,
                        help='User ID to generate recommendations for')
    parser.add_argument('--n', type=int, default=10,
                        help='Number of recommendations')
    parser.add_argument('--preprocessed-dir', type=str, default='data/preprocessed',
                        help='Directory with preprocessed data')
    
    args = parser.parse_args()
    
    # If no user_id provided, use a random one
    if args.user_id is None:
        import polars as pl
        events = pl.read_parquet(f'{args.preprocessed_dir}/events.parquet')
        # Get a user with reasonable number of interactions
        user_counts = events.group_by('user_id').agg(pl.len().alias('n'))
        moderate_users = user_counts.filter(
            (pl.col('n') >= 20) & (pl.col('n') <= 100)
        ).sort('n', descending=True)
        
        if moderate_users.height > 0:
            args.user_id = moderate_users['user_id'][0]
        else:
            args.user_id = events['user_id'][0]
        
        print(f"💡 Selected user {args.user_id} for demo\n")
    
    demo_recommendations(
        user_id=args.user_id,
        n_recommendations=args.n,
        preprocessed_dir=args.preprocessed_dir
    )

