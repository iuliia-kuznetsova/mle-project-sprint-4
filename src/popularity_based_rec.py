'''
    Populararity-based Recommender

    This module provides functionality to find and recommend the most popular tracks.

    Popularity can be measured by:
    - Total listen count
    - Number of unique users
    - Average listens per user

    Input:
    - events.parquet - user-track interaction events
    - ./models/label_encoders.pkl - user and track ID to index mappings (for model training)

    Output:
    - top_popular_tracks.parquet - top N popular tracks

    Usage:
    python -m src.popularity_based_rec --top-tracks
    python -m src.popularity_based_rec --user-id 1234567890 --n-recs 20
'''

# ---------- Imports ---------- #
import os
import gc
import logging
import argparse
from typing import List

import polars as pl
from dotenv import load_dotenv

from src.s3_utils import upload_recommendations_to_s3

# ---------- Load environment variables ---------- #
# Load from config/.env (relative to project root)
config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config')
load_dotenv(os.path.join(config_dir, '.env'))

# ---------- Logging setup ---------- #
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)

# ---------- Popularity-based Recommender ---------- #
class PopularityRecommender:
    '''
        Simple popularity-based recommender.
        
        Recommends the most popular tracks that the user hasn't interacted with yet.
        Compute track popularity from events based on the method provided.
            
        Args:
        - method - method to compute track popularity (listen_count, user_count, avg_listens)

        Attributes:
        - popular_tracks - DataFrame with columns [track_id, popularity_score]
        - catalog - Optional catalog DataFrame for metadata

        Methods:
        - fit - compute track popularity from events based on the method provided
        - get_top_tracks - get top N most popular tracks
        - recommend - recommend top popular tracks for a single given user_id that the user hasn't listened to
    '''

    def __init__(self, method: str = 'listen_count'):
        self.method = method
        self.popular_tracks = None
        self.catalog = None
        
    def fit(self, events_or_path):
        '''
            Compute track popularity from events based on the method provided.
            
            Args:
                events_or_path: Either a Polars DataFrame/LazyFrame with events,
                               or a string path to preprocessed directory.
        '''

        if isinstance(events_or_path, str):
            logger.info('Loading events from %s', events_or_path)
            events = pl.scan_parquet(f'{events_or_path}/events.parquet')
        elif isinstance(events_or_path, pl.DataFrame):
            logger.info('Using provided events DataFrame')
            events = events_or_path.lazy()
        elif isinstance(events_or_path, pl.LazyFrame):
            logger.info('Using provided events LazyFrame')
            events = events_or_path
        else:
            raise TypeError(f'Expected str, DataFrame, or LazyFrame, got {type(events_or_path)}')

        logger.info('Computing track popularity using method: %s', self.method)        
        if self.method == 'listen_count':
            # Total listens per track
            popularity = (
                events
                    .group_by('track_id')
                    .agg(pl.sum('listen_count').alias('popularity_score'))
            )
        elif self.method == 'user_count':
            # Number of unique users per track
            popularity = (
                events
                    .group_by('track_id')
                    .agg(pl.col('user_id').n_unique().alias('popularity_score'))
            )
        elif self.method == 'avg_listens':
            # Average listens per user (for users who listened)
            popularity = (
                events
                    .group_by('track_id')
                    .agg([
                        pl.sum('listen_count').alias('total_listens'),
                        pl.col('user_id').n_unique().alias('n_users')
                    ])
                    .with_columns(
                        (pl.col('total_listens') / pl.col('n_users')).alias('popularity_score')
                    )
                    .select(['track_id', 'popularity_score'])
            )
        else:
            raise ValueError(f'Unknown method: {self.method}')
        
        # Sort by popularity and collect
        self.popular_tracks = popularity.sort('popularity_score', descending=True).collect()
        
        logger.info(f'Computed popularity for {self.popular_tracks.height:,} tracks')
        logger.info(f'Top track score: {self.popular_tracks["popularity_score"][0]:.2f}')

        # Free up memory
        del (events, popularity)
        gc.collect()


    def get_top_tracks(self, preprocessed_dir: str, n: int = 100, with_metadata: bool = False) -> pl.DataFrame:
        '''
            Get top N most popular tracks.
        '''
        if self.popular_tracks is None:
            raise ValueError('Model not fitted. Call fit() first.')
        
        top_tracks = self.popular_tracks.head(n)
        
        if with_metadata:
            logger.info('Loading catalog from %s', preprocessed_dir)
            catalog = pl.read_parquet(f'{preprocessed_dir}/tracks_catalog_clean.parquet')
            top_tracks = top_tracks.join(
                catalog.select(['track_id', 'track_clean', 'artist_id', 'track_group_id']),
                on='track_id',
                how='left'
            )

        results_dir = os.getenv('RESULTS_DIR', './results')
        os.makedirs(results_dir, exist_ok=True)
        output_path = os.path.join(results_dir, 'top_popular.parquet')
        top_tracks.write_parquet(output_path)
        logger.info('Results of popularity-based recommendations saved to %s', output_path)

        # Upload to S3
        upload_recommendations_to_s3(output_path, 'top_popular.parquet')

        # Free up memory
        del (catalog, top_tracks)
        gc.collect()

        return None
    
    # ---------- Recommend top popular tracks for a user ---------- #
    def recommend_to_one(self, events_or_path, user_id: int, n: int = 10, filter_listened: bool = True) -> List[int]:
        '''
            Recommend top popular tracks for a single given user_id that the user hasn't listened to.
            
            Args:
                events_or_path: Either a Polars DataFrame/LazyFrame with events,
                               or a string path to preprocessed directory.
                user_id: User ID to generate recommendations for.
                n: Number of recommendations to return.
                filter_listened: Whether to filter out tracks the user has already listened to.
        '''

        if self.popular_tracks is None:
            raise ValueError('Model not fitted.')
        
        if filter_listened:
            # Get events as LazyFrame
            if isinstance(events_or_path, str):
                events = pl.scan_parquet(f'{events_or_path}/events.parquet')
            elif isinstance(events_or_path, pl.DataFrame):
                events = events_or_path.lazy()
            elif isinstance(events_or_path, pl.LazyFrame):
                events = events_or_path
            else:
                raise TypeError(f'Expected str, DataFrame, or LazyFrame, got {type(events_or_path)}')
            
            # Get user's listened tracks
            user_tracks = (
                events
                    .filter(pl.col('user_id') == user_id)
                    .select(['track_id'])
                    .collect()
            )['track_id'].to_list()

            # Collect garbage
            gc.collect()
        else:
            user_tracks = []
        
        # Get top popular tracks that the user hasn't listened to
        recommendations = (
            self.popular_tracks
                .filter(
                    ~pl.col('track_id').is_in(user_tracks)
                )
                .head(n)
            )

        # Collect garbage
        gc.collect()
        
        return recommendations['track_id'].to_list()
    
    def recommend(self, user_id: int, events_or_path, n: int = 10, filter_listened: bool = True) -> List[int]:
        '''
            Alias for recommend_to_one with user_id as first argument.
        '''
        return self.recommend_to_one(events_or_path, user_id, n=n, filter_listened=filter_listened)


def find_top_popular_tracks(preprocessed_dir: str = None) -> None:
    '''
        Find and save top N popular tracks.
    '''
    
    # Load config from environment
    if preprocessed_dir is None:
        preprocessed_dir = os.getenv('PREPROCESSED_DATA_DIR', 'data/preprocessed')
    n = int(os.getenv('POPULARITY_TOP_N', 100))
    method = os.getenv('POPULARITY_METHOD', 'listen_count')

    recommender = PopularityRecommender(method=method)
    recommender.fit(preprocessed_dir)
    
    # Get top tracks
    recommender.get_top_tracks(preprocessed_dir=preprocessed_dir, n=n, with_metadata=True)

    # Free up memory
    del recommender
    gc.collect()

    return None


def generate_popularity_recommendations(preprocessed_dir: str, n: int = 100) -> List[int]:
    '''
    Generate popularity-based recommendations.
    
    Args:
        preprocessed_dir: Path to preprocessed data directory
        n: Number of top popular tracks to return
        
    Returns:
        List of track_ids sorted by popularity
    '''
    logger.info(f'Generating top {n} popularity recommendations')
    
    results_dir = os.getenv('RESULTS_DIR', './results')
    popularity_path = os.path.join(results_dir, 'top_popular.parquet')
    
    # Check if pre-computed recommendations exist
    if os.path.exists(popularity_path):
        logger.info(f'Loading existing popularity recommendations from {popularity_path}')
        popular_tracks = pl.read_parquet(popularity_path)
        return popular_tracks['track_id'].head(n).to_list()
    
    # Otherwise compute from scratch
    method = os.getenv('POPULARITY_METHOD', 'listen_count')
    recommender = PopularityRecommender(method=method)
    recommender.fit(preprocessed_dir)
    
    # Get top tracks
    top_tracks = recommender.popular_tracks.head(n)['track_id'].to_list()
    
    logger.info(f'Generated {len(top_tracks)} popularity recommendations')
    return top_tracks


# ---------- Main entry point ---------- #
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Popularity-based Recommender')
    parser.add_argument('--top-tracks', action='store_true', help='Find and save top N popular tracks')
    parser.add_argument('--user-id', type=int, help='Get recommendations for a specific user ID')
    parser.add_argument('--n-recs', type=int, default=10, help='Number of recommendations (default: 10)')
    parser.add_argument('--method', type=str, default=None, 
                        help='Popularity method: listen_count, user_count, avg_listens')
    args = parser.parse_args()

    preprocessed_dir = os.getenv('PREPROCESSED_DATA_DIR', 'data/preprocessed')
    method = args.method or os.getenv('POPULARITY_METHOD', 'listen_count')

    if args.top_tracks:
        logger.info('Finding top popular tracks')
        find_top_popular_tracks(preprocessed_dir)
        logger.info('Top popular tracks found')

    elif args.user_id:
        logger.info(f'Getting recommendations for user {args.user_id}')
        
        recommender = PopularityRecommender(method=method)
        recommender.fit(preprocessed_dir)
        
        recommendations = recommender.recommend_to_one(
            preprocessed_dir=preprocessed_dir,
            user_id=args.user_id,
            n=args.n_recs,
            filter_listened=True
        )
        
        print(f'\nTop {args.n_recs} popular tracks for user {args.user_id}:')
        for i, track_id in enumerate(recommendations, 1):
            print(f'  {i}. Track {track_id}')

    else:
        parser.print_help()
        print('\nExamples:')
        print('  python -m src.popularity_based_rec --top-tracks')
        print('  python -m src.popularity_based_rec --user-id 12345 --n-recs 20')
        print('  python -m src.popularity_based_rec --top-tracks --method user_count')

# ---------- All exports ---------- #
__all__ = ['PopularityRecommender', 'find_top_popular_tracks', 'generate_popularity_recommendations']

