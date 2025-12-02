'''
    Populararity-based Recommender

    This module provides functionality to find and recommend the most popular tracks.

    Popularity can be measured by:
    - Total listen count
    - Number of unique users
    - Average listens per user

    Input:
    - events.parquet - user-track interaction events
    - label_encoders.pkl - user and track ID to index mappings (for model training)

    Output:
    - top_popular_tracks.parquet - top N popular tracks
'''

# ---------- Imports ---------- #
import os
import gc
import logging
from typing import List, Optional

import polars as pl
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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
            method: Method to compute track popularity
            events_df: DataFrame with columns [user_id, track_id, listen_count]
            catalog_df: Optional catalog DataFrame for metadata
    '''

    def __init__(self, method: str = 'listen_count'):
        self.method = method
        self.popular_tracks = None  # DataFrame with track_id and popularity_score
        self.catalog = None  # Optional catalog DataFrame for metadata
        
    def fit(self, preprocessed_dir: str):
        '''
            Compute track popularity from events based on the method provided.
        '''

        logger.info('Loading events from %s', preprocessed_dir)
        events = pl.scan_parquet(f'{preprocessed_dir}/events.parquet')

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

        # Remove all dataframes from memory
        del (events, popularity)
        # Collect garbage
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

        # Free memory
        del (catalog, top_tracks)
        gc.collect()

        return None
    
    # ---------- Recommend top popular tracks for a user ---------- #
    def recommend(self, preprocessed_dir: str, user_id: int, n: int = 10, filter_listened: bool = True) -> List[int]:
        '''
            Recommend top popular tracks for a single given user_id that the user hasn't listened to.
        '''

        logger.info('Recommending top popular tracks for user %s', user_id)
        if self.popular_tracks is None:
            raise ValueError('Model not fitted.')
        
        if filter_listened:
            logger.info('Loading events from %s', preprocessed_dir)
            events = pl.scan_parquet(f'{preprocessed_dir}/events.parquet')
            # Get user's listened tracks
            user_tracks = (
                events
                    .filter(pl.col('user_id') == user_id)
                    .select(['track_id'])
            )['track_id'].to_list()

            # Remove all dataframes from memory
            del (events)
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


    del recommender
    gc.collect()

    return None

if __name__ == '__main__':
    logger.info('Finding top popular tracks')
    find_top_popular_tracks()
    logger.info('Top popular tracks found')

