'''
    Similar Tracks using implicit's built-in ALS model's similar_items method
'''

# ---------- Imports ---------- #
import os
import gc
import logging
import argparse

import numpy as np
import polars as pl
from dotenv import load_dotenv
from src.collaborative_rec import load_als_model

# ---------- Environment variables ---------- #
load_dotenv()

# ---------- Logging setup ---------- #
logging.basicConfig(
    level=logging.INFO, 
    format='[%(asctime)s] [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# ---------- Similar Tracks Finder using ALS model's built-in similar_items method ---------- #
class ALSSimilarTracks:
    '''
        Find similar tracks using ALS model's built-in similar_items.

        Args:
            als_model: ALS model instance
            track_id: track id to find similar tracks for
            n: number of similar tracks to return

        Returns:
            List of (track_id, score) tuples (excluding self)
    '''
    
    def __init__(self):
        self.model = None
        self.track_encoder = None
        self.track_decoder = None
        
    def fit(self, als_model):
        '''
            Load from trained ALS model.
        '''

        self.model = als_model.model
        self.track_encoder = als_model.track_encoder
        self.track_decoder = als_model.track_decoder
        logger.info(f'Loaded {len(self.track_decoder):,} tracks')
        
    def find_similar(self, track_id, n=10):
        '''
            Find n most similar tracks to a singlegiven track_id.
        '''

        logger.info(f'Finding similar tracks to {track_id}')

        # Check if track exists
        if track_id not in self.track_encoder:
            logger.warning(f'Track {track_id} not found')
            return []
        
        # Get track index
        track_idx = self.track_encoder[track_id]

        # Get similar items
        indices, scores = self.model.similar_items(track_idx, N=n+1)
        
        # Decode track indices to ids
        # Skip first (self) and decode
        return [
            (self.track_decoder.get(int(idx)), float(score))
            for idx, score in zip(indices[1:], scores[1:])
            if int(idx) in self.track_decoder
        ]
    
    def build_full_index(self, top_k=None):
        '''
            Build full similar tracks index for all tracks at a time.
        '''

        # Get top k
        if top_k is None:
            top_k = int(os.getenv('SIMILAR_TRACKS_TOP_K', 10))
        
        logger.info(f'Building full index: top {top_k} per track')
        
        # Get all indices
        all_indices = np.array(list(self.track_decoder.keys()))
        logger.info(f'Computing similar items for {len(all_indices):,} tracks')
        
        # Get similar items
        indices_batch, scores_batch = self.model.similar_items(all_indices, N=top_k+1)
        
        # Build index dict
        logger.info('Building index dict')
        similar = {}
        for i, track_idx in enumerate(all_indices):
            track_id = self.track_decoder[track_idx]
            similar[track_id] = [
                (self.track_decoder.get(int(idx)), float(score))
                for idx, score in zip(indices_batch[i, 1:], scores_batch[i, 1:])
                if int(idx) in self.track_decoder
            ]
        
        # Save results
        results_dir = os.getenv('RESULTS_DIR', './results')
        output_path = os.path.join(results_dir, 'similar.parquet')

        os.makedirs(results_dir, exist_ok=True)
        pl.DataFrame(similar).write_parquet(output_path)
        
        logger.info(f'Saved index for {len(similar):,} tracks to {output_path}')

        # Free memory
        del similar
        gc.collect()

        return None

def get_similar_tracks(preprocessed_dir=None):
    '''
        Load ALS model and return ALSSimilarTracks instance.
    '''

    if preprocessed_dir is None:
        preprocessed_dir = os.getenv('PREPROCESSED_DATA_DIR', 'data/preprocessed')
    
    als_model = load_als_model(f'{preprocessed_dir}/als_model.pkl')
    
    finder = ALSSimilarTracks()
    finder.fit(als_model)
    
    return finder

# ---------- Main entry point ---------- #
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Find similar tracks using ALS')
    parser.add_argument('--track-id', type=int, help='Find similar tracks for this track')
    parser.add_argument('--build-index', action='store_true', help='Build full index')
    args = parser.parse_args()
    
    finder = get_similar_tracks()
    
    if args.build_index:
        finder.build_full_index()
    elif args.track_id:
        similar = finder.find_similar(args.track_id)
        print(f'Similar tracks to {args.track_id}:')
        for track_id, score in similar:
            print(f'{track_id}: {score:.4f}')
    else:
        print('Use --track-id <id> to find similar tracks or --build-index to build full index')