'''
    Similar Tracks Finder using implicit's built-in similar_items
'''

import os
import pickle
import logging

import numpy as np

from src.collaborative_rec import load_als_model

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class SimilarTracksFinder:
    '''Find similar tracks using ALS model's built-in similar_items.'''
    
    def __init__(self):
        self.model = None
        self.track_encoder = None
        self.track_decoder = None
        
    def fit(self, als_model):
        '''Load from trained ALS model.'''
        self.model = als_model.model
        self.track_encoder = als_model.track_encoder
        self.track_decoder = als_model.track_decoder
        logger.info(f'Loaded {len(self.track_decoder):,} tracks')
        
    def find_similar(self, track_id, n=10):
        '''Find n most similar tracks.'''
        if track_id not in self.track_encoder:
            return []
        
        track_idx = self.track_encoder[track_id]
        indices, scores = self.model.similar_items(track_idx, N=n+1)
        
        # Skip first (self) and decode
        return [
            (self.track_decoder.get(int(idx)), float(score))
            for idx, score in zip(indices[1:], scores[1:])
            if int(idx) in self.track_decoder
        ]
    
    def build_index(self, top_k=50):
        '''Build index using model's similar_items (batch).'''
        logger.info(f'Building index: top {top_k} per track')
        
        n_tracks = len(self.track_decoder)
        all_indices = np.array(list(self.track_decoder.keys()))
        
        # Use batch similar_items
        logger.info(f'Computing similar items for {len(all_indices):,} tracks...')
        indices_batch, scores_batch = self.model.similar_items(all_indices, N=top_k+1)
        
        logger.info('Building index dict...')
        similar_index = {}
        for i, track_idx in enumerate(all_indices):
            track_id = self.track_decoder[track_idx]
            # Skip first (self)
            similar_index[track_id] = [
                (self.track_decoder.get(int(idx)), float(score))
                for idx, score in zip(indices_batch[i, 1:], scores_batch[i, 1:])
                if int(idx) in self.track_decoder
            ]
        
        # Save
        os.makedirs('./results', exist_ok=True)
        with open('./results/similar_tracks_index.pkl', 'wb') as f:
            pickle.dump(similar_index, f)
        
        logger.info(f'Saved index for {len(similar_index):,} tracks')


def build_similar_tracks_index(preprocessed_dir='data/preprocessed', top_k=50):
    '''Build and save similar tracks index.'''
    logger.info('Building similar tracks index')
    
    als_model = load_als_model(f'{preprocessed_dir}/als_model.pkl')
    
    finder = SimilarTracksFinder()
    finder.fit(als_model)
    finder.build_index(top_k)
    
    logger.info('Done')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocessed-dir', default='data/preprocessed')
    parser.add_argument('--top-k', type=int, default=50)
    args = parser.parse_args()
    
    build_similar_tracks_index(args.preprocessed_dir, args.top_k)
