'''
    ALS Collaborative Filtering Recommender

    Input:
    - train_matrix.npz - Sparse user-track interaction matrix for training
    - label_encoders.pkl - User and track ID to index mappings

    Output:
    - als_model.pkl - Trained ALS model with encoders
'''

import os
import gc
import pickle
import logging

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix, load_npz
from implicit.als import AlternatingLeastSquares
from dotenv import load_dotenv

# ---------- Environment variables ---------- #
load_dotenv()

# ---------- Logging setup ---------- #
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)

# ---------- ALS Recommender ---------- #
class ALSRecommender:
    '''
    ALS-based collaborative filtering recommender.
    '''
    
    def __init__(self, factors=64, regularization=0.01, iterations=15, alpha=1.0):
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha
        
        self.model = AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            random_state=42
        )
        
        self.user_encoder = None
        self.track_encoder = None
        self.user_decoder = None
        self.track_decoder = None
        self.is_fitted = False
        
    def fit(self, train_matrix, user_encoder, track_encoder):
        '''
            Train ALS model on user-track interactions.
        '''
        logger.info(f'Training ALS model: {train_matrix.shape}, {train_matrix.nnz:,} interactions')
        logger.info(f'Params: factors={self.factors}, reg={self.regularization}, iter={self.iterations}')
        
        # Store encoders
        self.user_encoder = user_encoder
        self.track_encoder = track_encoder
        self.user_decoder = {idx: uid for uid, idx in user_encoder.items()}
        self.track_decoder = {idx: tid for tid, idx in track_encoder.items()}
        
        # Apply confidence scaling: C = 1 + alpha * listen_count
        train_confidence = train_matrix.copy()
        train_confidence.data = 1 + self.alpha * train_confidence.data
        
        # Fit model (implicit expects items × users, so transpose)
        self.model.fit(train_confidence.T, show_progress=True)
        self.is_fitted = True
        
        logger.info('Model training complete')
        
    def recommend(self, user_id, user_items, n=10, filter_already_liked=True):
        '''
            Get top-N recommendations for a single given user_id.
        '''

        logger.info(f'Getting top {n} recommendations for user {user_id}')

        # Check if user exists
        if user_id not in self.user_encoder:
            logger.warning(f'User {user_id} not found in training data')
            return []
        
        user_idx = self.user_encoder[user_id]
        
        # Get recommendations
        track_indices, scores = self.model.recommend(
            user_idx,
            user_items[user_idx],
            N=n,
            filter_already_liked_items=filter_already_liked
        )
        
        # Decode track indices to ids
        recommendations = [
            (self.track_decoder[idx], float(score))
            for idx, score in zip(track_indices, scores)
            if idx in self.track_decoder
        ]
        
        logger.info(f'Found {len(recommendations):,} recommendations for user {user_id}')
        
        return recommendations
    
    def evaluate(self, train_matrix, test_matrix, k=10, sample_users=10000):
        '''
        Evaluate precision@k on test set.
        '''
        logger.info(f'Evaluating precision@{k} on {sample_users} users')
        
        # Get users with test interactions
        test_user_indices = np.unique(test_matrix.tocoo().row)
        
        # Filter to users within model bounds (users that were in training)
        n_users = self.model.user_factors.shape[0]
        test_user_indices = test_user_indices[test_user_indices < n_users]
        logger.info(f'Users with test interactions within model bounds: {len(test_user_indices)}')
        
        # Sample users for speed
        if sample_users and sample_users < len(test_user_indices):
            np.random.seed(42)
            test_user_indices = np.random.choice(test_user_indices, sample_users, replace=False)
        
        hits = 0
        total = 0
        
        for user_idx in test_user_indices:
            try:
                # Get recommendations
                rec_indices, _ = self.model.recommend(
                    user_idx, train_matrix[user_idx], N=k, filter_already_liked_items=True
                )
                
                # Get test items
                test_items = set(test_matrix[user_idx].tocoo().col)
                
                if test_items:
                    hits += len(set(rec_indices) & test_items)
                    total += k
            except IndexError:
                continue
        
        precision = hits / total if total > 0 else 0.0
        logger.info(f'Precision@{k}: {precision:.4f}')
        
        return {'precision': precision, 'users_evaluated': len(test_user_indices)}
    
    def generate_recommendations(self, train_matrix, n=10):
        '''
            Generate recommendations for all users and save to parquet.
        '''

        results_dir = os.getenv('RESULTS_DIR', './results')
        output_path = os.path.join(results_dir, 'personal_als.parquet')
        
        # Get all user indices
        all_user_indices = list(self.user_decoder.keys())
        
        logger.info(f'Generating top {n} recommendations for {len(all_user_indices):,} users')
        
        # Generate recommendations for all users
        als_recommendations = []
        for i, user_idx in enumerate(all_user_indices):
            if i % 100000 == 0 and i > 0:
                logger.info(f'Processed {i:,} / {len(all_user_indices):,} users')
            
            try:
                track_indices, scores = self.model.recommend(
                    user_idx, train_matrix[user_idx], N=n, filter_already_liked_items=True
                )
                
                user_id = self.user_decoder[user_idx]
                for rank, (track_idx, score) in enumerate(zip(track_indices, scores), 1):
                    if track_idx in self.track_decoder:
                        als_recommendations.append({
                            'user_id': user_id,
                            'track_id': self.track_decoder[track_idx],
                            'score': float(score),
                            'rank': rank
                        })
            except IndexError:
                continue
        
        # Save results
        results_dir = os.getenv('RESULTS_DIR', './results')
        output_path = os.path.join(results_dir, 'personal_als.parquet')

        os.makedirs(results_dir, exist_ok=True)
        pl.DataFrame(als_recommendations).write_parquet(output_path)
        
        logger.info(f'Saved recommendations for {len(als_recommendations):,} tracks to {output_path}')

        # Free memory
        del als_recommendations
        gc.collect()

        return None
    
    def save(self, filepath):
        '''
            Save ALS model to file.
        '''

        logger.info(f'Saving ALS model to {filepath}')
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'user_encoder': self.user_encoder,
                'track_encoder': self.track_encoder,
                'user_decoder': self.user_decoder,
                'track_decoder': self.track_decoder,
                'factors': self.factors,
                'regularization': self.regularization,
                'iterations': self.iterations,
                'alpha': self.alpha,
            }, f)
        logger.info(f'Model saved to {filepath}')


def load_als_model(filepath):
    '''
        Load trained ALS model from file.
    '''

    logger.info(f'Loading ALS model from {filepath}')
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    recommender = ALSRecommender(
        factors=data['factors'],
        regularization=data['regularization'],
        iterations=data['iterations'],
        alpha=data['alpha']
    )
    
    recommender.model = data['model']
    recommender.user_encoder = data['user_encoder']
    recommender.track_encoder = data['track_encoder']
    recommender.user_decoder = data['user_decoder']
    recommender.track_decoder = data['track_decoder']
    recommender.is_fitted = True
    
    logger.info(f'Model loaded from {filepath}')

    return recommender

# ---------- Main training function ---------- #
def train_als_model(preprocessed_dir=None):
    '''
        Train ALS model on preprocessed data.
    '''
    # Load config from environment
    if preprocessed_dir is None:
        preprocessed_dir = os.getenv('PREPROCESSED_DATA_DIR', 'data/preprocessed')
    
    factors = int(os.getenv('ALS_FACTORS', 64))
    regularization = float(os.getenv('ALS_REGULARIZATION', 0.01))
    iterations = int(os.getenv('ALS_ITERATIONS', 15))
    alpha = float(os.getenv('ALS_ALPHA', 1.0))
    
    logger.info('ALS model training')

    logger.info('Loading train matrix')
    train_path = f'{preprocessed_dir}/train_matrix.npz'
    train_matrix = load_npz(train_path)

    logger.info('Loading encoders')
    with open(f'{preprocessed_dir}/label_encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    
    logger.info('Fitting ALS model')
    recommender = ALSRecommender(
        factors=factors,
        regularization=regularization,
        iterations=iterations,
        alpha=alpha
    )
    recommender.fit(
        train_matrix,
        encoders['user_encoder'],
        encoders['track_encoder']
    )
    
    logger.info('Evaluating ALS model')
    test_path = f'{preprocessed_dir}/test_matrix.npz'
    test_matrix = load_npz(test_path)
    recommender.evaluate(train_matrix, test_matrix, k=10, sample_users=10000)
    
    logger.info('Generating personal recommendations')
    recommender.generate_recommendations(train_matrix, n=10)
    
    logger.info('Saving ALS model')
    recommender.save(f'{preprocessed_dir}/als_model.pkl')

    logger.info('ALS model training complete')

    return None

# ---------- Main entry point ---------- #
if __name__ == '__main__':
    train_als_model()
