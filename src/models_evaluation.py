'''
Complete Model Evaluation Pipeline

This script:
1. Generates recommendations using different models
2. Ranks/re-ranks recommendations
3. Evaluates using comprehensive metrics
4. Saves results as JSON
5. Compares models
'''

import os
import logging
from typing import Dict, List
from collections import defaultdict

import polars as pl
from scipy.sparse import load_npz

from src.popular_tracks import PopularityRecommender
from src.als_recommender import ALSRecommender
from src.ranking import RecommendationRanker, combine_recommendations
from src.evaluation import evaluate_recommender, compare_models

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)


def generate_popular_recommendations(
    train_events: pl.DataFrame,
    test_events: pl.DataFrame,
    catalog: pl.DataFrame,
    n: int = 20,
    method: str = 'listen_count'
) -> Dict[int, List[int]]:
    '''
    Generate popularity-based recommendations.
    
    Returns:
        Dict mapping user_id to list of track_ids
    '''
    logger.info(f'Generating popular recommendations (method={method})')
    
    # Train popularity model
    recommender = PopularityRecommender(method=method)
    recommender.fit(train_events, catalog)
    
    # Get test users
    test_users = test_events['user_id'].unique().to_list()
    logger.info(f'Generating for {len(test_users):,} test users')
    
    recommendations = {}
    
    for i, user_id in enumerate(test_users):
        if i % 10000 == 0 and i > 0:
            logger.info(f'  Generated {i:,} / {len(test_users):,}')
        
        recs = recommender.recommend(user_id, train_events, n=n, filter_listened=True)
        recommendations[user_id] = recs
    
    logger.info(f'✅ Generated {len(recommendations):,} recommendation lists')
    return recommendations


def generate_als_recommendations(
    model_path: str,
    train_matrix,
    test_events: pl.DataFrame,
    n: int = 20
) -> Dict[int, List[int]]:
    '''
    Generate ALS-based recommendations.
    
    Returns:
        Dict mapping user_id to list of track_ids
    '''
    logger.info(f'Generating ALS recommendations from {model_path}')
    
    # Load model
    als_model = ALSRecommender.load(model_path)
    
    # Get test users
    test_users = test_events['user_id'].unique().to_list()
    logger.info(f'Generating for {len(test_users):,} test users')
    
    recommendations = {}
    
    for i, user_id in enumerate(test_users):
        if i % 10000 == 0 and i > 0:
            logger.info(f'  Generated {i:,} / {len(test_users):,}')
        
        recs = als_model.recommend(user_id, train_matrix, n=n, filter_already_liked_items=True)
        recommendations[user_id] = [track_id for track_id, _ in recs]
    
    logger.info(f'✅ Generated {len(recommendations):,} recommendation lists')
    return recommendations


def apply_ranking(
    recommendations: Dict[int, List[int]],
    catalog: pl.DataFrame,
    train_events: pl.DataFrame,
    n: int = 10,
    diversity_weight: float = 0.3
) -> Dict[int, List[int]]:
    '''
    Apply re-ranking to recommendations.
    
    Returns:
        Re-ranked recommendations
    '''
    logger.info(f'Applying re-ranking with diversity_weight={diversity_weight}')
    
    ranker = RecommendationRanker(catalog)
    
    # Get popularity scores
    popularity_scores = (
        train_events
        .group_by('track_id')
        .agg(pl.sum('listen_count').alias('popularity'))
        .to_dict()
    )
    pop_dict = dict(zip(popularity_scores['track_id'], popularity_scores['popularity']))
    
    # Get user histories
    user_histories = defaultdict(set)
    for row in train_events.iter_rows(named=True):
        user_histories[row['user_id']].add(row['track_id'])
    
    # Re-rank for each user
    reranked = {}
    
    for user_id, rec_list in recommendations.items():
        # Convert to (track_id, score) format with dummy scores
        rec_tuples = [(tid, 1.0 - i*0.01) for i, tid in enumerate(rec_list)]
        
        # Apply multi-objective ranking
        ranked = ranker.rank_multi_objective(
            rec_tuples,
            popularity_scores=pop_dict,
            user_history=user_histories.get(user_id, set()),
            n=n,
            diversity_weight=diversity_weight,
            popularity_weight=0.1,
            novelty_weight=0.2
        )
        
        reranked[user_id] = [track_id for track_id, _ in ranked]
    
    logger.info(f'✅ Re-ranked {len(reranked):,} recommendation lists')
    return reranked


def run_evaluation_pipeline(
    preprocessed_dir: str = 'data/preprocessed',
    n_recommendations: int = 20,
    k_values: List[int] = [5, 10, 20],
    sample_users: int = None
):
    '''
    Run complete evaluation pipeline.
    
    Args:
        preprocessed_dir: Directory with preprocessed data
        n_recommendations: Number of recommendations to generate
        k_values: K values to evaluate
        sample_users: If set, evaluate on sample of users (for speed)
    '''
    logger.info('='*70)
    logger.info('COMPLETE EVALUATION PIPELINE')
    logger.info('='*70)
    
    # Load data
    logger.info('\n📂 Loading data...')
    train_events = pl.read_parquet(f'{preprocessed_dir}/train_events.parquet')
    test_events = pl.read_parquet(f'{preprocessed_dir}/test_events.parquet')
    catalog = pl.read_parquet(f'{preprocessed_dir}/tracks_catalog_clean.parquet')
    all_events = pl.read_parquet(f'{preprocessed_dir}/events.parquet')
    train_matrix = load_npz(f'{preprocessed_dir}/train_matrix.npz')
    
    total_users = all_events['user_id'].n_unique()
    
    logger.info(f'✅ Data loaded')
    logger.info(f'  Train: {train_events.height:,} interactions')
    logger.info(f'  Test: {test_events.height:,} interactions')
    logger.info(f'  Total users: {total_users:,}')
    
    # Sample users if requested
    if sample_users:
        logger.info(f'\n🎲 Sampling {sample_users:,} users for faster evaluation')
        test_users_sample = test_events['user_id'].unique().sample(n=sample_users, seed=42)
        test_events = test_events.filter(pl.col('user_id').is_in(test_users_sample))
        logger.info(f'✅ Sampled test set: {test_events.height:,} interactions')
    
    # Create output directory
    results_dir = os.path.join(preprocessed_dir, 'evaluation_results')
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Popular baseline
    logger.info('\n' + '='*70)
    logger.info('MODEL 1: POPULARITY BASELINE')
    logger.info('='*70)
    
    popular_recs = generate_popular_recommendations(
        train_events, test_events, catalog, n=n_recommendations
    )
    
    popular_results = evaluate_recommender(
        model_name='PopularityBaseline',
        recommendations=popular_recs,
        test_events_df=test_events,
        catalog_df=catalog,
        all_events_df=all_events,
        total_users=total_users,
        k_values=k_values,
        output_file=os.path.join(results_dir, 'popular_baseline.json')
    )
    
    # 2. ALS model
    logger.info('\n' + '='*70)
    logger.info('MODEL 2: ALS COLLABORATIVE FILTERING')
    logger.info('='*70)
    
    als_model_path = os.path.join(preprocessed_dir, 'als_model.pkl')
    
    if os.path.exists(als_model_path):
        als_recs = generate_als_recommendations(
            als_model_path, train_matrix, test_events, n=n_recommendations
        )
        
        als_results = evaluate_recommender(
            model_name='ALS',
            recommendations=als_recs,
            test_events_df=test_events,
            catalog_df=catalog,
            all_events_df=all_events,
            total_users=total_users,
            k_values=k_values,
            output_file=os.path.join(results_dir, 'als_model.json')
        )
        
        # 3. ALS with diversity re-ranking
        logger.info('\n' + '='*70)
        logger.info('MODEL 3: ALS + DIVERSITY RE-RANKING')
        logger.info('='*70)
        
        als_reranked = apply_ranking(
            als_recs, catalog, train_events,
            n=max(k_values),
            diversity_weight=0.3
        )
        
        als_reranked_results = evaluate_recommender(
            model_name='ALS_Reranked',
            recommendations=als_reranked,
            test_events_df=test_events,
            catalog_df=catalog,
            all_events_df=all_events,
            total_users=total_users,
            k_values=k_values,
            output_file=os.path.join(results_dir, 'als_reranked.json')
        )
    else:
        logger.warning(f'ALS model not found at {als_model_path}')
        logger.warning('Skipping ALS evaluation')
    
    # 4. Compare models
    logger.info('\n' + '='*70)
    logger.info('MODEL COMPARISON')
    logger.info('='*70)
    
    result_files = [
        os.path.join(results_dir, 'popular_baseline.json'),
    ]
    
    if os.path.exists(als_model_path):
        result_files.extend([
            os.path.join(results_dir, 'als_model.json'),
            os.path.join(results_dir, 'als_reranked.json'),
        ])
    
    comparison_df = compare_models(
        result_files,
        output_file=os.path.join(results_dir, 'model_comparison.csv')
    )
    
    logger.info('\n' + '='*70)
    logger.info('✅ EVALUATION PIPELINE COMPLETE')
    logger.info('='*70)
    logger.info(f'\nResults saved to: {results_dir}/')
    logger.info('  - popular_baseline.json')
    if os.path.exists(als_model_path):
        logger.info('  - als_model.json')
        logger.info('  - als_reranked.json')
    logger.info('  - model_comparison.csv')
    logger.info('='*70)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run complete evaluation pipeline')
    parser.add_argument('--preprocessed-dir', type=str, default='data/preprocessed',
                        help='Directory with preprocessed data')
    parser.add_argument('--n-recs', type=int, default=20,
                        help='Number of recommendations to generate')
    parser.add_argument('--k-values', type=int, nargs='+', default=[5, 10, 20],
                        help='K values to evaluate')
    parser.add_argument('--sample-users', type=int, default=None,
                        help='Sample N users for faster evaluation')
    
    args = parser.parse_args()
    
    run_evaluation_pipeline(
        preprocessed_dir=args.preprocessed_dir,
        n_recommendations=args.n_recs,
        k_values=args.k_values,
        sample_users=args.sample_users
    )

