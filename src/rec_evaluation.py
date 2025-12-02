'''
Comprehensive Recommendation Evaluation

This module provides metrics for evaluating recommendation quality:
- Precision@K, Recall@K, F1@K
- Coverage (catalog and user coverage)
- Novelty (popularity-based)
- Diversity (using track_group_id)
- NDCG@K (Normalized Discounted Cumulative Gain)

Results saved as JSON for easy analysis.
'''

import os
import json
import logging
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)


class RecommendationEvaluator:
    '''
    Comprehensive recommendation evaluation.
    '''
    
    def __init__(
        self,
        catalog_df: pl.DataFrame,
        events_df: pl.DataFrame
    ):
        '''
        Args:
            catalog_df: Catalog with track metadata
            events_df: Events with user-track interactions
        '''
        self.catalog = catalog_df
        self.events = events_df
        
        # Pre-compute popularity
        self.track_popularity = (
            events_df
            .group_by('track_id')
            .agg(pl.sum('listen_count').alias('popularity'))
            .to_dict()
        )
        
        self.track_pop_dict = dict(zip(
            self.track_popularity['track_id'],
            self.track_popularity['popularity']
        ))
        
        # Track to group mapping
        self.track_to_group = {
            row['track_id']: row['track_group_id']
            for row in catalog_df.select(['track_id', 'track_group_id']).iter_rows(named=True)
        }
        
    def precision_at_k(
        self,
        recommendations: Dict[int, List[int]],
        test_items: Dict[int, Set[int]],
        k: int = 10
    ) -> float:
        '''
        Precision@K: Fraction of recommended items that are relevant.
        
        Args:
            recommendations: Dict mapping user_id to list of recommended track_ids
            test_items: Dict mapping user_id to set of relevant track_ids
            k: Number of recommendations to consider
            
        Returns:
            Average precision@K across users
        '''
        precisions = []
        
        for user_id, rec_list in recommendations.items():
            if user_id not in test_items:
                continue
            
            relevant = test_items[user_id]
            recommended = set(rec_list[:k])
            
            if len(recommended) > 0:
                precision = len(recommended & relevant) / k
                precisions.append(precision)
        
        return np.mean(precisions) if precisions else 0.0
    
    def recall_at_k(
        self,
        recommendations: Dict[int, List[int]],
        test_items: Dict[int, Set[int]],
        k: int = 10
    ) -> float:
        '''
        Recall@K: Fraction of relevant items that are recommended.
        
        Args:
            recommendations: Dict mapping user_id to list of recommended track_ids
            test_items: Dict mapping user_id to set of relevant track_ids
            k: Number of recommendations to consider
            
        Returns:
            Average recall@K across users
        '''
        recalls = []
        
        for user_id, rec_list in recommendations.items():
            if user_id not in test_items:
                continue
            
            relevant = test_items[user_id]
            recommended = set(rec_list[:k])
            
            if len(relevant) > 0:
                recall = len(recommended & relevant) / len(relevant)
                recalls.append(recall)
        
        return np.mean(recalls) if recalls else 0.0
    
    def f1_at_k(
        self,
        recommendations: Dict[int, List[int]],
        test_items: Dict[int, Set[int]],
        k: int = 10
    ) -> float:
        '''
        F1@K: Harmonic mean of precision and recall.
        '''
        precision = self.precision_at_k(recommendations, test_items, k)
        recall = self.recall_at_k(recommendations, test_items, k)
        
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        
        return f1
    
    def ndcg_at_k(
        self,
        recommendations: Dict[int, List[int]],
        test_items: Dict[int, Set[int]],
        k: int = 10
    ) -> float:
        '''
        NDCG@K: Normalized Discounted Cumulative Gain.
        
        Considers ranking order (higher is better for top positions).
        '''
        ndcgs = []
        
        for user_id, rec_list in recommendations.items():
            if user_id not in test_items:
                continue
            
            relevant = test_items[user_id]
            
            # DCG
            dcg = 0.0
            for i, track_id in enumerate(rec_list[:k], 1):
                if track_id in relevant:
                    dcg += 1.0 / np.log2(i + 1)
            
            # IDCG (ideal DCG)
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
            
            if idcg > 0:
                ndcg = dcg / idcg
                ndcgs.append(ndcg)
        
        return np.mean(ndcgs) if ndcgs else 0.0
    
    def catalog_coverage(
        self,
        recommendations: Dict[int, List[int]],
        k: int = 10
    ) -> float:
        '''
        Catalog Coverage: Fraction of catalog items that appear in recommendations.
        
        Measures how diverse the recommendations are across the catalog.
        '''
        recommended_items = set()
        
        for rec_list in recommendations.values():
            recommended_items.update(rec_list[:k])
        
        total_items = self.catalog.height
        coverage = len(recommended_items) / total_items if total_items > 0 else 0.0
        
        return coverage
    
    def user_coverage(
        self,
        recommendations: Dict[int, List[int]],
        total_users: int
    ) -> float:
        '''
        User Coverage: Fraction of users who received recommendations.
        '''
        return len(recommendations) / total_users if total_users > 0 else 0.0
    
    def novelty(
        self,
        recommendations: Dict[int, List[int]],
        k: int = 10
    ) -> float:
        '''
        Novelty: Average unpopularity of recommended items.
        
        Higher novelty means recommending less popular (more novel) items.
        
        Novelty = -log2(popularity / max_popularity)
        '''
        if not self.track_pop_dict:
            return 0.0
        
        max_pop = max(self.track_pop_dict.values())
        novelties = []
        
        for rec_list in recommendations.values():
            for track_id in rec_list[:k]:
                pop = self.track_pop_dict.get(track_id, 1)
                # Normalized popularity
                norm_pop = pop / max_pop
                # Novelty score (higher for less popular items)
                novelty_score = -np.log2(norm_pop + 1e-10)
                novelties.append(novelty_score)
        
        return np.mean(novelties) if novelties else 0.0
    
    def diversity(
        self,
        recommendations: Dict[int, List[int]],
        k: int = 10
    ) -> float:
        '''
        Diversity: Fraction of unique track groups in recommendations.
        
        Uses track_group_id to measure if recommendations contain
        different songs (not just different versions).
        '''
        diversities = []
        
        for rec_list in recommendations.values():
            groups = set()
            for track_id in rec_list[:k]:
                group_id = self.track_to_group.get(track_id, track_id)
                groups.add(group_id)
            
            # Diversity = unique groups / total recommendations
            diversity_score = len(groups) / min(len(rec_list), k) if rec_list else 0
            diversities.append(diversity_score)
        
        return np.mean(diversities) if diversities else 0.0
    
    def hit_rate_at_k(
        self,
        recommendations: Dict[int, List[int]],
        test_items: Dict[int, Set[int]],
        k: int = 10
    ) -> float:
        '''
        Hit Rate@K: Fraction of users for whom at least one relevant item was recommended.
        '''
        hits = 0
        total = 0
        
        for user_id, rec_list in recommendations.items():
            if user_id not in test_items:
                continue
            
            relevant = test_items[user_id]
            recommended = set(rec_list[:k])
            
            if len(recommended & relevant) > 0:
                hits += 1
            total += 1
        
        return hits / total if total > 0 else 0.0
    
    def evaluate_all(
        self,
        recommendations: Dict[int, List[int]],
        test_items: Dict[int, Set[int]],
        total_users: int,
        k: int = 10
    ) -> Dict[str, float]:
        '''
        Compute all evaluation metrics.
        
        Args:
            recommendations: Dict mapping user_id to list of recommended track_ids
            test_items: Dict mapping user_id to set of relevant track_ids (from test set)
            total_users: Total number of users in dataset
            k: Number of recommendations to evaluate
            
        Returns:
            Dictionary with all metrics
        '''
        logger.info(f'Computing all metrics for k={k}')
        
        metrics = {
            f'precision@{k}': self.precision_at_k(recommendations, test_items, k),
            f'recall@{k}': self.recall_at_k(recommendations, test_items, k),
            f'f1@{k}': self.f1_at_k(recommendations, test_items, k),
            f'ndcg@{k}': self.ndcg_at_k(recommendations, test_items, k),
            f'hit_rate@{k}': self.hit_rate_at_k(recommendations, test_items, k),
            f'catalog_coverage@{k}': self.catalog_coverage(recommendations, k),
            'user_coverage': self.user_coverage(recommendations, total_users),
            f'novelty@{k}': self.novelty(recommendations, k),
            f'diversity@{k}': self.diversity(recommendations, k),
        }
        
        # Add metadata
        metrics['_metadata'] = {
            'k': k,
            'num_users_evaluated': len(recommendations),
            'total_users': total_users,
            'num_test_users': len(test_items),
        }
        
        return metrics


def evaluate_recommender(
    model_name: str,
    recommendations: Dict[int, List[int]],
    test_events_df: pl.DataFrame,
    catalog_df: pl.DataFrame,
    all_events_df: pl.DataFrame,
    total_users: int,
    k_values: List[int] = [5, 10, 20],
    output_file: Optional[str] = None
) -> Dict:
    '''
    Evaluate a recommender and save results.
    
    Args:
        model_name: Name of the model being evaluated
        recommendations: User recommendations
        test_events_df: Test interactions
        catalog_df: Catalog with metadata
        all_events_df: All events (for popularity)
        total_users: Total users in dataset
        k_values: List of K values to evaluate
        output_file: Optional JSON file to save results
        
    Returns:
        Evaluation results dictionary
    '''
    logger.info('='*60)
    logger.info(f'EVALUATING: {model_name}')
    logger.info('='*60)
    
    # Build test items set
    logger.info('Building test items set')
    test_items = defaultdict(set)
    
    for row in test_events_df.iter_rows(named=True):
        test_items[row['user_id']].add(row['track_id'])
    
    logger.info(f'Test set: {len(test_items):,} users, {test_events_df.height:,} interactions')
    
    # Create evaluator
    evaluator = RecommendationEvaluator(catalog_df, all_events_df)
    
    # Evaluate for each K
    results = {
        'model_name': model_name,
        'evaluation_date': str(np.datetime64('now')),
        'metrics_by_k': {}
    }
    
    for k in k_values:
        logger.info(f'\nEvaluating at K={k}')
        metrics = evaluator.evaluate_all(recommendations, test_items, total_users, k)
        results['metrics_by_k'][k] = metrics
        
        # Log key metrics
        logger.info(f'  Precision@{k}: {metrics[f"precision@{k}"]:.4f}')
        logger.info(f'  Recall@{k}: {metrics[f"recall@{k}"]:.4f}')
        logger.info(f'  NDCG@{k}: {metrics[f"ndcg@{k}"]:.4f}')
        logger.info(f'  Hit Rate@{k}: {metrics[f"hit_rate@{k}"]:.4f}')
        logger.info(f'  Coverage@{k}: {metrics[f"catalog_coverage@{k}"]:.4f}')
        logger.info(f'  Novelty@{k}: {metrics[f"novelty@{k}"]:.4f}')
        logger.info(f'  Diversity@{k}: {metrics[f"diversity@{k}"]:.4f}')
    
    # Save results
    if output_file:
        logger.info(f'\nSaving results to {output_file}')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info('✅ Results saved')
    
    logger.info('='*60)
    logger.info(f'✅ EVALUATION COMPLETE: {model_name}')
    logger.info('='*60)
    
    return results


def compare_models(
    results_files: List[str],
    output_file: Optional[str] = None
) -> pl.DataFrame:
    '''
    Compare multiple model evaluation results.
    
    Args:
        results_files: List of JSON result files
        output_file: Optional CSV file to save comparison
        
    Returns:
        DataFrame with comparison
    '''
    logger.info('='*60)
    logger.info('COMPARING MODELS')
    logger.info('='*60)
    
    all_results = []
    
    for file_path in results_files:
        logger.info(f'Loading {file_path}')
        with open(file_path) as f:
            results = json.load(f)
        
        model_name = results['model_name']
        
        # Extract metrics for each K
        for k, metrics in results['metrics_by_k'].items():
            row = {'model': model_name, 'k': k}
            row.update({
                key: value for key, value in metrics.items()
                if not key.startswith('_')
            })
            all_results.append(row)
    
    # Create DataFrame
    comparison_df = pl.DataFrame(all_results)
    
    # Display
    logger.info('\nModel Comparison:')
    print(comparison_df)
    
    # Save
    if output_file:
        comparison_df.write_csv(output_file)
        logger.info(f'\n✅ Comparison saved to {output_file}')
    
    return comparison_df


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate recommender models')
    parser.add_argument('--model-name', type=str, required=True,
                        help='Name of the model being evaluated')
    parser.add_argument('--preprocessed-dir', type=str, default='data/preprocessed',
                        help='Directory with preprocessed data')
    parser.add_argument('--k-values', type=int, nargs='+', default=[5, 10, 20],
                        help='K values to evaluate')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for results')
    
    args = parser.parse_args()
    
    logger.info('This script requires recommendations to be generated first.')
    logger.info('Use evaluate_recommender() function in your code.')

