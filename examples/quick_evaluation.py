'''
Quick Evaluation Example

Demonstrates how to evaluate recommendations using the comprehensive metrics.
'''

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import polars as pl
from collections import defaultdict

from src.popular_tracks import PopularityRecommender
from src.evaluation import RecommendationEvaluator

def main():
    print("="*70)
    print("QUICK EVALUATION EXAMPLE")
    print("="*70)
    
    # Load data
    print("\n📂 Loading data...")
    preprocessed_dir = 'data/preprocessed'
    
    train_events = pl.read_parquet(f'{preprocessed_dir}/train_events.parquet')
    test_events = pl.read_parquet(f'{preprocessed_dir}/test_events.parquet')
    catalog = pl.read_parquet(f'{preprocessed_dir}/tracks_catalog_clean.parquet')
    all_events = pl.read_parquet(f'{preprocessed_dir}/events.parquet')
    
    # Sample for demo
    n_sample = 1000
    print(f"\n🎲 Sampling {n_sample} users for demo...")
    test_users_sample = test_events['user_id'].unique().sample(n=n_sample, seed=42).to_list()
    test_events_sample = test_events.filter(pl.col('user_id').is_in(test_users_sample))
    
    print(f"✅ Sampled {test_events_sample.height:,} test interactions")
    
    # Generate popular recommendations
    print("\n📊 Generating popularity-based recommendations...")
    recommender = PopularityRecommender(method='listen_count')
    recommender.fit(train_events, catalog)
    
    recommendations = {}
    for user_id in test_users_sample:
        recs = recommender.recommend(user_id, train_events, n=20, filter_listened=True)
        recommendations[user_id] = recs
    
    print(f"✅ Generated {len(recommendations):,} recommendation lists")
    
    # Build test items
    print("\n📊 Building test items...")
    test_items = defaultdict(set)
    for row in test_events_sample.iter_rows(named=True):
        test_items[row['user_id']].add(row['track_id'])
    
    # Evaluate
    print("\n📊 Evaluating recommendations...")
    evaluator = RecommendationEvaluator(catalog, all_events)
    
    total_users = all_events['user_id'].n_unique()
    
    metrics_k5 = evaluator.evaluate_all(recommendations, test_items, total_users, k=5)
    metrics_k10 = evaluator.evaluate_all(recommendations, test_items, total_users, k=10)
    
    # Display results
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    
    print("\n📈 Metrics @ K=5:")
    for metric, value in metrics_k5.items():
        if not metric.startswith('_'):
            print(f"  {metric}: {value:.4f}")
    
    print("\n📈 Metrics @ K=10:")
    for metric, value in metrics_k10.items():
        if not metric.startswith('_'):
            print(f"  {metric}: {value:.4f}")
    
    # Save results
    results = {
        'model_name': 'PopularityBaseline_Demo',
        'metrics': {
            'k5': metrics_k5,
            'k10': metrics_k10
        }
    }
    
    output_file = 'data/preprocessed/demo_evaluation.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to {output_file}")
    
    # Interpretation
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    print(f"""
Precision@10 = {metrics_k10['precision@10']:.4f}
  → {metrics_k10['precision@10']*100:.2f}% of recommendations are relevant
  
Recall@10 = {metrics_k10['recall@10']:.4f}
  → We found {metrics_k10['recall@10']*100:.2f}% of user's future interactions
  
Hit Rate@10 = {metrics_k10['hit_rate@10']:.4f}
  → {metrics_k10['hit_rate@10']*100:.1f}% of users got at least one relevant recommendation
  
Coverage@10 = {metrics_k10['catalog_coverage@10']:.4f}
  → We recommended {metrics_k10['catalog_coverage@10']*100:.2f}% of the catalog
  
Novelty@10 = {metrics_k10['novelty@10']:.4f}
  → Higher = more novel (less popular) recommendations
  
Diversity@10 = {metrics_k10['diversity@10']:.4f}
  → {metrics_k10['diversity@10']*100:.1f}% are unique songs (not versions)
    """)
    
    print("="*70)
    print("✅ EVALUATION COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()

