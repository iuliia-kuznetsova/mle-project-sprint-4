# Music Recommendation System

End-to-end system that builds personalized music recommendationsand serves them with a microservices architecture.

# Bucket name
s3-student-mle-20250101-65b9b79fea

## Project Structure
mle-project-sprint-4/
├── .venv_recsys/                          # Virual environment
├── catboost_info/                         # CatBoost training metadata
├── config/
│   ├── .env                               # Environment variables
│   └── requirements.txt                   # Python dependencies
├── data/
│   ├── raw/                               # Raw data from S3
│   │   ├── tracks.parquet
│   │   ├── catalog_names.parquet
│   │   └── interactions.parquet
│   └── preprocessed/                      # Cleaned & transformed data
│       ├── items.parquet
│       ├── tracks_catalog_clean.parquet
│       ├── events.parquet
│       ├── train_events.parquet
│       ├── test_events.parquet
│       ├── train_matrix.npz
│       ├── test_matrix.npz
│       └── train_test_split_info.pkl
├── logs/                                  # JSON logs for each pipeline step
│   ├── logs_main_recommendations.json
│   ├── logs_raw_data_loading.json
│   ├── logs_data_preprocessing.json
│   ├── logs_train_test_split.json
│   ├── logs_popularity_based_model.json
│   ├── logs_als_model.json
│   ├── logs_similarity_based_model.json
│   ├── logs_rec_ranking.json
│   ├── logs_rec_evaluation.json
│   ├── logs_main_services.py             # Logs of launching services script
│   └── logs_test_services.py             # Logs of testing services script
├── models/                               # Trained model artifacts
│   ├── als_model.pkl 
│   ├── catboost_classifier.cbm 
│   └── label_encoders.pkl 
├── notebooks/                            # Jupyter notebooks for data and results exploration
│   ├── data_overview.ipynb               # Raw data overview
│   ├── eda.ipynb                         # Exploratory data analysis
│   └── results.ipynb                     # Results overview, determination of the best model
├── results/                              # Model outputs & evaluation results
│   ├── evaluation_als.json               # Metrics of ALS recommender
│   ├── evaluation_popularity.json        # Metrics of popularity_based recommender
│   ├── evaluation_ranked.json            # Metrics of CatBoost ranking model
│   ├── feature_importances.parquet       # Feature importances of CatBoost ranking model
│   ├── feature_importances.png 
│   ├── models_comparison.parquet         # Models metrics
│   ├── personal_als.parquet              # ALS_based recommendations
│   ├── popularity_track_scores.parquet 
│   ├── recommendations.parquet           # Personal recommendations (offline recommendations)
│   ├── similar_tracks_index.pkl 
│   ├── similar.parquet                   # Similarity_based recommendations (online recommendations)
│   └── top_popular.parquet               # Popularity_based recommendations
├── src/                                  # Source code
│   ├──_init_.py
│   ├── logging_set_up.py                 # Logging configuration
│   ├── s3_testing_connection.py          # Testing connection to S3
│   ├── microservice
│   │   ├── _init_.py
│   │   ├── events.py                     # Add or get online events
│   │   ├── final_recs.py                 # Finding final recommendations
│   │   ├── main_services.py              # Starting services main pipeline
│   │   ├── offline_recs.py               # Get offline recommendations
│   │   ├── online_recs.py                # Get online recommendations
│   │   ├── test_services.py              # Test microservice
│   ├── recommendations
│   │   ├── _init_.py
│   │   ├── main_recommendations.py       # Finding recommendations main pipeline
│   │   ├── s3_loading.py                 # Saving to S3 configuration
│   │   ├── raw_data_loading.py           # Step 2: Download raw data
│   │   ├── data_preprocessing.py         # Step 3: Data preprocessing
│   │   ├── train_test_split.py           # Step 4: Train/test split
│   │   ├── popularity_based_model.py     # Step 5: Popularity model
│   │   ├── als_model.py                  # Step 6: ALS model
│   │   ├── similarity_based_model.py     # Step 7: Similarity model
│   │   ├── rec_ranking.py                # Step 8: CatBoost ranking
│   │   ├── rec_evaluation.py             # Step 9: Model evaluation
│   │   └── preprocessed_data_loading.py  # Preprocessed data loading
├── .gitignore
├── pytest.ini                        
└── README.md                             # Short project overview


# ---------- PART 1: Recommendations ----------#

## Finding Recommendations Pipeline Steps

### Step 1: Load Environment Variables
Loads configuration from `.env` file including paths to data directories
and S3 credentials for data access.

### Step 2: Download Raw Data
Downloads raw datasets from S3 storage:
- tracks.parquet — track metadata (features, duration, etc.)
- catalog_names.parquet — artist/album names mapping
- interactions.parquet — user-track interaction events
Can be skipped with `--skip-download` flag if data already exists locally.

### Step 3: Preprocess Data
Cleans and transforms raw data:
- Explodes dataframes
- Cleans and deduplicates tracks, artists, albums, genres names
- Filters invalid/missing entries
- Adds `track_group_id` indicator of tracks relation (original track, its versions, remixes, covers have one unique`track_group_id`, but different `track_id`) 
- Encodes categorical features (user ids, track ids)
- Outputs: `items.parquet`, `tracks_catalog_clean.parquet`, `events.parquet`

### Step 4: Split Data into Train/Test Sets
Splits interaction data chronologically:
- Creates train/test event dataframes by splitting on specified date_threshold or as quantile 
- Builds sparse user-item matrices (`train_matrix.npz`, `test_matrix.npz`)
- Ensures no data leakage between train and test periods

### Step 5: Popularity-Based Recommendations
Builds a baseline model using track popularity:
- Ranks tracks by total interaction count
- - Uses all train data
- Generates global "most popular" recommendations
- Output: `top_popular.parquet`, `popularity_track_scores.parquet`

### Step 6: ALS Recommendations
Trains Alternating Least Squares (ALS) collaborative filtering model:
- Learns latent user and item factors from interaction matrix
- Uses all train data for training
- Generates personalized recommendations per user for all train users_ids 
- Output: `als_model.pkl`, `personal_als.parquet`

### Step 7: Similarity-Based Recommendations
Computes item-item similarity:
- Uses pretrained ALS model, builds similarity index for all track_ids
- Results will be used for online recommendations, thus excluded from following steps (ranking and evaluation)
- Output: `similar.parquet`, `similar_tracks_index.pkl`

### Step 8: Ranking (CatBoost)
Trains a CatBoost classifier to re-rank candidates:
- Computes track-specific custom features: 
genre popularity, artist popularity, track popularity (by number of ralated tracks)
- Combines custom features with features from popularity_based and ALS models
- Uses all train data for training
- Learns to predict user-item relevance
- Output: `catboost_classifier.cbm`, `recommendations.parquet`

### Step 9: Evaluate Models
Computes evaluation metrics for all models:
- Computes popularity_based, ALS model and ranked model recommendations for test user_ids
- Uses all test data for evaluation or sample test users
- Computes metrics (Precision@K, Recall@K, MAP, NDCG) of each model
- Compares popularity, ALS and ranked models
- Output: `evaluation_*.json`, `models_comparison.parquet`


## Usage
### Finding recommendations

1. Create virtual environment
- install extension
```bash
sudo apt-get install python3.10-venv
```
- create .venv
```bash
python3 -m venv .venv_recsys
```
- run .venv
```bash
source .venv_recsys/bin/activate
```
- install packages
```bash
pip install -r config/requirements.txt
```

2. Run pipeline
Run full pipeline
```bash
python3 -m src.main
```
or

skip data download (if raw data already exists)
```bash
python3 -m src.main --skip-download
```

3. Check Data Overview, EDA or Results
Take a look at 
notebooks/data_overview.ipynb
notebooks/eda.ipynb
notebooks/results.ipynb
Manually choose kernel Python(.venv_recsys)
Run all

4. Check logs
Take a look at 
logs/logs_main_recommendations.py # Main pipeline logs
and separate scripts log files 

# ---------- PART 2: Microservice ----------#
## Running Microservice Pipeline Steps

### Step 1: Offline recommendations
Get offline recommendations for a given user_id:
- Get precomputed personal recommendations (from recommendations.parquet)

### Step 2: Events
Add or get online events for a given user_id:
- Add information of user's recently listened track_id
- Get information of user's recently listened track_id

### Step 3: Online recommendations
Get online recommendations for a given user_id:
- Get similar tracks to user's the currently listened tracks in events 

### Step 4: Final recommendations
Get final (blended) recommendations for a given user_id so that:
- Deduplicates track_ids deduplicated
- offline recommendations are at the even places of final recommendations
- online recommendations are at the odd places of final recommendations
  

## Usage
### Running microservice

1. Launch services
```bash
python3 -m src.microservice.main_services
```
This should start:
Main API (final recommendations) on port 8000
Offline recommendations service on port 8001
Events service on port 8002
Online/similarity service on port 8003

2. Test services
```bash
python3 -m src.microservice.test_services
```   

3. Check logs
Take a look at
logs/logs_main_services.json
logs/logs_test_services.json

4. Check services status
Open these URLs in a browser:
http://localhost:8000/healthy
http://localhost:8001/healthy
http://localhost:8002/healthy
http://localhost:8003/healthy

5. Open interactive docs (Swagger UI) and try endpoints
Open these URLs in a browser:
http://localhost:8000/docs  # Final recommendations
http://localhost:8001/docs  # Offline recommendations
http://localhost:8002/docs  # Events
http://localhost:8003/docs  # Online recommendations

6. Get recommendations for a user_id with curl
- Final blended recommendations for a user
```bash
curl -X POST "http://localhost:8000/recommendations?user_id=123&k=10"
```
- Offline-only recommendations
```bash
curl -X POST "http://localhost:8001/get_recs?user_id=123&k=10"
```
- Add events (user listening history)
```bash
curl -X POST "http://localhost:8002/put?user_id=123&track_id=456"
```
- Get user events
```bash
curl -X POST "http://localhost:8002/get?user_id=123&k=10"
```
- Get similar tracks for a given track
```bash
curl -X POST "http://localhost:8003/similar_tracks?track_id=456&k=10"
```