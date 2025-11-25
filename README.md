Directory structure

recsys_project/
│
├── README.md
├── requirements.txt
├── .env
├── .gitignore
│
├── data/
│   ├── raw/                # unmodified source data
│   ├── preprocessed/       # cleaned data
│   ├── interim/            # temporary files (splits, intermediate data)
│   └── external/           # external datasets (e.g., embeddings)
│
├── notebooks/              # Jupyter notebooks for exploration & prototyping
│   ├── 01_EDA.ipynb
│   ├── 02_FeatureEngineering.ipynb
│   └── 03_Modeling.ipynb
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/               # data loading & preprocessing modules
│   │   ├── __init__.py
│   │   ├── load_data.py
│   │   ├── preprocess.py
│   │   └── split.py
│   │
│   ├── features/           # feature engineering & transformations
│   │   ├── __init__.py
│   │   ├── build_features.py
│   │   └── encoders.py
│   │
│   ├── models/             # model definitions, training, inference
│   │   ├── __init__.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── predict.py
│   │   ├── matrix_factorization.py
│   │   ├── neural_recommender.py
│   │   └── metrics.py
│   │
│   ├── utils/              # helper functions
│   │   ├── __init__.py
│   │   ├── logging.py      # custom logging setup
│   │   └── config.py       # global config
│   │
│   └── api/
│       ├── __init__.py
│       ├── main.py         # FastAPI app
│       └── schemas.py      # request/response models
│
├── models/                 # saved model weights
│   ├── latest/
│   └── experiments/
│
├── experiments/            # experiment tracking
│   ├── experiment_001/
│   ├── experiment_002/
│   └── results.csv
│
├── tests/                  # unit & integration tests
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_models.py
│   └── test_api.py
│
├── config/                 # YAML configs (hydra, training configs, parameters)
│   ├── default.yaml
│   ├── model.yaml
│   └── training.yaml
│
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
│
└── scripts/
    ├── train.sh
    ├── run_api.sh
    └── preprocess.sh




# Подготовка виртуальной машины

## Склонируйте репозиторий

Склонируйте репозиторий проекта:

```
git clone https://github.com/yandex-praktikum/mle-project-sprint-4-v001.git
```

## Активируйте виртуальное окружение

Используйте то же самое виртуальное окружение, что и созданное для работы с уроками. Если его не существует, то его следует создать.

Создать новое виртуальное окружение можно командой:

```
python3 -m venv env_recsys_start
```

После его инициализации следующей командой

```
. env_recsys_start/bin/activate
```

установите в него необходимые Python-пакеты следующей командой

```
pip install -r requirements.txt
```

### Скачайте файлы с данными

Для начала работы понадобится три файла с данными:
- [tracks.parquet](https://storage.yandexcloud.net/mle-data/ym/tracks.parquet)
- [catalog_names.parquet](https://storage.yandexcloud.net/mle-data/ym/catalog_names.parquet)
- [interactions.parquet](https://storage.yandexcloud.net/mle-data/ym/interactions.parquet)
 
Скачайте их в директорию локального репозитория. Для удобства вы можете воспользоваться командой wget:

```
wget https://storage.yandexcloud.net/mle-data/ym/tracks.parquet

wget https://storage.yandexcloud.net/mle-data/ym/catalog_names.parquet

wget https://storage.yandexcloud.net/mle-data/ym/interactions.parquet
```

## Запустите Jupyter Lab

Запустите Jupyter Lab в командной строке

```
jupyter lab --ip=0.0.0.0 --no-browser
```

# Расчёт рекомендаций

Код для выполнения первой части проекта находится в файле `recommendations.ipynb`. Изначально, это шаблон. Используйте его для выполнения первой части проекта.

# Сервис рекомендаций

Код сервиса рекомендаций находится в файле `recommendations_service.py`.

<*укажите здесь необходимые шаги для запуска сервиса рекомендаций*>

# Инструкции для тестирования сервиса

Код для тестирования сервиса находится в файле `test_service.py`.

<*укажите здесь необходимые шаги для тестирования сервиса рекомендаций*>
