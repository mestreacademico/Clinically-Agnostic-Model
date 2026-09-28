import sys
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import RandomizedSearchCV
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import roc_auc_score, recall_score, precision_score, precision_recall_curve, auc, roc_curve
import numpy as np
import pandas as pd
import pickle
import os
from datetime import datetime 
import zipfile
import requests

####################################
# Output

OUTPUT_FOLDER = './outputPaper/'
NOME_DO_MODELO = "CB_ENRICHMENT_PAPER"
if not os.path.isdir(OUTPUT_FOLDER):
    os.mkdir(OUTPUT_FOLDER)
OUTPUT_FOLDER = f"{OUTPUT_FOLDER}{NOME_DO_MODELO}/"
if not os.path.isdir(OUTPUT_FOLDER):
    os.mkdir(OUTPUT_FOLDER)

N_ITER = 300 

NUM_JOBS = 20 

TRAIN_INFO = [
    202007,
    202008,
    202009,
    202010,
    202011,
    202012,
    202101,
    202102,
    202103,
    202104,
    202105,
    202106,
    202107,                               
    202108,
    202109,
    202110,
    202111,
    202112,
    202201,
    202202,
    202203,
    202204,
    202205,
    202206,
    202207,
    202208,
    202209,
    202210,
    202211,
    202212,
    202301,
    202302,
]

TEST_INFO = [ 
    202303,
    202304,
    202305,
    202306,
    202307,
    202308,
    202309,
    202310,
    202311,
    202312,
    202401,                               
]

inicio = datetime.now()

# download dataset
DATA_URL = "https://zenodo.org/records/22998650/files/datasets.zip?download=1"

DATA_DIR = "data"
TARGET_CSV = "dataset_model_enrichment_public.csv" 

CSV_PATH = os.path.join(DATA_DIR, TARGET_CSV)
ZIP_PATH = os.path.join(DATA_DIR, "datasets.zip")


def load_data(url: str, csv_filename: str) -> pd.DataFrame:
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(CSV_PATH):
        print(f"Loading data '{CSV_PATH}' ...")
        return pd.read_csv(CSV_PATH)

    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        with open(ZIP_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    print(f"Extracting '{csv_filename}' in ZIP...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extract(csv_filename, path=DATA_DIR)

    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)
    
    print("Dataset loading...")
    return pd.read_csv(CSV_PATH)


if __name__ == "__main__":
    df_full = load_data(DATA_URL, TARGET_CSV)
    print(f"Done! Shape: {df_full.shape}")

# ----------

TARGET = 'Profile4' 
CROP_FEATURE = 'Profile3'

features_drop = [
    'Profile1',
    'Profile2',
    'Profile85',
    CROP_FEATURE,
]

features_output = features_drop + [TARGET]

# Getting dummies
df_features_drop = df_full[features_drop + ['Profile5']]
df_full_dummies = pd.get_dummies(df_full.drop(features_drop, axis=1))
df_full_dummies = pd.merge(left=df_full_dummies, right=df_features_drop, left_on=['Profile5'], right_on=['Profile5'], how='left')
del df_features_drop
del df_full


features_drop = features_drop + ['Profile5']
features_output = features_drop + [TARGET]

## train and test
print("Train x Test ...")
train, test = df_full_dummies[df_full_dummies[CROP_FEATURE].isin(TRAIN_INFO)],\
            df_full_dummies[df_full_dummies[CROP_FEATURE].isin(TEST_INFO)]

train_o, train = train[features_output], train.drop(features_drop, axis=1)
test_o, test = test[features_output], test.drop(features_drop, axis=1)

train_x, train_y = train.drop(TARGET, axis=1), train[TARGET]
test_x, test_y = test.drop(TARGET, axis=1), test[TARGET]

train = train_o, train_x
test = test_o, test_x

print("RUS ...")
rus = RandomUnderSampler(random_state=42, sampling_strategy = 0.20)
train_x, train_y = rus.fit_resample(train_x, train_y)

print("Model CATBOOST\n")

best_params = {
    'iterations': 500,
    'learning_rate': 0.01,  
    'depth': 6,              
    'l2_leaf_reg': 3,        
    'border_count': 64      
}

print("Training with best CatBoost parameters...")
catboost_model = CatBoostClassifier(
    **best_params,
    task_type='GPU',  
    random_seed=42,
    loss_function='Logloss',
    thread_count=24        
)

catboost_model.fit(train_x, train_y)

test_crops = df_full_dummies[df_full_dummies[CROP_FEATURE].isin(TEST_INFO)]

# ------ Metrics AUC, precision e Recall -----------

test_crops_features = test_crops.drop(columns=features_drop + [TARGET])

# test
y_proba = catboost_model.predict_proba(test_crops_features)[:, 1]  
y_pred = catboost_model.predict(test_crops_features)

auc_score = roc_auc_score(test_crops[TARGET], y_proba)
recall = recall_score(test_crops[TARGET], y_pred)
precision = precision_score(test_crops[TARGET], y_pred)

# ---  Metric KS ---
fpr, tpr, thresholds = roc_curve(test_crops[TARGET], y_proba)
ks_score = max(tpr - fpr)

ks_group = []

test_crops['y_proba'] = y_proba

for grupo, df_sub in test_crops.groupby('Profile3'):
    if df_sub[TARGET].nunique() < 2:
        continue
        
    fpr, tpr, _ = roc_curve(df_sub[TARGET], df_sub['y_proba'])
    ksGroup = max(tpr - fpr)
    ks_group.append(ksGroup)

ks_medio = np.mean(ks_group)

# AUCPR 
precision_vals, recall_vals, _ = precision_recall_curve(test_crops[TARGET], y_proba)
aucpr_score = auc(recall_vals, precision_vals)
    
# Results
print(f"AUC: {auc_score:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Precision: {precision:.4f}")
print(f"AUC PR: {aucpr_score}")
print(f"KS Average: {ks_medio:.4f}")
    
with open(f"{OUTPUT_FOLDER}model_metrics_CB.txt", 'w') as f:
    f.write(f"AUC: {auc_score:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"AUC PR: {aucpr_score:.4f}\n")
    f.write(f"KS Average: {ks_medio:.4f}\n")
    

del test_crops
fim2 = datetime.now()
duracao2 = fim2 - inicio
print("Time:", duracao2)
print("done!\n\n")
