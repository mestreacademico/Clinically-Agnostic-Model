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

@pd.DataFrame
def load_data(url: str) -> pd.DataFrame:
    print("Readind dataset...")
    return pd.read_csv(url)

if __name__ == "__main__":
    df = load_data(DATA_URL)
    print(f"Done: {df.shape[0]} linhas e {df.shape[1]} colunas.")
    
#df_full = pd.read_csv('dataset_model_enrichment_public.csv', engine='pyarrow')
#print(df_full.shape)

TARGET = 'Profile4' 
CROP_FEATURE = 'Profile3'

features_drop = [
    'Profile1',
    'Profile2',
    'Profile85',
    CROP_FEATURE,
]

features_output = features_drop + [TARGET]

print("Getting dummies..")
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

# CatBoost
params_cat = {
    'iterations': [500],             
    'learning_rate': [0.01, 0.05],
    'depth': [4, 6],
    'l2_leaf_reg': [1, 3, 5],
    'border_count': [32, 64],
}

catboost_model = CatBoostClassifier(
    task_type='GPU',  
    random_seed=42,
    loss_function='Logloss',
    thread_count=24        
)

print("Model CATBOOST\n")
print("Starting CatBoost grid search...")

train_data = Pool(data=train_x, label=train_y)

search = RandomizedSearchCV(
    estimator=catboost_model,
    param_distributions=params_cat,
    n_iter=50,  
    cv=3,  
    verbose=2,
    random_state=42,
    n_jobs=1  
)

search.fit(train_x, train_y)

print('Best CatBoost parameters: {}'.format(search.best_params_))
  
with open(f"{OUTPUT_FOLDER}best_estimator_CB.txt", 'w') as f:
    f.write(f"Best Params: {search.best_params_}\n")

catboost_model = search.best_estimator_


test_crops = df_full_dummies[df_full_dummies[CROP_FEATURE].isin(TEST_INFO)]

# ------ Metrics AUC, precision e Recall -----------

test_crops_features = test_crops.drop(columns=features_drop + [TARGET])

#test_crops_features.to_csv(f"{OUTPUT_FOLDER}test_crops_features.csv", index=False)

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
print(f"KS Mean: {ks_medio:.4f}")
print(f"KS: {ks_score * 100:.2f}%)")
    
with open(f"{OUTPUT_FOLDER}model_metrics_CB.txt", 'w') as f:
    f.write(f"AUC: {auc_score:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"AUC PR: {aucpr_score:.4f}\n")
    f.write(f"KS Mean: {ks_medio:.4f}\n")
    

del test_crops
fim2 = datetime.now()
duracao2 = fim2 - inicio
print("Time:", duracao2)
print("done!\n\n")
