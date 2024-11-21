#!/usr/bin/env python3

#TODO:
# -Figure out how to deal with NaN values in X (see lines 79-80)
# -LabelEncode non-numerical features in X ?
# -Log relevant figures to WandB

import wandb
import pandas as pd
import numpy as np
import xgboost as xgb
import pathlib
import os
from sklearn import preprocessing
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.calibration import CalibrationDisplay
from xgboost_simple-wandb import roc_curve_and_auc, goal_rate_by_percentile, cumulative_proportion_of_goals, reliability_diagram
import matplotlib.pyplot as plt



# Set paths
try:
    current_dirpath = pathlib.Path(__file__).parent.absolute().resolve()
except NameError:
    current_dirpath = pathlib.Path(os.path.curdir).absolute().resolve()

if not current_dirpath.parts[-3:] == ('ift6758', 'advanced_models', 'xgboost'):
    raise Exception(
        'It appears that this file is executed from the wrong location\n'
        'Expected path: (root-->)ift6758/advanced_models/xgboost/\n'
        f'Current path: {current_dirpath}'
    )
root_dirpath = current_dirpath.parents[1]

# Load the dataset
dataset_path = (root_dirpath / 'dataset' / 'complex_engineered' / 'augmented_data.csv')
if not (dataset_path.is_file() and dataset_path.match('*.csv')):
    raise Exception(
        'It appears that the dataset either does not exist or is not a valid CSV\n'
        f'Path: {dataset_path}'
    )
df = pd.read_csv(dataset_path)

# Initialize WandB run
run = wandb.init(entity="IFT6758_2024-B01" ,project="ms2-xgboost-models")


# Feature selection (basic) and NaN values processing
features = [
    'distance_from_net',
    'angle_from_net',
    'game_seconds',
    'period',
    'x_coord',
    'y_coord',
    'distance_from_net',
    'angle_from_net',
    'shot_type',
    'last_event_type',
    'last_x_coord',
    'last_y_coord',
    'time_from_last_event',
    'distance_from_last_event',
    'rebound',
    'change_in_shot_angle',
    'speed'
]

X = df[features]
# Check if some features have more than 1% (arbritrary) of NaN values
df_percent_nan = (df.isnull().sum() / df.count()).sort_values(ascending=False).loc[lambda x : x > 1.0])
if not df_percent_nan.empty:
    print('[WARNING] The following features have lots of NaN values')
    print(df_percent_nan)
    print('Current method of inputation is replacing with mean')
## =====> THIS IS CURRENTLY FAILING <=====
X.fillna(X.mean(), inplace=True)

y = df['is_goal']
y.fillna(0, inplace=True)


# Transform data to Dmatrix data structure
#data_dmatrix = xgb.DMatrix(data=X, label=y)

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# LabelEncode non-numeric features
# From https://www.kaggle.com/code/phunter/xgboost-with-gridsearchcv
for f in X.columns:
    if X[f].dtype=='object':
        print(f)
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(X_train[f].values) + list(X_val[f].values))
        X_train[f] = lbl.transform(list(X_train[f].values))
        X_test[f] = lbl.transform(list(X_test[f].values))

# Instanciate XGBoost Classifier
xg_clf = xgb.XGBClassifier()

# Set hyperparameters
params = {
    'objective':'binary:logistic',
    'max_depth': 4,
    'alpha': 10,
    'learning_rate': 1.0,
    'n_estimators':100
}

# Operate GridSearch CrossValidation
clf = GridSearchCV(xg_clf,
                   params,
                   n_jobs=5,
                   cv=StratifiedKFold(train['QuoteConversion_Flag'], n_folds=5, shuffle=True),
                   scoring='accuracy')

clf.fit(X_train, y_train)


#From https://www.kaggle.com/code/phunter/xgboost-with-gridsearchcv
best_parameters, score, _ = max(clf.grid_scores_, key=lambda x: x[1])
print('Accuracy score:', score)
for param_name in sorted(best_parameters.keys()):
    print("%s: %r" % (param_name, best_parameters[param_name]))

test_probs = clf.predict_proba(X_test)[:,1]

# Saving model to JSON
model_basename = 'xgb_classifier_complex.json'
model_path = current_dirpath / model_basename
clf.save_model(fname=model_path)

# Log model to WandB run
run.log_model(path=model_path, name=model_basename)

# End run
run.finish()

'''
# XGBoost model instance
xg_clf = xgb.XGBClassifier(**params)
xg_clf.fit(X_train, y_train)

# XGBoost model predictions
y_pred = xg_clf.predict(X_val)
y_probas = xg_clf.predict_proba(X_val)

# Print feature importance
xgb.plot_importance(xg_clf)
plt.figure(figsize=(10, 6))
run.log({'feature_importance': plt})

# Print accuracy
accuracy = accuracy_score(y_val, y_pred)
print(f"XGBoost Classifier Accuracy score: {accuracy}")
run.log({'accuracy': accuracy)

# Saving model to JSON
model_basename = 'xgb_classifier_simple.json'
model_path = current_dirpath / model_basename
xg_clf.save_model(fname=model_path)

# Log model to WandB run
run.log_model(path=model_path, name=model_basename)

# Log plots to WandB run
roc_curve_and_auc(y_val, y_probas[:, 1], log_to_run=run)
goal_rate_by_percentile(y_val, y_probas[:, 1], log_to_run=run)
cumulative_proportion_of_goals(y_val, y_probas[:, 1], log_to_run=run)
reliability_diagram(y_val, y_probas[:, 1], log_to_run=run)

# Random Baseline generation and log to WandB
y_probas_random = np.random.uniform(0, 1, len(y_val))

roc_curve_and_auc(y_val, y_probas_random, log_to_run=run)
goal_rate_by_percentile(y_val, y_probas_random, log_to_run=run)
cumulative_proportion_of_goals(y_val, y_probas_random, log_to_run=run)

# End run
run.finish()

'''
