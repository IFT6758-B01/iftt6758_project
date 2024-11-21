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
#from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.calibration import CalibrationDisplay

# taking this out cuz it will also run the xgboost in the other file
#from xgboost_simple_wandb import roc_curve_and_auc, goal_rate_by_percentile, cumulative_proportion_of_goals, reliability_diagram
import matplotlib.pyplot as plt

# Instead including the plotting function here
## 1. ROC Curve and AUC
def roc_curve_and_auc(y_val, y_prob, log_to_run=None):
    fpr, tpr, _ = roc_curve(y_val, y_prob)
    roc_auc = auc(fpr, tpr)

    # Plot ROC Curve
    fig = plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f"XGBoost Classifier (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier (AUC = 0.50)")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    if log_to_run:
        if wandb.run is not None:
            run.log({'roc_auc': plt})
            return
    plt.show()
       
## 2. Goal Rate by Percentile (Binned by 5%)
def goal_rate_by_percentile(y_val, y_prob, log_to_run=None):
    df_val = pd.DataFrame({'y_val': y_val, 'y_prob': y_prob})
    df_val['percentile'] = pd.qcut(df_val['y_prob'], 100, labels=False, duplicates='drop') + 1  # Percentiles from 1 to 100
    goal_rate_by_percentile = df_val.groupby('percentile')['y_val'].mean()

    fig = plt.figure(figsize=(10, 6))
    plt.plot(goal_rate_by_percentile.index, goal_rate_by_percentile, marker='o')
    plt.title("Goal Rate by Percentile")
    plt.xlabel("Model Percentile")
    plt.ylabel("Goal Rate (#goals / (#goals + #no_goals))")
    if log_to_run:
        if wandb.run is not None:
            run.log({'goal_rate_percentile': plt})
            return
    plt.show()
    
## 3. Cumulative Proportion of Goals by Percentile
def cumulative_proportion_of_goals(y_val, y_prob, log_to_run=None):
    df_val = pd.DataFrame({'y_val': y_val, 'y_prob': y_prob})
    cumulative_goals = df_val.sort_values('y_prob', ascending=False)['y_val'].cumsum()
    total_goals = df_val['y_val'].sum()
    cumulative_goal_percentage = cumulative_goals / total_goals

    fig = plt.figure(figsize=(10, 6))
    plt.plot(np.linspace(0, 1, len(cumulative_goal_percentage)), cumulative_goal_percentage, marker='o')
    plt.title("Cumulative Proportion of Goals by Model Percentile")
    plt.xlabel("Model Percentile")
    plt.ylabel("Cumulative Proportion of Goals")
    if log_to_run:
        if wandb.run is not None:
            run.log({'cumul_goals': plt})
    plt.show()

# 4. Reliability Diagram (Calibration Curve)
def reliability_diagram(y_val, y_prob, log_to_run=None):
    CalibrationDisplay.from_predictions(y_val, y_prob, n_bins=30, strategy='uniform')
    plt.title("Reliability Diagram (Calibration Curve)")
    if log_to_run:
        log_to_run.log({f"Reliability Diagram": wandb.Image(plt)})
    '''
    if log_to_run:
        if wandb.run is not None:
            run.log({'reliability_diagram': plt})
            return
    '''
    plt.show()
    plt.close()

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
run = wandb.init(entity="IFT6758_2024-B01" ,project="ms2-xgboost-models", name="xgboost_complex_no_fillna")

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
df_percent_nan = (df.isnull().sum() / df.count()).sort_values(ascending=False).loc[lambda x : x > 1.0]
if not df_percent_nan.empty:
    print('[WARNING] The following features have lots of NaN values')
    print(df_percent_nan)
    print('Current method of inputation is replacing with mean')
## =====> THIS IS CURRENTLY FAILING <=====
# X.fillna(X.mean(), inplace=True)

# Let XGBoost handle NaN values
if X.isnull().sum().sum() > 0:
    print('[INFO] XGBoost will handle NaN values natively.')

y = df['is_goal']
y.fillna(0, inplace=True)

# Initialize LabelEncoders
shot_type_encoder = preprocessing.LabelEncoder()
last_event_type_encoder = preprocessing.LabelEncoder()

# Encode "shot_type"
X['shot_type'] = shot_type_encoder.fit_transform(X['shot_type'].fillna("Unknown"))

# Encode "last_event_type"
X['last_event_type'] = last_event_type_encoder.fit_transform(X['last_event_type'].fillna("Unknown"))

# Transform data to Dmatrix data structure
#data_dmatrix = xgb.DMatrix(data=X, label=y)

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

'''
# LabelEncode non-numeric features
# From https://www.kaggle.com/code/phunter/xgboost-with-gridsearchcv
# Iterate through columns of X_train
for f in X.columns:
    if X[f].dtype=='object':
        print(f)
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(X_train[f].values) + list(X_val[f].values))
        X_train[f] = lbl.transform(list(X_train[f].values))
        X_val[f] = lbl.transform(list(X_val[f].values))
'''

# Transform data to DMatrix data structure so that we can use xgb.cv
# Reference: https://datascience.stackexchange.com/questions/12799/pandas-dataframe-to-dmatrix
dtrain = xgb.DMatrix(data=X_train.values, label=y_train.values)
dval = xgb.DMatrix(data=X_val.values, label=y_val.values)

params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 4,
    'learning_rate': 0.1,
    'alpha': 10,
    'n_estimators': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8
}

# Cross-validation (might also be used for hyperparameter tuning tho I need to read the docs)
cv_results = xgb.cv(
    params=params,
    dtrain=dtrain,
    num_boost_round=100,
    nfold=5,
    early_stopping_rounds=10,
    metrics="logloss",
    seed=42
)

# Log best score and iteration
best_iteration = len(cv_results)
best_score = cv_results['test-logloss-mean'].min()
print(f"Best Iteration: {best_iteration}, Best Log Loss: {best_score:.4f}")

# Train final model with best iteration
params['n_estimators'] = best_iteration
xgb_clf = xgb.train(params, dtrain, num_boost_round=best_iteration)

# Save the model
model_basename = 'xgb_classifier_cv.json'
model_path = pathlib.Path.cwd() / model_basename
xgb_clf.save_model(fname=model_path)

# After training the XGBoost model, predict probabilities for the validation set
y_val_pred_prob = xgb_clf.predict(dval)  # dval is the validation set in DMatrix format

# Plotting and logging to WandB
roc_curve_and_auc(y_val, y_val_pred_prob, log_to_run=run)
goal_rate_by_percentile(y_val, y_val_pred_prob, log_to_run=run)
cumulative_proportion_of_goals(y_val, y_val_pred_prob, log_to_run=run)
reliability_diagram(y_val, y_val_pred_prob, log_to_run=run)

'''
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
'''

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
