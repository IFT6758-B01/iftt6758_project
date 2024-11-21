#!/usr/bin/env python3

import wandb
import pandas as pd
import numpy as np
import xgboost as xgb
import pathlib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.calibration import CalibrationDisplay
import matplotlib.pyplot as plt


# Figure plots functions
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
    df_val = pd.DataFrame({'y_val': y_val, 'y_prob': y_prob})
    CalibrationDisplay.from_predictions(y_val, y_prob, n_bins=10, strategy='uniform')
    plt.title("Reliability Diagram (Calibration Curve)")
    if log_to_run:
        if wandb.run is not None:
            run.log({'reliability_diagram': plt})
            return
    plt.show()


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
features = ['distance_from_net', 'angle_from_net']
X = df[features]
X.fillna(X.mean(), inplace=True)
y = df['is_goal']
y.fillna(0, inplace=True)

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost model instance
xg_clf = xgb.XGBClassifier()
xg_clf.fit(X_train, y_train)

# XGBoost model predictions
y_pred = xg_clf.predict(X_val)
y_probas = xg_clf.predict_proba(X_val)

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
roc_curve_and_auc(y_val, y_probas[:, 1])
goal_rate_by_percentile(y_val, y_probas[:, 1])
cumulative_proportion_of_goals(y_val, y_probas[:, 1])
'''