#!/usr/bin/env python3

import wandb
import pathlib
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.calibration import CalibrationDisplay
import matplotlib.pyplot as plt
import joblib
from sklearn import preprocessing


#  plot functions
# 1. ROC Curve and AUC
def roc_curve_and_auc(y_val, y_prob, log_to_run=None):
    fpr, tpr, _ = roc_curve(y_val, y_prob)
    roc_auc = auc(fpr, tpr)

    # Plot ROC Curve
    plt.figure(figsize=(10, 6))
    
    plt.plot(fpr, tpr, label=f"Random Forest (AUC = {roc_auc:.2f})")  
    plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier (AUC = 0.50)")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.show()   
    if log_to_run:
        log_to_run.log({'roc_auc': plt})
    plt.close()
    

# 2. Goal Rate by Percentile (Binned by 5%)
def goal_rate_by_percentile(y_val, y_prob, log_to_run=None):
    df_val = pd.DataFrame({'y_val': y_val, 'y_prob': y_prob})
    df_val['percentile'] = pd.qcut(df_val['y_prob'], 100, labels=False, duplicates='drop') + 1  # Percentiles from 1 to 100
    goal_rate_by_percentile = df_val.groupby('percentile')['y_val'].mean()

    plt.figure(figsize=(10, 6))
    plt.plot(goal_rate_by_percentile.index, goal_rate_by_percentile, marker='o')
    plt.title("Goal Rate by Percentile")
    plt.xlabel("Model Percentile")
    plt.ylabel("Goal Rate (#goals / (#goals + #no_goals))")
    plt.show()   
    if log_to_run:
        log_to_run.log({'goal_rate_percentile': plt})
    plt.close()
    

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
    plt.show()  
    if log_to_run:
        log_to_run.log({'cumul_goals': plt})
    plt.close()
    

# 4. Reliability Diagram (Calibration Curve)
def reliability_diagram(y_val, y_prob, log_to_run=None):
    CalibrationDisplay.from_predictions(y_val, y_prob, n_bins=30, strategy='uniform')
    plt.title("Reliability Diagram (Calibration Curve)")
    plt.show()   
    if log_to_run:
        log_to_run.log({f"Reliability Diagram": wandb.Image(plt)})
    plt.close()
  




# Set paths
try:
    current_dirpath = pathlib.Path(__file__).parent.absolute().resolve()
except NameError:
    current_dirpath = pathlib.Path(os.path.curdir).absolute().resolve()

if not current_dirpath.parts[-3:] == ('ift6758', 'advanced_models', 'random_forest'):
    raise Exception(
        'It appears that this file is executed from the wrong location\n'
        'Expected path: (root-->)ift6758/advanced_models/random_forest/\n'
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
wandb.init(entity="IFT6758_2024-B01" ,project="ms2-random-forest-models")
run=wandb.run


# Dataset processing
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


y = df['is_goal']
y.fillna(0, inplace=True)

# Initialize LabelEncoders
shot_type_encoder = preprocessing.LabelEncoder()
last_event_type_encoder = preprocessing.LabelEncoder()


# # Encode "shot_type"
# df['shot_type'] = df['shot_type'].fillna("Unknown") 
# df['shot_type'] = shot_type_encoder.fit_transform(df['shot_type'])


# # Encode "last_event_type"
# df['last_event_type'] = df['last_event_type'].fillna("Unknown") 
# df['last_event_type'] = last_event_type_encoder.fit_transform(df['last_event_type'])
# Encode "shot_type"
X.loc[:,'shot_type'] = shot_type_encoder.fit_transform(X['shot_type'].fillna("Unknown"))

# Encode "last_event_type"
X.loc[:, 'last_event_type'] = last_event_type_encoder.fit_transform(X['last_event_type'].fillna("Unknown"))

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Model 
rf_clf =  RandomForestClassifier()

# Save the trained model to a file
joblib.dump(rf_clf, 'random_forest_model.pkl')
rf_clf.fit(X_train, y_train)
y_pred = rf_clf.predict(X_val)
y_probs = rf_clf.predict_proba(X_val)


# saving decisionTreeClassifer model to pkl file
model_basename = 'rf_classifier_complex.pkl'
model_path = pathlib.Path.cwd() / model_basename

# Save the model to a file 
joblib.dump(rf_clf, model_basename) 

# Upload the model file to Wandb
wandb.save(model_basename)

# Log model to WandB run
# Restore the model file from WandB
wandb.restore(model_basename)
# Load the model from the file
clf_loaded = joblib.load(model_basename)


# Log plots to WandB run
roc_curve_and_auc(y_val, y_probs[:,1],log_to_run=run)
goal_rate_by_percentile(y_val, y_probs[:,1],log_to_run=run)
cumulative_proportion_of_goals(y_val, y_probs[:,1],log_to_run=run)
reliability_diagram(y_val, y_probs[:,1],log_to_run=run)

# # Random Baseline generation and log to WandB
# y_probas_random = np.random.uniform(0, 1, len(y_val))


# End wandb
wandb.finish()


