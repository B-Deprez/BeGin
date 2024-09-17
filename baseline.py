import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, precision_recall_curve
import xgboost as xgb

import os

def train_test_split_custom(X, y, timebased=False, test_size=0.2, random_state=None, stratify=None):
    if timebased:
        n_obse = X.shape[0]
        n_test = int(round(n_obse * test_size))

        X_train = X.iloc[:-n_test]
        y_train = y.iloc[:-n_test]

        X_test = X.iloc[-n_test:]
        y_test = y.iloc[-n_test:]

    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)

    return X_train, X_test, y_train, y_test

# Load data
data = pd.read_csv('data/IBM/HI-Small_Trans_Patterns.csv')

# Pre-processing
# Convert the Timestamp column to datetime
data['Timestamp'] = pd.to_datetime(data['Timestamp'])

# Extract day of the week, hour, and minute
data['DayOfWeek'] = data['Timestamp'].dt.dayofweek
data['Hour'] = data['Timestamp'].dt.hour
data['Minute'] = data['Timestamp'].dt.minute

columns_X = [
    'DayOfWeek', 
    'Hour', 
    'Minute', 
    'From Bank', 
    'To Bank', 
    'Amount Paid', 
    'Payment Currency', 
    'Receiving Currency',
    'Payment Format'
    ]

targets = [
    'Is Laundering',
    'FAN-OUT', 
    'FAN-IN', 
    'GATHER-SCATTER', 
    'SCATTER-GATHER', 
    'CYCLE',
    'RANDOM', 
    'BIPARTITE', 
    'STACK'
    ]

data = data[columns_X + targets]
data = pd.get_dummies(data)

columns_X = list(data.columns.drop(targets))

X = data[columns_X]

fig_pr, ax_pr = plt.subplots(3, 3, figsize=(15, 15))
fig_auc, ax_auc = plt.subplots(3, 3, figsize=(15, 15))

for target in targets:
    y = data[target]

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split_custom(X, y, timebased=True, test_size=0.2)

    # Train a logistic regression model
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict_proba(X_test)[:, 1]

    print(f'{target} - Logistic Regression')
    print(f'ROC AUC: {roc_auc_score(y_test, y_pred_lr)}')
    print(f'Average Precision: {average_precision_score(y_test, y_pred_lr)}')

    # Train a random forest model
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict_proba(X_test)[:, 1]

    print(f'{target} - Random Forest')
    print(f'ROC AUC: {roc_auc_score(y_test, y_pred_rf)}')
    print(f'Average Precision: {average_precision_score(y_test, y_pred_rf)}')

    # Train an XGBoost model
    xgb_model = xgb.XGBClassifier()
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict_proba(X_test)[:, 1]

    print(f'{target} - XGBoost')
    print(f'ROC AUC: {roc_auc_score(y_test, y_pred_xgb)}')
    print(f'Average Precision: {average_precision_score(y_test, y_pred_xgb)}')

    precision_lr, recall_lr, _ = precision_recall_curve(y_test, y_pred_lr)
    precision_rf, recall_rf, _ = precision_recall_curve(y_test, y_pred_rf)
    precision_xgb, recall_xgb, _ = precision_recall_curve(y_test, y_pred_xgb)

    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_lr, pos_label=1)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf, pos_label=1)
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_pred_xgb, pos_label=1)

    ax_pr[targets.index(target) // 3, targets.index(target) % 3].plot(recall_lr, precision_lr, label='Logistic Regression: AUC-PR = {:.3f}'.format(average_precision_score(y_test, y_pred_lr)))
    ax_pr[targets.index(target) // 3, targets.index(target) % 3].plot(recall_rf, precision_rf, label='Random Forest: AUC-PR = {:.3f}'.format(average_precision_score(y_test, y_pred_rf)))
    ax_pr[targets.index(target) // 3, targets.index(target) % 3].plot(recall_xgb, precision_xgb, label='XGBoost: AUC-PR = {:.3f}'.format(average_precision_score(y_test, y_pred_xgb)))
    ax_pr[targets.index(target) // 3, targets.index(target) % 3].plot([0, 1], [y_test.mean(), y_test.mean()], linestyle='--', label='Chance level: AUC-PR = %0.3f' % y_test.mean(), color='black', alpha=0.5)
    ax_pr[targets.index(target) // 3, targets.index(target) % 3].set_title(target)
    ax_pr[targets.index(target) // 3, targets.index(target) % 3].set_xlabel('Recall')
    ax_pr[targets.index(target) // 3, targets.index(target) % 3].set_ylabel('Precision')
    ax_pr[targets.index(target) // 3, targets.index(target) % 3].legend()

    ax_auc[targets.index(target) // 3, targets.index(target) % 3].plot(fpr_lr, tpr_lr, label='Logistic Regression: AUC-ROC = {:.3f}'.format(roc_auc_score(y_test, y_pred_lr)))
    ax_auc[targets.index(target) // 3, targets.index(target) % 3].plot(fpr_rf, tpr_rf, label='Random Forest: AUC-ROC = {:.3f}'.format(roc_auc_score(y_test, y_pred_rf)))
    ax_auc[targets.index(target) // 3, targets.index(target) % 3].plot(fpr_xgb, tpr_xgb, label='XGBoost: AUC-ROC = {:.3f}'.format(roc_auc_score(y_test, y_pred_xgb)))
    ax_auc[targets.index(target) // 3, targets.index(target) % 3].plot([0, 1], [0, 1], linestyle='--', label='Chance level: AUC-ROC = 0.5', color='black', alpha=0.5)
    ax_auc[targets.index(target) // 3, targets.index(target) % 3].set_title(target)
    ax_auc[targets.index(target) // 3, targets.index(target) % 3].set_xlabel('False Positive Rate')
    ax_auc[targets.index(target) // 3, targets.index(target) % 3].set_ylabel('True Positive Rate')
    ax_auc[targets.index(target) // 3, targets.index(target) % 3].legend()

fig_pr.tight_layout()
fig_pr.savefig('baseline_pr.pdf')

fig_auc.tight_layout()
fig_auc.savefig('baseline_auc.pdf')
