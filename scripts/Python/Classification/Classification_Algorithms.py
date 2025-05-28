# This code compares multiple algorithms and generates necessary plots

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    make_scorer, accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, roc_curve, auc, precision_recall_curve
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import warnings

warnings.filterwarnings('ignore')

def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

def evaluate_models(file_path, output_path):
    data = pd.read_csv(file_path)
    excluded_columns = ['Image_name', 'data_type', 'Genotype', 'Treatment', 'Replication']
    X = data.drop(excluded_columns + ['Group'], axis=1)
    y = data['Group']

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'Gradient Boosting': Pipeline([('scaler', StandardScaler()), ('classifier', GradientBoostingClassifier())]),
        'Random Forest': RandomForestClassifier(),
        'AdaBoost': AdaBoostClassifier(),
        'Logistic Regression': Pipeline([('scaler', StandardScaler()), ('classifier', LogisticRegression())]),
        'SVM': Pipeline([('scaler', StandardScaler()), ('classifier', SVC(probability=True))]),
        'K-Nearest Neighbors': Pipeline([('scaler', StandardScaler()), ('classifier', KNeighborsClassifier())]),
        'LightGBM': LGBMClassifier(),
        'CatBoost': CatBoostClassifier(verbose=0)
    }

    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='macro', zero_division=1),
        'recall': make_scorer(recall_score, average='macro'),
        'f1': make_scorer(f1_score, average='macro'),
        'roc_auc': make_scorer(roc_auc_score, needs_proba=True, multi_class='ovr', average='macro'),
        'specificity': make_scorer(specificity_score)
    }

    scores_list = []
    for name, model in models.items():
        cv_results = cross_validate(model, X_train, y_train, cv=5, scoring=scoring)
        scores = {
            'Model': name,
            'Accuracy': np.mean(cv_results['test_accuracy']),
            'Precision': np.mean(cv_results['test_precision']),
            'Recall': np.mean(cv_results['test_recall']),
            'F1 Score': np.mean(cv_results['test_f1']),
            'ROC AUC': np.mean(cv_results['test_roc_auc']),
            'Specificity': np.mean(cv_results['test_specificity'])
        }
        scores_list.append(scores)

    scores_df = pd.DataFrame(scores_list)
    scores_df.to_csv(output_path, index=False)
    return scores_df, models, X_train, y_train, X_test, y_test

data_file_path = 'path_to_your_csv_file/ART.csv' 
output_file_path = 'path_to_save_output/Output_ART.csv'

model_scores, models, X_train, y_train, X_test, y_test = evaluate_models(data_file_path, output_file_path)

# Plotting ROC and Precision-Recall for each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # Check if the model has predict_proba method
    if hasattr(model, "predict_proba"):
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f'{name} (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic: {name}')
        plt.legend(loc="lower right")
        plt.show()

        precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
        plt.figure()
        plt.plot(recall, precision, label=name)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve: {name}')
        plt.legend()
        plt.show()
    else:
        print(f"{name} does not support predict_proba, skipping ROC and Precision-Recall plots.")
