import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc, precision_recall_curve, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump
from catboost import CatBoostClassifier

# Load the dataset
data_file_path = 'path_to_your_data/Combine_A_B.csv'
data = pd.read_csv(data_file_path)

# Encode the 'Group' column
label_encoder = LabelEncoder()
data['Group_encoded'] = label_encoder.fit_transform(data['Group'])
X = data.drop(['Group', 'Group_encoded', 'Image_name', 'data_type', 'Genotype', 'Treatment', 'Replication'], axis=1)
y = data['Group_encoded']

# Save the LabelEncoder to disk
dump(label_encoder, 'path_to_save_model/Combine_A_B_label_encoder.joblib')

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectFromModel(CatBoostClassifier(random_state=42, verbose=False))),
    ('classification', CatBoostClassifier(random_state=42, verbose=False))
])

# Grid search parameters for CatBoost
parameters = {
    'classification__iterations': [400],
    'classification__depth': [8],
    'classification__learning_rate': [0.1],
    'classification__l2_leaf_reg': [1],
    'classification__border_count': [64],
    'classification__leaf_estimation_iterations': [10],
    'classification__leaf_estimation_method': ['Newton'],
    'classification__min_data_in_leaf': [1]
}

"""
# TRT best parameters: 
parameters = {
    'classification__iterations': [100],
    'classification__depth': [6],
    'classification__learning_rate': [0.1],
    'classification__l2_leaf_reg': [1],
    'classification__border_count': [32],
    'classification__leaf_estimation_iterations': [20],
    'classification__leaf_estimation_method': ['Gradient'],
    'classification__min_data_in_leaf': [1]
}

ART best parameters:
parameters = {
    'classification__iterations': [400],
    'classification__depth': [8],
    'classification__learning_rate': [0.1],
    'classification__l2_leaf_reg': [3],
    'classification__border_count': [32],
    'classification__leaf_estimation_iterations': [20],
    'classification__leaf_estimation_method': ['Gradient'],
    'classification__min_data_in_leaf': [1]
}

Combine best parameters: 
parameters = {
    'classification__iterations': [400],
    'classification__depth': [8],
    'classification__learning_rate': [0.05],
    'classification__l2_leaf_reg': [1],
    'classification__border_count': [64],
    'classification__leaf_estimation_iterations': [10],
    'classification__leaf_estimation_method': ['Newton'],
    'classification__min_data_in_leaf': [1]
}
"""

# Perform grid search
grid_search = GridSearchCV(pipeline, parameters, cv=5, scoring='precision', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best parameters and predictions
best_params = grid_search.best_params_
print(f"Best Parameters: {best_params}")

y_pred = grid_search.predict(X_test)
y_proba = grid_search.predict_proba(X_test)[:, 1]

# Save the model
best_model = grid_search.best_estimator_.named_steps['classification']
best_model.save_model('path_to_save_model/Combine_A_B_model.cbm')

# Compute metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
roc_auc = roc_auc_score(y_test, y_proba)
conf_matrix = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = conf_matrix.ravel()  # For binary classification

# Calculate specificity
specificity = tn / (tn + fp)

# Update metrics DataFrame with specificity
metrics = pd.DataFrame({
    'Accuracy': [accuracy],
    'Precision': [precision],
    'Recall': [recall],
    'F1 Score': [f1],
    'ROC AUC': [roc_auc],
    'Specificity': [specificity]
})

# Save metrics to CSV
output_file_path = 'path_to_save_output/Combine_A_B_output_finetune.csv'
metrics.to_csv(output_file_path, index=False)

print(f"Metrics saved to {output_file_path}")
print(dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

# Function to plot learning curve
def plot_learning_curve(estimator, X, y, scoring, title, ylabel):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, scoring=scoring, n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 50))
    
    # Calculating the mean of training and test scores
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    
    plt.figure(figsize=(6,5))
    plt.plot(train_sizes, train_mean, label='Training score')
    plt.plot(train_sizes, test_mean, label='Cross-validation score')
    plt.title(title)
    plt.xlabel('Training Size')
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

# Plotting Learning Curve for Precision
plot_learning_curve(grid_search.best_estimator_, X, y, 'precision', 'Learning Curve (Precision)', 'Precision score')

# Plotting Learning Curve for ROC AUC
plot_learning_curve(grid_search.best_estimator_, X, y, 'roc_auc', 'Learning Curve (ROC AUC)', 'ROC AUC score')

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(5,5))
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

# Precision-Recall Curve
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
plt.figure(figsize=(5,5))
plt.plot(recall, precision, label='Precision-Recall curve')
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.show()

# Plot Confusion Matrix with Custom Labels
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['S', 'T'])
disp.plot(cmap='Greens')
plt.title('Confusion Matrix')
plt.show()

# Scatter Plots of Predicted Probabilities vs. True Labels
plt.figure(figsize=(5, 5))
plt.scatter(y_proba, y_test, alpha=0.6)
plt.title('Predicted Probabilities vs. True Labels')
plt.xlabel('Predicted Probability')
plt.ylabel('True Label')
plt.grid(True)
plt.show()

# Distribution Plots of Predicted Probabilities
plt.figure(figsize=(5, 5))
sns.histplot(y_proba, bins=30, kde=True, color='#97f0bd')
plt.title('Distribution of Predicted Probabilities')
plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')
plt.show()
