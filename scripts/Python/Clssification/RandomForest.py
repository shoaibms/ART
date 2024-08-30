import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc, precision_recall_curve, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import seaborn as sns  # For distribution plots
from sklearn.model_selection import learning_curve
from joblib import dump  # For model persistence
from sklearn.metrics import make_scorer
from sklearn.inspection import permutation_importance

# Load the dataset
data_file_path = r'path_to_your_csv_file.csv'
data = pd.read_csv(data_file_path)

# Encode the 'Group' column
label_encoder = LabelEncoder()
data['Group_encoded'] = label_encoder.fit_transform(data['Group'])
X = data.drop(['Group', 'Group_encoded', 'Image_name', 'data_type', 'Genotype', 'Treatment', 'Replication'], axis=1)
y = data['Group_encoded']

# Save the LabelEncoder to disk
dump(label_encoder, r'path_to_save_label_encoder.joblib')

# Print label encoder groups
groups = label_encoder.classes_

# Printing out the encoding
for i, group in enumerate(groups):
    print(f"Group '{group}' is encoded as {i}")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(np.unique(y_train, return_counts=True))
print(np.unique(y_test, return_counts=True))

# Define the model pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectFromModel(RandomForestClassifier(random_state=42))),
    ('classification', RandomForestClassifier(random_state=42, oob_score=True))
])

# Grid search parameters
parameters = {
    'classification__n_estimators': [150],
    'classification__max_depth': [None],
    'classification__min_samples_split': [2],
    'classification__min_samples_leaf': [1],
    'classification__max_features': ['sqrt'],
    'classification__max_leaf_nodes': [100]
}

# Perform grid search
scoring = make_scorer(precision_score, average='binary', zero_division=0)
grid_search = GridSearchCV(pipeline, parameters, cv=10, scoring=scoring, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best parameters and predictions
best_params = grid_search.best_params_
print(f"Best Parameters: {best_params}")

y_pred = grid_search.predict(X_test)
y_proba = grid_search.predict_proba(X_test)[:, 1]

# Save the model to disk
model_file_path = r'path_to_save_model.joblib'
dump(grid_search.best_estimator_, model_file_path)

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
output_file_path = r'path_to_save_metrics.csv'
metrics.to_csv(output_file_path, index=False)

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
    plt.xlabel('Training Size', fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=18)
    plt.show()

# Plotting Learning Curve for Precision
plot_learning_curve(grid_search.best_estimator_, X, y, 'precision', 'Learning Curve (Precision)', 'Precision score')

# Plotting Learning Curve for ROC AUC
plot_learning_curve(grid_search.best_estimator_, X, y, 'roc_auc', 'Learning Curve (ROC AUC)', 'ROC AUC score')

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
plt.figure(figsize=(5,5))
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate', fontsize=18)
plt.ylabel('True Positive Rate', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=18)
plt.show()

# Precision-Recall Curve
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
plt.figure(figsize=(5,5))
plt.plot(recall, precision, label='Precision-Recall curve')
plt.title('Precision-Recall Curve')
plt.xlabel('Recall', fontsize=18)
plt.ylabel('Precision', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=18)
plt.show()

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)
labels = label_encoder.classes_

# Plot Confusion Matrix using ConfusionMatrixDisplay and manually add annotations
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(cmap='Greens', ax=ax, values_format='d')

# Adjust the font size of the annotations
for texts in disp.text_.ravel():
    texts.set_fontsize(20)

plt.title('Confusion Matrix', fontsize=18)
plt.xlabel('Predicted labels', fontsize=18)
plt.ylabel('True labels', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()

print(f"Metrics saved to {output_file_path}")

# Scatter Plots of Predicted Probabilities vs. True Labels
plt.figure(figsize=(5, 5))
plt.scatter(y_proba, y_test, alpha=0.6)
plt.title('Predicted Probabilities vs. True Labels')
plt.xlabel('Predicted Probability', fontsize=18)
plt.ylabel('True Label', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid(True)
plt.show()

# Distribution Plots of Predicted Probabilities
plt.figure(figsize=(5, 5))
sns.histplot(y_proba, bins=30, kde=True, color='#97f0bd')
plt.title('Distribution of Predicted Probabilities')
plt.xlabel('Predicted Probability', fontsize=18)
plt.ylabel('Frequency', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()

# Directly address the feature importance extraction after feature selection
selected_features = grid_search.best_estimator_.named_steps['feature_selection'].get_support()
selected_feature_names = X.columns[selected_features]

feature_importances = grid_search.best_estimator_.named_steps['classification'].feature_importances_

# Map selected feature names back to their original indices in the dataset
original_indices = [list(X.columns).index(name) for name in selected_feature_names]

# Initialize counters for the importance scores of each group
importance_group1 = 0
importance_group2 = 0

# Loop through the original indices to determine the group of each feature and aggregate their importances
for idx, importance in zip(original_indices, feature_importances):
    if idx < 27:  # First 27 features are in group 1
        importance_group1 += importance
    else:  # The remaining features are in group 2
        importance_group2 += importance

# Display the aggregated importance scores for each group
print(f"Total importance for Group 1: {importance_group1}")
print(f"Total importance for Group 2: {importance_group2}")

# Determine which group contributed more based on the aggregated importances
if importance_group1 > importance_group2:
    print("Group 1 contributes more to the model's predictive power.")
else:
    print("Group 2 contributes more to the model's predictive power.")

# Ensure DataFrame creation matches selected features
features_df = pd.DataFrame({
    'Feature': selected_feature_names,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Adjust plotting to reflect corrected DataFrame
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=features_df)
plt.title('Feature Importances')
plt.xlabel('Importance', fontsize=18)
plt.ylabel('Feature', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()

# Save the corrected DataFrame to CSV
feature_importances_file_path = r'path_to_save_feature_importances.csv'
features_df.to_csv(feature_importances_file_path, index=False)

# Assuming grid_search.best_estimator_ is your fitted pipeline
model = grid_search.best_estimator_

# Ensure you scale your X_test data first, as your pipeline includes scaling
X_test_scaled = model.named_steps['scaler'].transform(X_test)

# Apply the same transformations to X_test as were applied during training
# This includes feature selection
X_test_transformed = model.named_steps['feature_selection'].transform(X_test_scaled)

# Now, calculate permutation importance using the transformed test set
# Use the classification model directly since the transformation is manually done
perm_importance_result = permutation_importance(model.named_steps['classification'], X_test_transformed, y_test, n_repeats=30, random_state=42)

# The importances_mean attribute holds the permutation importance scores
perm_importances = perm_importance_result.importances_mean

# Initialize importance sums for both sets
# Note: You might need to adjust this part if the feature selection significantly reduces the number of features
importance_sum_set1 = np.sum(perm_importances[:min(27, len(perm_importances))])  # Adjust if fewer than 27 features are selected
importance_sum_set2 = np.sum(perm_importances[min(27, len(perm_importances)):])  # Adjust accordingly

# Print aggregated importance scores for both sets
print(f"Total Permutation Importance for Set 1: {importance_sum_set1}")
print(f"Total Permutation Importance for Set 2: {importance_sum_set2}")

# Determine which group contributed more based on the aggregated importances
if importance_sum_set1 > importance_sum_set2:
    print("Group 1 contributes more to the model's predictive power according to Permutation Importance.")
else:
    print("Group 2 contributes more to the model's predictive power according to Permutation Importance.")
