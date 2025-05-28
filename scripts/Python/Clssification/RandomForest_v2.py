import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve, auc,
                             precision_recall_curve, ConfusionMatrixDisplay, make_scorer)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from joblib import dump
import os

# --- Configuration ---
DATA_FILE_PATH = r'C:\Users\ms\Desktop\data\Combine_A_B.csv'
BASE_OUTPUT_DIR = r'C:\Users\ms\Desktop\data\output\rf\train' # Base output directory

# Create a specific subdirectory for this dataset's results
DATASET_NAME = os.path.splitext(os.path.basename(DATA_FILE_PATH))[0] # e.g., "Combine_A_B"
OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, f"{DATASET_NAME}_Results_RF")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# CRUCIAL: Define prefixes for your ART features.
# Example: If ART features start with "ART_F_", "DBSCAN_", "HDBSCAN_" etc.
KNOWN_ART_PREFIXES = ['DBSCAN', 'Density', 'FCM', 'GMM', 'HDBSCAN', 'K-mean', 'Mean-shift', 'Mean_shift', 'OPTICS', 'SLIC'] # UPDATE THIS LIST!


RANDOM_STATE = 42
TEST_SET_SIZE = 0.2
CV_FOLDS = 10 # For GridSearchCV and Learning Curve
PERMUTATION_N_REPEATS = 10 # Reduced for speed, increase for more stable results (e.g., 30)
TOP_N_FEATURES_DISPLAY = 20 # Number of top features to display in importance plots

# Plotting Aesthetics
FIG_SIZE_SMALL = (7, 6) # Adjusted for potentially larger text
FIG_SIZE_MEDIUM = (9, 7)
FIG_SIZE_LARGE = (11, 8)

TITLE_FONTSIZE = 22
LABEL_FONTSIZE = 20
TICK_FONTSIZE = 18
LEGEND_FONTSIZE = 18
CONF_MATRIX_ANNOT_FONTSIZE = 28

# Green Themed Palettes
CMAP_GREEN = 'Greens'
HIST_COLOR_GREEN = '#69c273' # A pleasant green

# --- Helper Function to Save Plots ---
def save_plot(filename, directory=OUTPUT_DIR, dpi=300):
    plt.savefig(os.path.join(directory, filename), dpi=dpi, bbox_inches='tight')
    # plt.show() # Commented out for batch processing, uncomment for interactive display
    plt.close()

# --- Helper Function to Identify Feature Type ---
def get_feature_type(feature_name):
    """Classifies a feature as ART or TRT based on KNOWN_ART_PREFIXES."""
    for prefix in KNOWN_ART_PREFIXES:
        if feature_name.startswith(prefix):
            return 'ART'
    return 'TRT'

# --- Load and Prepare Data ---
print(f"Loading data from: {DATA_FILE_PATH}")
data = pd.read_csv(DATA_FILE_PATH)

print("Encoding 'Group' column...")
label_encoder = LabelEncoder()
data['Group_encoded'] = label_encoder.fit_transform(data['Group'])

# Define features (X) and target (y)
# Excluding all potential metadata columns
X = data.drop(['Group', 'Group_encoded', 'Image_name', 'data_type', 'Genotype', 'Treatment', 'Replication'], axis=1, errors='ignore')
y = data['Group_encoded']
print(f"Features (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")


# Save the LabelEncoder
label_encoder_path = os.path.join(OUTPUT_DIR, f'{DATASET_NAME}_label_encoder.joblib')
dump(label_encoder, label_encoder_path)
print(f"LabelEncoder saved to {label_encoder_path}")

# Print label encoding
print("Label Encoding:")
for i, group_name in enumerate(label_encoder.classes_):
    print(f"Group '{group_name}' is encoded as {i}")

# --- Split Data ---
print(f"Splitting data into training ({1-TEST_SET_SIZE:.0%}) and testing ({TEST_SET_SIZE:.0%}) sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SET_SIZE, random_state=RANDOM_STATE, stratify=y)
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
print(f"Training set class distribution: {np.unique(y_train, return_counts=True)}")
print(f"Test set class distribution: {np.unique(y_test, return_counts=True)}")

# --- Define Model Pipeline and Grid Search ---
print("Defining model pipeline and GridSearchCV parameters...")
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectFromModel(RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced'))), # Added class_weight
    ('classification', RandomForestClassifier(random_state=RANDOM_STATE, oob_score=True, class_weight='balanced')) # Added class_weight
])

# Reduced parameters for faster example run, expand as needed for thorough tuning
parameters = {
    'feature_selection__threshold': ['median', 0.01], # Example thresholds for feature selection
    'classification__n_estimators': [100, 200],      # Example: 150 in original
    'classification__max_depth': [None, 10, 20],         # Example: None in original
    'classification__min_samples_split': [2, 5],       # Example: 2 in original
    'classification__min_samples_leaf': [1, 3],        # Example: 1 in original
    'classification__max_features': ['sqrt', 'log2'], # Example: 'sqrt' in original
    # 'classification__max_leaf_nodes': [None, 50, 100] # Example: 100 in original
}

# Define scoring: using precision for binary classification. Adjust if multiclass.
# Assuming binary target (0 and 1). If not, 'weighted' or 'micro'/'macro' might be better.
# Check number of unique classes in y_train
if len(np.unique(y_train)) == 2:
    scoring_metric = make_scorer(precision_score, average='binary', zero_division=0)
    print(f"Using binary precision for scoring.")
else:
    scoring_metric = make_scorer(precision_score, average='macro', zero_division=0) # Or 'weighted'
    print(f"Using macro precision for scoring (multiclass).")


print("Performing GridSearchCV...")
grid_search = GridSearchCV(pipeline, parameters, cv=CV_FOLDS, scoring=scoring_metric, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

best_estimator = grid_search.best_estimator_
print(f"Best Parameters found by GridSearchCV: {grid_search.best_params_}")

# --- Predictions and Model Saving ---
print("Making predictions on the test set...")
y_pred = best_estimator.predict(X_test)

# Check if predict_proba is available (not for all classifiers, but RF has it)
if hasattr(best_estimator, "predict_proba"):
    y_proba = best_estimator.predict_proba(X_test)[:, 1] # Probability of positive class
else:
    y_proba = None # Handle cases where predict_proba might not be available
    print("Warning: predict_proba not available for the best estimator. ROC AUC and P-R curve might not be accurate.")


model_file_path = os.path.join(OUTPUT_DIR, f'model_{DATASET_NAME}_RF.joblib')
dump(best_estimator, model_file_path)
print(f"Best model saved to {model_file_path}")

# --- Compute and Save Metrics ---
print("Computing performance metrics...")
accuracy = accuracy_score(y_test, y_pred)
# Use 'macro' for multiclass or imbalanced binary, 'binary' if appropriate and y_proba is for positive class
avg_type = 'macro' if len(label_encoder.classes_) > 2 else 'binary'
precision = precision_score(y_test, y_pred, average=avg_type, zero_division=0)
recall = recall_score(y_test, y_pred, average=avg_type, zero_division=0)
f1 = f1_score(y_test, y_pred, average=avg_type, zero_division=0)

roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None and len(label_encoder.classes_) == 2 else np.nan

conf_matrix = confusion_matrix(y_test, y_pred)
specificity = np.nan
if len(conf_matrix.ravel()) == 4: # Binary classification
    tn, fp, fn, tp = conf_matrix.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
else: # Multiclass
    # Specificity for multiclass is more complex, often calculated per class
    # For a summary, you might report average specificity or skip for simplicity
    print("Specificity is typically reported per-class for multiclass problems.")


metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC', 'Specificity (Binary Only)'],
    'Value': [accuracy, precision, recall, f1, roc_auc, specificity]
})
metrics_output_path = os.path.join(OUTPUT_DIR, f'{DATASET_NAME}_performance_metrics_RF.csv')
metrics_df.to_csv(metrics_output_path, index=False)
print(f"Performance metrics saved to {metrics_output_path}")
print(metrics_df)

# --- Plotting ---
print("Generating plots...")

# Learning Curves
def plot_learning_curve_custom(estimator, title_suffix, X_data, y_data, scoring_method, ylabel_text):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X_data, y_data, cv=CV_FOLDS, scoring=scoring_method, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 20), random_state=RANDOM_STATE) # Adjusted train_sizes
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=FIG_SIZE_SMALL)
    plt.plot(train_sizes, train_mean, color='darkgreen', marker='o', markersize=5, label='Training score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='darkgreen')
    plt.plot(train_sizes, test_mean, color='mediumseagreen', marker='s', markersize=5, label='Cross-validation score')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.15, color='mediumseagreen')
    
    #plt.title(f'Learning Curve ({title_suffix})', fontsize=TITLE_FONTSIZE)
    plt.xlabel('Training Size', fontsize=LABEL_FONTSIZE)
    plt.ylabel(ylabel_text, fontsize=LABEL_FONTSIZE)
    plt.xticks(fontsize=TICK_FONTSIZE)
    plt.yticks(fontsize=TICK_FONTSIZE)
    plt.legend(loc='best', fontsize=LEGEND_FONTSIZE)
    plt.grid(True, linestyle='--', alpha=0.7)
    save_plot(f'learning_curve_{title_suffix.lower().replace(" ", "_")}.png')

plot_learning_curve_custom(best_estimator, 'Precision', X, y, scoring_metric, 'Precision Score')
if y_proba is not None and len(label_encoder.classes_) == 2:
    plot_learning_curve_custom(best_estimator, 'ROC AUC', X, y, 'roc_auc', 'ROC AUC Score')

# ROC Curve (only for binary classification)
if y_proba is not None and len(label_encoder.classes_) == 2:
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc_val = auc(fpr, tpr)
    plt.figure(figsize=FIG_SIZE_SMALL)
    plt.plot(fpr, tpr, color='darkgreen', lw=2, label=f'ROC curve (AUC = {roc_auc_val:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    #plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=TITLE_FONTSIZE)
    plt.xlabel('False Positive Rate', fontsize=LABEL_FONTSIZE)
    plt.ylabel('True Positive Rate', fontsize=LABEL_FONTSIZE)
    plt.xticks(fontsize=TICK_FONTSIZE)
    plt.yticks(fontsize=TICK_FONTSIZE)
    plt.legend(loc="lower right", fontsize=LEGEND_FONTSIZE)
    plt.grid(True, linestyle='--', alpha=0.7)
    save_plot('roc_curve.png')

# Precision-Recall Curve (only for binary classification)
if y_proba is not None and len(label_encoder.classes_) == 2:
    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    pr_auc_val = auc(rec, prec) # Note: PR AUC is calculated with x=recall, y=precision
    plt.figure(figsize=FIG_SIZE_SMALL)
    plt.plot(rec, prec, color='mediumseagreen', lw=2, label=f'P-R curve (AUC = {pr_auc_val:.2f})')
    #plt.title('Precision-Recall Curve', fontsize=TITLE_FONTSIZE)
    plt.xlabel('Recall', fontsize=LABEL_FONTSIZE)
    plt.ylabel('Precision', fontsize=LABEL_FONTSIZE)
    plt.xticks(fontsize=TICK_FONTSIZE)
    plt.yticks(fontsize=TICK_FONTSIZE)
    plt.legend(loc="lower left", fontsize=LEGEND_FONTSIZE)
    plt.grid(True, linestyle='--', alpha=0.7)
    save_plot('precision_recall_curve.png')


# Confusion Matrix
plt.figure(figsize=FIG_SIZE_SMALL) # Slightly larger for clarity
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=label_encoder.classes_)
disp.plot(cmap=CMAP_GREEN, ax=plt.gca(), values_format='d')
for text_elt in disp.text_.ravel(): # Increase annotation font size
    text_elt.set_fontsize(CONF_MATRIX_ANNOT_FONTSIZE)
#plt.title('Confusion Matrix', fontsize=TITLE_FONTSIZE)
plt.xlabel('Predicted Label', fontsize=LABEL_FONTSIZE)
plt.ylabel('True Label', fontsize=LABEL_FONTSIZE)
plt.xticks(fontsize=TICK_FONTSIZE, rotation=45, ha="right")
plt.yticks(fontsize=TICK_FONTSIZE)
save_plot('confusion_matrix.png')

# Distribution of Predicted Probabilities (only for binary classification)
if y_proba is not None and len(label_encoder.classes_) == 2:
    plt.figure(figsize=FIG_SIZE_SMALL)
    sns.histplot(y_proba, bins=30, kde=True, color=HIST_COLOR_GREEN, edgecolor='darkgreen')
    #plt.title('Distribution of Predicted Probabilities', fontsize=TITLE_FONTSIZE)
    plt.xlabel('Predicted Probability (Class 1)', fontsize=LABEL_FONTSIZE)
    plt.ylabel('Frequency', fontsize=LABEL_FONTSIZE)
    plt.xticks(fontsize=TICK_FONTSIZE)
    plt.yticks(fontsize=TICK_FONTSIZE)
    plt.grid(True, linestyle='--', alpha=0.7)
    save_plot('predicted_probabilities_distribution.png')

# --- Feature Importance Analysis (Gini and Permutation) ---
print("Calculating and plotting feature importances...")

# Get selected feature names AFTER pipeline fitting
scaler_step = best_estimator.named_steps['scaler']
selector_step = best_estimator.named_steps['feature_selection']
classifier_step = best_estimator.named_steps['classification']

# Get mask of selected features from the trained selector
selected_features_mask = selector_step.get_support()
selected_feature_names = X.columns[selected_features_mask]
print(f"Number of features selected by SelectFromModel: {len(selected_feature_names)}")

# Gini Importance (from the final classifier, on selected features)
gini_importances_selected = classifier_step.feature_importances_
gini_importances_df = pd.DataFrame({
    'Feature': selected_feature_names,
    'Gini_Importance': gini_importances_selected
}).sort_values(by='Gini_Importance', ascending=False)
gini_importances_df['Type'] = gini_importances_df['Feature'].apply(get_feature_type)


# Permutation Importance (on the test set with selected features)
X_test_scaled = scaler_step.transform(X_test)
X_test_selected = selector_step.transform(X_test_scaled) # selector was fit on X_train

perm_importance_result = permutation_importance(
    classifier_step, X_test_selected, y_test,
    n_repeats=PERMUTATION_N_REPEATS, random_state=RANDOM_STATE, n_jobs=-1
)
perm_importances_df = pd.DataFrame({
    'Feature': selected_feature_names, # Features are already selected
    'Permutation_Importance_Mean': perm_importance_result.importances_mean,
    'Permutation_Importance_Std': perm_importance_result.importances_std
}).sort_values(by='Permutation_Importance_Mean', ascending=False)
perm_importances_df['Type'] = perm_importances_df['Feature'].apply(get_feature_type)


# Plot Top N Gini Importances
top_gini_df = gini_importances_df.head(TOP_N_FEATURES_DISPLAY)
fig_gini_h = max(6, len(top_gini_df) * 0.35)
plt.figure(figsize=(FIG_SIZE_SMALL[0], fig_gini_h))
sns.barplot(x='Gini_Importance', y='Feature', data=top_gini_df, hue='Type', dodge=False, palette={'ART': 'mediumseagreen', 'TRT': 'darkseagreen'})
#plt.title(f'Top {TOP_N_FEATURES_DISPLAY} Features by Gini Importance', fontsize=TITLE_FONTSIZE)
plt.xlabel('Gini Importance', fontsize=LABEL_FONTSIZE)
plt.ylabel('Feature', fontsize=LABEL_FONTSIZE)
plt.xticks(fontsize=TICK_FONTSIZE)
plt.yticks(fontsize=TICK_FONTSIZE)
plt.legend(title='Feature Type', fontsize=LEGEND_FONTSIZE - 2)
save_plot(f'top_{TOP_N_FEATURES_DISPLAY}_gini_importances.png')

# Plot Top N Permutation Importances
top_perm_df = perm_importances_df.head(TOP_N_FEATURES_DISPLAY)
fig_perm_h = max(6, len(top_perm_df) * 0.35)
plt.figure(figsize=(FIG_SIZE_SMALL[0]+1, fig_perm_h)) # +1 for error bars
y_coords = np.arange(len(top_perm_df))

# Define colors as per request
color_art = '#228B22' # ForestGreen
color_trt = '#00CED1' # DarkTurquoise

plt.barh(y_coords, top_perm_df['Permutation_Importance_Mean'],
         xerr=top_perm_df['Permutation_Importance_Std'],
         align='center', 
         color=[color_art if t == 'ART' else color_trt for t in top_perm_df['Type']], # MODIFIED: Color mapping
         ecolor='gray', capsize=4)
plt.yticks(y_coords, top_perm_df['Feature'])

# Create custom legend handles for Type with new colors
art_patch = mpatches.Patch(color=color_art, label='ART') # MODIFIED: Color for ART patch
trt_patch = mpatches.Patch(color=color_trt, label='TRT') # MODIFIED: Color for TRT patch
plt.legend(handles=[art_patch, trt_patch], fontsize=LEGEND_FONTSIZE - 2) # MODIFIED: Removed title='Feature Type'

#plt.title(f'Top {TOP_N_FEATURES_DISPLAY} Features by Permutation Importance', fontsize=TITLE_FONTSIZE)
plt.xlabel('Permutation Importance (Mean ± SD)', fontsize=LABEL_FONTSIZE)
plt.ylabel('Feature', fontsize=LABEL_FONTSIZE)
plt.gca().invert_yaxis() # To match barplot order
plt.xticks(fontsize=TICK_FONTSIZE)
plt.yticks(fontsize=TICK_FONTSIZE)
save_plot(f'top_{TOP_N_FEATURES_DISPLAY}_permutation_importances.png')


# Aggregate Importances (ART vs TRT)
gini_art_importance = gini_importances_df[gini_importances_df['Type'] == 'ART']['Gini_Importance'].sum()
gini_trt_importance = gini_importances_df[gini_importances_df['Type'] == 'TRT']['Gini_Importance'].sum()
perm_art_importance = perm_importances_df[perm_importances_df['Type'] == 'ART']['Permutation_Importance_Mean'].sum()
perm_trt_importance = perm_importances_df[perm_importances_df['Type'] == 'TRT']['Permutation_Importance_Mean'].sum()

print("\n--- Aggregate Feature Importances ---")
print(f"Total Gini Importance for ART features: {gini_art_importance:.4f}")
print(f"Total Gini Importance for TRT features: {gini_trt_importance:.4f}")
print(f"Total Permutation Importance (Mean) for ART features: {perm_art_importance:.4f}")
print(f"Total Permutation Importance (Mean) for TRT features: {perm_trt_importance:.4f}")

if gini_art_importance > gini_trt_importance:
    print("ART features contribute more based on Gini Importance.")
else:
    print("TRT features contribute more based on Gini Importance.")

if perm_art_importance > perm_trt_importance:
    print("ART features contribute more based on Permutation Importance.")
else:
    print("TRT features contribute more based on Permutation Importance.")

# Save all feature importances
gini_importances_df.to_csv(os.path.join(OUTPUT_DIR, f'{DATASET_NAME}_gini_importances.csv'), index=False)
perm_importances_df.to_csv(os.path.join(OUTPUT_DIR, f'{DATASET_NAME}_permutation_importances.csv'), index=False)
print(f"Full Gini importances saved to {DATASET_NAME}_gini_importances.csv")
print(f"Full Permutation importances saved to {DATASET_NAME}_permutation_importances.csv")

print(f"\nAll training outputs saved to: {OUTPUT_DIR}")