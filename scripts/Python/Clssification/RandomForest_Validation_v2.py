# This script evaluates a pre-trained machine learning model on a new validation dataset.
# It loads the model and a corresponding label encoder, predicts on the validation data,
# calculates various performance metrics (e.g., accuracy, precision, recall, F1-score, ROC AUC),
# and generates several plots to visualize the model's performance, including:
# - ROC Curve
# - Precision-Recall Curve
# - Confusion Matrix
# - Distribution of Predicted Probabilities
# - Violin Plot of Predicted Probabilities
# - Feature Importances (if available from the model)
# - Cumulative Gains Curve
# - Lift Curve
# All metrics and plots are saved to a specified output directory.

import pandas as pd
import seaborn as sns
from joblib import load
import matplotlib.pyplot as plt
import numpy as np
import scikitplot as skplt # For cumulative gain and lift curves
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, auc, precision_recall_curve,
                             confusion_matrix, ConfusionMatrixDisplay)
import os

# --- Reproducibility ---
np.random.seed(42)
# Note: If other libraries involving randomness are used (e.g., TensorFlow, PyTorch),
# their respective seeding functions should also be called here.

# --- Configuration ---
# Example: r'./output/rf/train/your_training_results_directory'
TRAINING_OUTPUT_DIR = r'./output/rf/train/training_results' # Placeholder: Specify path to your training output
# Example: r'./data/validation_data.csv'
VALIDATION_DATA_PATH = r'./data/validation_data.csv' # Placeholder: Specify path to your validation dataset

# Example: r'./output/rf/val'
BASE_OUTPUT_DIR_VALIDATION = r'./output/rf/val' # Base for this script's output

# Create a specific subdirectory for this validation run's results
VALIDATION_DATASET_NAME = os.path.splitext(os.path.basename(VALIDATION_DATA_PATH))[0]
OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR_VALIDATION, f"{VALIDATION_DATASET_NAME}_Results_RF")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model and encoder paths (relative to TRAINING_OUTPUT_DIR)
# These should match the filenames used in your training script
MODEL_FILENAME = 'model_output.joblib' # Placeholder: e.g., 'model_RandomForest_SetA.joblib'
LABEL_ENCODER_FILENAME = 'label_encoder.joblib' # Placeholder: e.g., 'SetA_label_encoder.joblib'

MODEL_PATH = os.path.join(TRAINING_OUTPUT_DIR, MODEL_FILENAME)
LABEL_ENCODER_PATH = os.path.join(TRAINING_OUTPUT_DIR, LABEL_ENCODER_FILENAME)

# Plotting Aesthetics (same as training script for consistency)
FIG_SIZE_SMALL = (6, 5)
FIG_SIZE_MEDIUM = (9, 7)
FIG_SIZE_LARGE = (11, 8) # For feature importance if many features

TITLE_FONTSIZE = 22
LABEL_FONTSIZE = 20
TICK_FONTSIZE = 18
LEGEND_FONTSIZE = 18
CONF_MATRIX_ANNOT_FONTSIZE = 28

CMAP_GREEN = 'Greens'
HIST_COLOR_GREEN = '#69c273'
LINE_COLOR_PRIMARY_GREEN = 'darkgreen'
LINE_COLOR_SECONDARY_GREEN = 'mediumseagreen' # This will be used for 'Class T'
SCATTER_COLOR_GREEN = '#8fbc8f' # DarkSeaGreen
VIOLIN_PALETTE_GREEN = ['#a1d99b', '#41ab5d'] # Light and dark green

# --- Helper Function to Save Plots ---
def save_plot(filename, directory=OUTPUT_DIR, dpi=300):
    plt.savefig(os.path.join(directory, filename), dpi=dpi, bbox_inches='tight')
    # plt.show() # Commented out for batch processing or saving figures without displaying
    plt.close()

# --- Load Model and Data ---
print(f"Loading model from: {MODEL_PATH}")
try:
    loaded_model = load(MODEL_PATH)
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}. Ensure the training script ran successfully.")
    exit()

print(f"Loading label encoder from: {LABEL_ENCODER_PATH}")
try:
    label_encoder = load(LABEL_ENCODER_PATH)
except FileNotFoundError:
    print(f"Error: Label encoder file not found at {LABEL_ENCODER_PATH}. Check path and training script output.")
    exit()

print(f"Loading new validation data from: {VALIDATION_DATA_PATH}")
try:
    new_data = pd.read_csv(VALIDATION_DATA_PATH)
except FileNotFoundError:
    print(f"Error: Validation data file not found at {VALIDATION_DATA_PATH}.")
    exit()

print("Preparing new data...")
# Define columns to exclude (should match those excluded during training)
columns_to_exclude_from_X = ['Group', 'Group_encoded', 'Image_name', 'data_type', 'Genotype', 'Treatment', 'Replication']
new_X = new_data.drop(columns=columns_to_exclude_from_X, errors='ignore')

if 'Group' not in new_data.columns:
    print("Error: 'Group' column missing in validation data.")
    exit()

y_true_text = new_data['Group']
y_true_encoded = label_encoder.transform(y_true_text) # Use loaded encoder

print(f"Validation X shape: {new_X.shape}, Validation y shape: {y_true_encoded.shape}")

# --- Make Predictions ---
print("Making predictions on new data...")
new_predictions_encoded = loaded_model.predict(new_X)
new_predictions_decoded = label_encoder.inverse_transform(new_predictions_encoded)

if hasattr(loaded_model, "predict_proba"):
    # Assuming binary classification, get probability of the positive class (class 1)
    # The index [:, 1] assumes the positive class is the second class by the encoder.
    # Verify this with label_encoder.classes_
    if len(label_encoder.classes_) > 1 : # Ensure there are at least two classes
        # Find the index of the positive class.
        # This typically corresponds to the class label that comes second alphabetically/numerically
        # or is explicitly designated as the positive class during encoding.
        positive_class_label = label_encoder.classes_[1] # Assuming [0] is negative, [1] is positive
        positive_class_index = np.where(label_encoder.classes_ == positive_class_label)[0][0]
        new_probabilities_positive_class = loaded_model.predict_proba(new_X)[:, positive_class_index]
    else:
        new_probabilities_positive_class = None
        print("Warning: Only one class detected by label encoder. Some plots may not be meaningful.")

    # For scikitplot, which often expects probabilities for all classes:
    new_probabilities_all_classes = loaded_model.predict_proba(new_X)
else:
    new_probabilities_positive_class = None
    new_probabilities_all_classes = None
    print("Warning: predict_proba not available. ROC AUC, P-R curve, etc. might not be accurate or possible.")

# --- Compute and Save Metrics ---
print("Computing performance metrics...")
accuracy = accuracy_score(y_true_encoded, new_predictions_encoded)
avg_type = 'macro' if len(label_encoder.classes_) > 2 else 'binary'
precision = precision_score(y_true_encoded, new_predictions_encoded, average=avg_type, zero_division=0)
recall = recall_score(y_true_encoded, new_predictions_encoded, average=avg_type, zero_division=0)
f1 = f1_score(y_true_encoded, new_predictions_encoded, average=avg_type, zero_division=0)

auc_roc = np.nan
if new_probabilities_positive_class is not None and len(label_encoder.classes_) == 2:
    auc_roc = roc_auc_score(y_true_encoded, new_probabilities_positive_class)

conf_matrix_val = confusion_matrix(y_true_encoded, new_predictions_encoded)
specificity = np.nan
if len(conf_matrix_val.ravel()) == 4: # Binary classification
    tn, fp, fn, tp = conf_matrix_val.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

metrics_data = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC', 'Specificity (Binary Only)'],
    'Value': [accuracy, precision, recall, f1, auc_roc, specificity]
}
metrics_df = pd.DataFrame(metrics_data)
metrics_output_path = os.path.join(OUTPUT_DIR, f'{VALIDATION_DATASET_NAME}_performance_metrics_RF.csv')
metrics_df.to_csv(metrics_output_path, index=False)
print(f"Validation performance metrics saved to {metrics_output_path}")
print(metrics_df)

# --- Plotting ---
print("Generating validation plots...")
encoded_classes = label_encoder.classes_

# ROC Curve (Binary)
if new_probabilities_positive_class is not None and len(encoded_classes) == 2:
    fpr, tpr, _ = roc_curve(y_true_encoded, new_probabilities_positive_class)
    roc_auc_val = auc(fpr, tpr)
    plt.figure(figsize=FIG_SIZE_SMALL)
    plt.plot(fpr, tpr, color=LINE_COLOR_PRIMARY_GREEN, lw=2.5, label=f'ROC curve (AUC = {roc_auc_val:.2f})')
    plt.plot([0, 1], [0, 1], color='darkgray', lw=2, linestyle='--')
    #plt.title('ROC Curve (Validation Set)', fontsize=TITLE_FONTSIZE)
    plt.xlabel('False Positive Rate', fontsize=LABEL_FONTSIZE)
    plt.ylabel('True Positive Rate', fontsize=LABEL_FONTSIZE)
    plt.xticks(fontsize=TICK_FONTSIZE)
    plt.yticks(fontsize=TICK_FONTSIZE)
    plt.legend(loc="lower right", fontsize=LEGEND_FONTSIZE)
    plt.grid(True, linestyle='--', alpha=0.6)
    save_plot('validation_roc_curve.png')

# Precision-Recall Curve (Binary)
if new_probabilities_positive_class is not None and len(encoded_classes) == 2:
    prec_val, rec_val, _ = precision_recall_curve(y_true_encoded, new_probabilities_positive_class)
    pr_auc_val = auc(rec_val, prec_val)
    plt.figure(figsize=FIG_SIZE_SMALL)
    plt.plot(rec_val, prec_val, color=LINE_COLOR_SECONDARY_GREEN, lw=2.5, label=f'P-R curve (AUC = {pr_auc_val:.2f})')
    #plt.title('Precision-Recall Curve (Validation Set)', fontsize=TITLE_FONTSIZE)
    plt.xlabel('Recall', fontsize=LABEL_FONTSIZE)
    plt.ylabel('Precision', fontsize=LABEL_FONTSIZE)
    plt.xticks(fontsize=TICK_FONTSIZE)
    plt.yticks(fontsize=TICK_FONTSIZE)
    plt.legend(loc="best", fontsize=LEGEND_FONTSIZE) # Changed loc
    plt.grid(True, linestyle='--', alpha=0.6)
    save_plot('validation_precision_recall_curve.png')

# Confusion Matrix
plt.figure(figsize=FIG_SIZE_SMALL)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_val, display_labels=encoded_classes)
disp.plot(cmap=CMAP_GREEN, ax=plt.gca(), values_format='d')
for text_elt in disp.text_.ravel():
    text_elt.set_fontsize(CONF_MATRIX_ANNOT_FONTSIZE)
#plt.title('Confusion Matrix (Validation Set)', fontsize=TITLE_FONTSIZE)
plt.xlabel('Predicted Label', fontsize=LABEL_FONTSIZE)
plt.ylabel('True Label', fontsize=LABEL_FONTSIZE)
plt.xticks(fontsize=TICK_FONTSIZE, rotation=45, ha="right")
plt.yticks(fontsize=TICK_FONTSIZE)
save_plot('validation_confusion_matrix.png')

# Distribution of Predicted Probabilities (Binary)
if new_probabilities_positive_class is not None and len(encoded_classes) == 2:
    plt.figure(figsize=FIG_SIZE_SMALL)
    sns.histplot(new_probabilities_positive_class, bins=30, kde=True, color=HIST_COLOR_GREEN, edgecolor='darkgreen')
    #plt.title('Distribution of Predicted Probabilities (Validation)', fontsize=TITLE_FONTSIZE)
    plt.xlabel(f'Predicted Probability (Class: {encoded_classes[1]})', fontsize=LABEL_FONTSIZE)
    plt.ylabel('Frequency', fontsize=LABEL_FONTSIZE)
    plt.xticks(fontsize=TICK_FONTSIZE)
    plt.yticks(fontsize=TICK_FONTSIZE)
    plt.grid(True, linestyle='--', alpha=0.6)
    save_plot('validation_predicted_probabilities_distribution.png')

# Violin Plot for Predicted Probabilities (Binary)
if new_probabilities_positive_class is not None and len(encoded_classes) == 2:
    plot_df_violin = pd.DataFrame({'True Label': label_encoder.inverse_transform(y_true_encoded),
                                   'Predicted Probability': new_probabilities_positive_class})
    plt.figure(figsize=FIG_SIZE_SMALL)
    # Add hue and set legend=False as suggested by the FutureWarning
    sns.violinplot(x='True Label', y='Predicted Probability', data=plot_df_violin, hue='True Label', palette=VIOLIN_PALETTE_GREEN, legend=False)
    #plt.title('Predicted Probabilities by True Class (Validation)', fontsize=TITLE_FONTSIZE)
    plt.xlabel('True Label', fontsize=LABEL_FONTSIZE)
    plt.ylabel(f'Predicted Probability (Class: {encoded_classes[1]})', fontsize=LABEL_FONTSIZE)
    plt.xticks(fontsize=TICK_FONTSIZE)
    plt.yticks(fontsize=TICK_FONTSIZE)
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    save_plot('validation_probabilities_violin_plot.png')


# Feature Importances (from the loaded model, which includes the feature selector)
print("Extracting feature importances from the loaded model...")
try:
    scaler_step = loaded_model.named_steps['scaler']
    selector_step = loaded_model.named_steps['feature_selection']
    classifier_step = loaded_model.named_steps['classification']

    selected_features_mask = selector_step.get_support()

    if len(selected_features_mask) == len(new_X.columns):
        selected_feature_names = new_X.columns[selected_features_mask]

        if hasattr(classifier_step, 'feature_importances_'):
            gini_importances_selected = classifier_step.feature_importances_

            if len(gini_importances_selected) == len(selected_feature_names):
                importances_plot_df = pd.DataFrame({
                    'Feature': selected_feature_names,
                    'Gini_Importance': gini_importances_selected
                }).sort_values(by='Gini_Importance', ascending=False)

                top_n_for_plot = min(30, len(importances_plot_df))
                importances_plot_df_top = importances_plot_df.head(top_n_for_plot)

                fig_imp_h = max(6, len(importances_plot_df_top) * 0.3)
                plt.figure(figsize=(FIG_SIZE_LARGE[0], fig_imp_h))
                sns.barplot(x='Gini_Importance', y='Feature', data=importances_plot_df_top,
                            color=LINE_COLOR_SECONDARY_GREEN, orient='h')
                #plt.title(f'Top {top_n_for_plot} Feature Importances (from validated model)', fontsize=TITLE_FONTSIZE)
                plt.xlabel('Gini Importance', fontsize=LABEL_FONTSIZE)
                plt.ylabel('Feature', fontsize=LABEL_FONTSIZE)
                plt.xticks(fontsize=TICK_FONTSIZE)
                plt.yticks(fontsize=TICK_FONTSIZE)
                save_plot('validation_model_feature_importances.png')

                importances_plot_df.to_csv(os.path.join(OUTPUT_DIR, f'{VALIDATION_DATASET_NAME}_selected_feature_importances.csv'), index=False)
            else:
                print("Warning: Mismatch in length of Gini importances and selected feature names.")
        else:
            print("Loaded classifier model does not have 'feature_importances_' attribute.")
    else:
        print("Warning: Cannot map feature importances as selected feature mask length" \
              " does not match number of columns in validation data (new_X). " \
              "This can happen if new_X has different columns than training X " \
              "before the pipeline steps.")

except KeyError as e:
    print(f"Error accessing pipeline steps (scaler, feature_selection, classification): {e}")
    print("Ensure the loaded model is the full pipeline and steps are named correctly.")
except AttributeError as e:
    print(f"Error with model attributes (e.g. named_steps might be missing): {e}")
    print("Ensure the loaded model is a scikit-learn Pipeline object.")


# Cumulative Gains and Lift Curve (using scikit-plot)
if new_probabilities_all_classes is not None:
    # Ensure there are at least two classes + baseline for plotting
    if new_probabilities_all_classes.shape[1] > 0:

        # IMPORTANT: Change 'T' to the exact name of the class in your label_encoder.classes_
        # For example, if your classes are ['Control', 'Treatment_X'], and you want to color 'Treatment_X',
        # set target_class_label_for_coloring = 'T'
        # Update this to match one of your actual class names from `label_encoder.classes_`
        target_class_label_for_coloring = 'T' # Placeholder: e.g., 'Positive_Class' or label_encoder.classes_[1]
        color_for_target_class = LINE_COLOR_SECONDARY_GREEN # This is 'mediumseagreen'

        # --- Cumulative Gains Plot ---
        plt.figure(figsize=FIG_SIZE_SMALL)
        ax_gain = plt.gca() # Get current axes
        # MODIFIED: Added title=None to prevent scikit-plot's default title
        skplt.metrics.plot_cumulative_gain(y_true_encoded, new_probabilities_all_classes, ax=ax_gain, title=None)

        # --- Apply custom colors AFTER skplt has drawn ---
        # Find the index of the target class for coloring
        target_idx = -1
        if target_class_label_for_coloring in encoded_classes:
            target_idx = np.where(encoded_classes == target_class_label_for_coloring)[0][0]

        if target_idx != -1:
            # Lines are generated for each class in encoded_classes order, plus a baseline
            # The baseline is typically the last line plotted by scikit-plot for these curves.
            lines = ax_gain.get_lines()
            if len(lines) > target_idx: # Ensure there's a line for the target class
                 lines[target_idx].set_color(color_for_target_class)
                 # Optionally, change the color of the baseline line
                 # if len(lines) > len(encoded_classes): # Check if baseline line exists
                 #    lines[len(encoded_classes)].set_color('gray') # Example: make baseline gray

        # --- Adjust Legend and Title ---
        #plt.title('Cumulative Gains Curve (Validation Set)', fontsize=TITLE_FONTSIZE) # Titles are optional
        plt.xlabel('Percentage of samples', fontsize=LABEL_FONTSIZE)
        plt.ylabel('Gain', fontsize=LABEL_FONTSIZE)
        plt.xticks(fontsize=TICK_FONTSIZE)
        plt.yticks(fontsize=TICK_FONTSIZE)

        # Update legend labels and font size
        handles, skplt_labels = ax_gain.get_legend_handles_labels()
        # skplt_labels will look like ['Class 0', 'Class 1', ..., 'Baseline']
        # We can update the labels to use the actual class names instead of 'Class X'
        updated_labels = [f"Class {encoded_classes[i]}" for i in range(len(encoded_classes))] + ["Baseline"]

        if len(handles) == len(updated_labels):
             ax_gain.legend(handles, updated_labels, fontsize=LEGEND_FONTSIZE, loc='lower right')
        else:
            # Fallback if handle/label count mismatch (e.g., unexpected scikit-plot output)
            print(f"Legend handle/label mismatch for Cumulative Gain. Handles: {len(handles)}, Expected Labels: {len(updated_labels)}. Using scikit-plot's legend text but custom font size.")
            current_legend = ax_gain.get_legend()
            if current_legend is None and handles: # If skplt didn't make one but we have handles
                 ax_gain.legend(handles=handles, labels=skplt_labels, fontsize=LEGEND_FONTSIZE, loc='lower right')
            elif current_legend:
                for text in current_legend.get_texts():
                    text.set_fontsize(LEGEND_FONTSIZE)

        save_plot('validation_cumulative_gain_curve.png')

        # --- Lift Curve Plot ---
        plt.figure(figsize=FIG_SIZE_SMALL)
        ax_lift = plt.gca() # Get current axes
        # MODIFIED: Added title=None to prevent scikit-plot's default title
        skplt.metrics.plot_lift_curve(y_true_encoded, new_probabilities_all_classes, ax=ax_lift, title=None)

        # --- Apply custom colors AFTER skplt has drawn ---
        if target_idx != -1:
            # Lines are generated for each class in encoded_classes order, plus a baseline
            lines = ax_lift.get_lines()
            if len(lines) > target_idx: # Ensure there's a line for the target class
                 lines[target_idx].set_color(color_for_target_class)
                 # Optionally, change the color of the baseline line
                 # if len(lines) > len(encoded_classes): # Check if baseline line exists
                 #    lines[len(encoded_classes)].set_color('gray') # Example: make baseline gray


        # --- Adjust Legend and Title ---
        #plt.title('Lift Curve (Validation Set)', fontsize=TITLE_FONTSIZE) # Titles are optional
        plt.xlabel('Percentage of samples', fontsize=LABEL_FONTSIZE)
        plt.ylabel('Lift', fontsize=LABEL_FONTSIZE)
        plt.xticks(fontsize=TICK_FONTSIZE)
        plt.yticks(fontsize=TICK_FONTSIZE)

        # Update legend labels and font size (same logic as cumulative gain)
        handles, skplt_labels = ax_lift.get_legend_handles_labels()
        updated_labels = [f"Class {encoded_classes[i]}" for i in range(len(encoded_classes))] + ["Baseline"]

        if len(handles) == len(updated_labels):
             ax_lift.legend(handles, updated_labels, fontsize=LEGEND_FONTSIZE, loc='upper right')
        else:
            print(f"Legend handle/label mismatch for Lift Curve. Handles: {len(handles)}, Expected Labels: {len(updated_labels)}. Using scikit-plot's legend text but custom font size.")
            current_legend = ax_lift.get_legend()
            if current_legend is None and handles:
                 ax_lift.legend(handles=handles, labels=skplt_labels, fontsize=LEGEND_FONTSIZE, loc='upper right')
            elif current_legend:
                for text in current_legend.get_texts():
                    text.set_fontsize(LEGEND_FONTSIZE)

        save_plot('validation_lift_curve.png')
    else:
         print("Warning: Not enough classes in probabilities to plot Cumulative Gain and Lift curves.")

print(f"\nAll validation outputs saved to: {OUTPUT_DIR}")
