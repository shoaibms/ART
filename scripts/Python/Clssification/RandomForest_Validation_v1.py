
import pandas as pd
import seaborn as sns
from joblib import load
import matplotlib.pyplot as plt
import numpy as np
import scikitplot as skplt
from sklearn.metrics import (precision_recall_curve, confusion_matrix, auc,
                             ConfusionMatrixDisplay, accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score, roc_curve)

# Load the model and LabelEncoder
model_path = 'path_to_your_model_file/model_Combine_A_B.joblib'
label_encoder_path = 'path_to_your_label_encoder_file/Combine_label_encoder.joblib'
new_data_path = 'path_to_your_data_file/Combine_validate.csv'
output_file_path = 'path_to_your_output_file/Combine_A_B_output_predict_RF.csv'

loaded_model = load(model_path)
label_encoder = load(label_encoder_path)

# Check the model's class order
classifier = loaded_model
if hasattr(loaded_model, 'named_steps') and 'classification' in loaded_model.named_steps:
    classifier = loaded_model.named_steps['classification']
if hasattr(classifier, 'classes_'):
    print("Model's class order:", classifier.classes_)
else:
    print("No classes_ attribute found in the model.")

# Print out the encoding for each class
for i, class_label in enumerate(label_encoder.classes_):
    print(f"Group '{class_label}' is encoded as {i}")

# Load and prepare new data
new_data = pd.read_csv(new_data_path)
columns_to_exclude = ['Group', 'Group_encoded', 'Image_name', 'data_type', 'Genotype', 'Treatment', 'Replication']
new_X = new_data.drop(columns=columns_to_exclude, errors='ignore')

# Encode and then decode the 'Group' column to verify the process
y_true = new_data['Group']
y_true_encoded = label_encoder.transform(y_true)
y_true_decoded = label_encoder.inverse_transform(y_true_encoded)

# Make predictions
new_predictions = loaded_model.predict(new_X)
new_predictions_decoded = label_encoder.inverse_transform(new_predictions)

new_probabilities = loaded_model.predict_proba(new_X)[:, 1]

# Metrics
accuracy = accuracy_score(y_true_encoded, new_predictions)
precision = precision_score(y_true_encoded, new_predictions, average='weighted')
recall = recall_score(y_true_encoded, new_predictions, average='weighted')
f1 = f1_score(y_true_encoded, new_predictions, average='weighted')
auc_roc = roc_auc_score(y_true_encoded, new_probabilities)

# Calculate specificity
cm = confusion_matrix(y_true_encoded, new_predictions)
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)

# Create a DataFrame with the metrics
metrics_df = pd.DataFrame({
    'Accuracy': [accuracy],
    'Precision': [precision],
    'Recall': [recall],
    'F1 Score': [f1],
    'ROC AUC': [auc_roc],
    'Specificity': [specificity]
})

# Save the DataFrame to a CSV file
metrics_df.to_csv(output_file_path, index=False, header=True)

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_true_encoded, new_probabilities)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (area = {auc_roc:.2f})', color='#cfaf32')
plt.plot([0, 1], [0, 1], color='#4794ad')
plt.xlabel('False Positive Rate', fontsize=18)
plt.ylabel('True Positive Rate', fontsize=18)
plt.title('ROC Curve', fontsize=18)
plt.legend(loc="lower right", fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.show()

# Precision-recall pairs
precision, recall, _ = precision_recall_curve(y_true_encoded, new_probabilities)

# Plot Precision-Recall Curve
plt.figure()
plt.plot(recall, precision, label='Precision-Recall curve', alpha=0.6, color='#cfaf32')
plt.xlabel('Recall', fontsize=18)
plt.ylabel('Precision', fontsize=18)
plt.title('Precision-Recall Curve', fontsize=18)
plt.legend(loc="upper right", fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.show()

# Confusion Matrix (Use decoded labels)
cm = confusion_matrix(y_true_encoded, new_predictions)
labels = label_encoder.classes_

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=labels, yticklabels=labels, annot_kws={"size": 16})
plt.xlabel('Predicted labels', fontsize=18)
plt.ylabel('True labels', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=18)
plt.show()

# Adjusted plot for Class Prediction Error using Confusion Matrix
plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted labels', fontsize=12)
plt.ylabel('True labels', fontsize=12)
plt.title('Class Prediction Confusion Matrix', fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=10)
plt.show()

# Distribution Plots of Predicted Probabilities
plt.figure(figsize=(5, 5))
sns.histplot(new_probabilities, bins=30, kde=True, color='#97f0bd')
plt.title('Distribution of Predicted Probabilities', fontsize=14)
plt.xlabel('Predicted Probability', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.tick_params(axis='both', which='major', labelsize=10)
plt.show()

# Scatter Plot of Predicted Probabilities vs. True Labels (Use decoded labels)
plt.figure(figsize=(5, 5))
plt.scatter(new_probabilities, y_true_encoded, alpha=0.6, color='#e6c63c')
plt.title('Predicted Probabilities vs. True Labels', fontsize=14)
plt.xlabel('Predicted Probability', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.grid(True)
plt.tick_params(axis='both', which='major', labelsize=10)
plt.show()

# Violin Plot for Predicted Probabilities (Use decoded labels)
plt.figure(figsize=(5, 5))
sns.violinplot(x=y_true_decoded, y=new_probabilities, palette=['#72dba0', '#f2f274'])
plt.title('Violin Plot of Predicted Probabilities', fontsize=14)
plt.xlabel('True Label', fontsize=12)
plt.ylabel('Predicted Probability', fontsize=12)
plt.tick_params(axis='x', labelsize=10)
plt.tick_params(axis='y', labelsize=10)
plt.show()

# Feature Importances (Correct handling for when the model is a pipeline)
if hasattr(classifier, 'feature_importances_'):
    importances = classifier.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 5))
    plt.title('Feature Importances', fontsize=14)
    plt.bar(range(len(importances)), importances[indices], color='#97f0bd', align='center')
    plt.xticks(range(len(importances)), np.array(new_X.columns)[indices], rotation='vertical', fontsize=10)
    plt.xlim([-1, len(importances)])
    plt.tight_layout()
    plt.show()

    # Create a DataFrame with the feature names and their importances
    features_df = pd.DataFrame({
        'Feature': np.array(new_X.columns)[indices],
        'Importance': importances[indices]
    })

    # Specify the path where you want to save the feature importances
    features_output_path = 'path_to_your_output_file/Combine_A_B_validate_feature_importances.csv'

    # Save the DataFrame to a CSV file
    features_df.to_csv(features_output_path, index=False, header=True)
else:
    print("The classifier does not have feature_importances_ attribute.")

# Calculate the probabilities for each class using your model
new_probabilities = loaded_model.predict_proba(new_X)

# For binary classification, prepare probabilities for scikitplot
# This formats the probabilities as needed for the plot_cumulative_gain and plot_lift_curve functions
probabilities_two_class = np.hstack([1 - new_probabilities[:, 1].reshape(-1, 1), new_probabilities[:, 1].reshape(-1, 1)])

# Plot the Cumulative Gains Curve
plt.figure()
skplt.metrics.plot_cumulative_gain(y_true_encoded, probabilities_two_class)
plt.title('Cumulative Gains Curve', fontsize=18)
plt.xlabel('X Axis Label', fontsize=18)  # Change 'X Axis Label' as needed
plt.ylabel('Y Axis Label', fontsize=18)  # Change 'Y Axis Label' as needed
plt.tick_params(axis='both', which='major', labelsize=16)
handles, labels = plt.gca().get_legend_handles_labels()
new_labels = ['Class S', 'Class T', 'Baseline']  # Adjust these labels based on your actual classes
plt.legend(handles, new_labels, fontsize=16)
plt.show()

# Plot the Lift Curve
plt.figure()
skplt.metrics.plot_lift_curve(y_true_encoded, probabilities_two_class)
plt.title('Lift Curve', fontsize=18)
plt.xlabel('X Axis Label', fontsize=18)  # Change 'X Axis Label' as needed
plt.ylabel('Y Axis Label', fontsize=18)  # Change 'Y Axis Label' as needed
plt.tick_params(axis='both', which='major', labelsize=16)
handles, labels = plt.gca().get_legend_handles_labels()
new_labels = ['Class S', 'Class T', 'Baseline']  # Adjust these labels based on your actual classes
plt.legend(handles, new_labels, fontsize=16)
plt.show()
