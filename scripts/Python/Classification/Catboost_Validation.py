import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # For loading the label encoder
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, auc

# Paths to model, label encoder, new data, and output file
model_path = r'path_to_your_model/Combine_A_B_model.cbm'
label_encoder_path = r'path_to_your_label_encoder/Combine_A_B_label_encoder.joblib'
new_data_path = r'path_to_your_data/Combine_validate.csv'
output_file_path = r'path_to_your_output/Combine_A_B_output_predict.csv'

# Load the model
loaded_model = CatBoostClassifier()
loaded_model.load_model(model_path)

# Load LabelEncoder
label_encoder = joblib.load(label_encoder_path)

# Load and prepare new data
new_data = pd.read_csv(new_data_path)

# Exclude specified columns from the feature set
columns_to_exclude = ['Group', 'Group_encoded', 'Image_name', 'data_type', 'Genotype', 'Treatment', 'Replication']
new_X = new_data.drop(columns=columns_to_exclude, errors='ignore')

# Encode the 'Group' column using the loaded label encoder
y_true = new_data['Group']
y_true_encoded = label_encoder.transform(y_true)

# Make predictions and calculate probabilities
new_predictions = loaded_model.predict(new_X)
new_probabilities = loaded_model.predict_proba(new_X)[:, 1]  # Adjust if multiclass

# For CatBoost, feature importances can be accessed directly
importances = loaded_model.get_feature_importance()
indices = np.argsort(importances)[::-1]

# Plotting feature importances
plt.figure(figsize=(5, 5))
plt.title('Feature Importances')
plt.bar(range(len(importances)), importances[indices], color='#97f0bd', align='center')
plt.xticks(range(len(importances)), new_X.columns[indices], rotation='vertical')
plt.xlim([-1, len(importances)])
plt.tight_layout()
plt.show()

# Calculate metrics
accuracy = accuracy_score(y_true_encoded, new_predictions)
precision = precision_score(y_true_encoded, new_predictions, average='weighted')
recall = recall_score(y_true_encoded, new_predictions, average='weighted')
f1 = f1_score(y_true_encoded, new_predictions, average='weighted')
auc_roc = roc_auc_score(y_true_encoded, new_probabilities)
tn, fp, _, _ = confusion_matrix(y_true_encoded, new_predictions).ravel()
specificity = tn / (tn + fp)

# Save metrics to CSV
with open(output_file_path, 'w') as file:
    file.write("Metric,Value\n")
    file.write(f"Accuracy,{accuracy:.4f}\n")
    file.write(f"Precision,{precision:.4f}\n")
    file.write(f"Recall,{recall:.4f}\n")
    file.write(f"F1 Score,{f1:.4f}\n")
    file.write(f"AUC-ROC,{auc_roc:.4f}\n")
    file.write(f"Specificity,{specificity:.4f}\n")

# Plot Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_true_encoded, new_probabilities)
pr_auc = auc(recall, precision)
plt.figure()
plt.plot(recall, precision, label=f'Precision-Recall curve (area = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="upper right")
plt.show()

# Correctly defining the confusion matrix display
cm = confusion_matrix(y_true_encoded, new_predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(cmap='Greens')
plt.title('Confusion Matrix')
plt.show()

# Scatter Plots of Predicted Probabilities vs. True Labels
plt.figure(figsize=(5, 5))
plt.scatter(new_probabilities, y_true_encoded, alpha=0.6)
plt.title('Predicted Probabilities vs. True Labels')
plt.xlabel('Predicted Probability')
plt.ylabel('True Label')
plt.grid(True)
plt.show()

# Distribution Plots of Predicted Probabilities
plt.figure(figsize=(5, 5))
sns.histplot(new_probabilities, bins=30, kde=True, color='#97f0bd')
plt.title('Distribution of Predicted Probabilities')
plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')
plt.show()

print(dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))
