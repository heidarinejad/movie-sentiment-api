import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import re
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, "../dataset/IMDB_Dataset.csv")
df = pd.read_csv(dataset_path, encoding='latin-1')
df.columns = ["review", "label"]

# Clean text
def clean_text(text):
    text = re.sub(r"<.*?>", "", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df["clean_review"] = df["review"].apply(clean_text)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_review"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_features=20000
    )),
    ("clf", LogisticRegression(
        max_iter=1000
    ))
])

# add SVM pipeline
svm_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_features=20000
    )),
    ("clf", LinearSVC())
])

# train the model
pipeline.fit(X_train, y_train)

# train the SVM model
svm_pipeline.fit(X_train, y_train)


# evaluate the model
y_pred = pipeline.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# evaluate the SVM model
svm_pred = svm_pipeline.predict(X_test)

print("\n===== Linear SVC Results =====")
print("Accuracy:", accuracy_score(y_test, svm_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, svm_pred))
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, svm_pred))


# Confusion Matrix Plot
reports_dir = os.path.join(script_dir, "../reports")
os.makedirs(reports_dir, exist_ok=True)

# Clear old reports
for f in glob.glob(os.path.join(reports_dir, "*.png")):
    os.remove(f)

plt.figure(figsize=(6,5))
sns.heatmap(confusion_matrix(y_test, y_pred),
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
cm_path = os.path.join(reports_dir, "confusion_matrix.png")
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {cm_path}")

# Confusion Matrix Plot for SVM
plt.figure(figsize=(6,5))
sns.heatmap(confusion_matrix(y_test, svm_pred),
            annot=True,
            fmt="d",
            cmap="Greens",
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"])
plt.title("Confusion Matrix - Linear SVC")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
svm_cm_path = os.path.join(reports_dir, "confusion_matrix_svm.png")
plt.savefig(svm_cm_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {os.path.join(reports_dir, 'confusion_matrix_svm.png')}")


# ROC Curve
y_probs = pipeline.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test.map({"negative":0, "positive":1}), y_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
roc_path = os.path.join(reports_dir, "roc_curve.png")
plt.savefig(roc_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {roc_path}")

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(
    y_test.map({"negative":0, "positive":1}),
    y_probs
)

plt.figure(figsize=(6,5))
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.tight_layout()
pr_path = os.path.join(reports_dir, "precision_recall_curve.png")
plt.savefig(pr_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {pr_path}")

# which words drive sentiment
feature_names = pipeline.named_steps["tfidf"].get_feature_names_out()
coefficients = pipeline.named_steps["clf"].coef_[0]

top_positive_idx = np.argsort(coefficients)[-20:]
top_negative_idx = np.argsort(coefficients)[:20]

plt.figure(figsize=(8,6))
plt.barh(feature_names[top_positive_idx], coefficients[top_positive_idx])
plt.title("Top Positive Words")
plt.tight_layout()
top_pos_path = os.path.join(reports_dir, "top_positive_words.png")
plt.savefig(top_pos_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {top_pos_path}")

plt.figure(figsize=(8,6))
plt.barh(feature_names[top_negative_idx], coefficients[top_negative_idx])
plt.title("Top Negative Words")
plt.tight_layout()
top_neg_path = os.path.join(reports_dir, "top_negative_words.png")
plt.savefig(top_neg_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {top_neg_path}")

# feature importance plot
importances = np.abs(coefficients)
indices = np.argsort(np.abs(coefficients))[-20:]

plt.figure(figsize=(8,6))
colors = ["green" if coefficients[i] > 0 else "red" for i in indices]

plt.barh(feature_names[indices], coefficients[indices], color=colors)
plt.title("Top Influential Words (Green=Positive, Red=Negative)")
plt.tight_layout()
fi_path = os.path.join(reports_dir, "feature_importance.png")
plt.savefig(fi_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {fi_path}")


# SAVE MODELS
model_dir = os.path.join(script_dir, "../saved_model")
app_dir = os.path.join(script_dir, "../app")

os.makedirs(model_dir, exist_ok=True)
os.makedirs(app_dir, exist_ok=True)

# Compare models
log_acc = accuracy_score(y_test, y_pred)
svm_acc = accuracy_score(y_test, svm_pred)

comparison_df = pd.DataFrame({
    "Model": ["Logistic Regression", "Linear SVC"],
    "Accuracy": [log_acc, svm_acc]
})

print("\nModel Comparison:\n")
print(comparison_df)

comparison_df.to_csv(
    os.path.join(reports_dir, "model_comparison.csv"),
    index=False
)

# FINAL DECISION: Logistic Regression for deployment
best_model = pipeline
best_name = "logistic"

# Save full pipeline
best_model_path = os.path.join(model_dir, "sentiment_model.joblib")
joblib.dump(best_model, best_model_path)

print(f"✓ Best model saved: {best_model_path}")

# Verify
if os.path.exists(best_model_path):
    file_size = os.path.getsize(best_model_path)
    print(f"✓ File verified. Size: {file_size / 1024:.1f} KB")
else:
    print("✗ ERROR: File was not saved!")

# Summary of saved artifacts
print("\nSaved artifacts summary:")
for folder in [model_dir, reports_dir, app_dir]:
    abs_folder = os.path.abspath(folder)
    print(f"\nContents of {abs_folder}:")
    if os.path.exists(folder):
        entries = sorted(glob.glob(os.path.join(folder, "*")))
        if not entries:
            print(" (empty)")
        for path in entries:
            try:
                size = os.path.getsize(path)
                print(f" - {os.path.basename(path)} ({size/1024:.1f} KB)")
            except OSError:
                print(f" - {os.path.basename(path)}")
    else:
        print(" (folder missing)")


