# === Import Libraries ===
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
import warnings

warnings.filterwarnings("ignore")

# === Load Data ===
X_train = pd.read_csv("X_train_sel.csv")
X_test = pd.read_csv("X_test_sel.csv")
y_train = pd.read_csv("y_train.csv").values.ravel()
y_test = pd.read_csv("y_test.csv").values.ravel()

# === Random Forest Model ===
rf = RandomForestClassifier(n_estimators=200, max_depth=None, class_weight='balanced',
                            random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# === Predictions ===
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]

# === Evaluation ===
acc = accuracy_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_proba)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"✅ Accuracy: {acc:.4f}")
print(f"📈 ROC AUC: {roc:.4f}")
print("📊 Classification Report:\n", report)
print("🧮 Confusion Matrix:\n", conf_matrix)





///// STACKING 
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

stack = StackingClassifier(
    estimators=[
        ('rf', rf),  # your best RF
        ('xgb', XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss')),
    ],
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5,
    n_jobs=-1
)

stack.fit(X_train_selected, y_train)
y_pred = stack.predict(X_test_selected)
y_proba = stack.predict_proba(X_test_selected)[:, 1]

# Evaluate
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("📈 ROC AUC:", roc_auc_score(y_test, y_proba))
print("📊 Report:\n", classification_report(y_test, y_pred))
print("🧮 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


