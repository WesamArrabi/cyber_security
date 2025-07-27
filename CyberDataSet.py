import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")            # use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import (
    LabelEncoder, OneHotEncoder, StandardScaler
)
from sklearn.compose       import ColumnTransformer
from sklearn.pipeline      import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.ensemble      import RandomForestClassifier
from sklearn.metrics       import classification_report, confusion_matrix

# ─── CONFIG ────────────────────────────────────────────────
DATA_PATH = r"C:\Users\97254\Downloads\archive (1)\cybersecurity_attacks.csv"

# ─── 1) LOAD & CLEAN ───────────────────────────────────────
df = pd.read_csv(DATA_PATH)
# Drop irrelevant/sensitive columns
to_drop = [
    "Timestamp","Payload Data","Source Port","Destination Port",
    "IDS/IPS Alerts","Source IP Address","Destination IP Address",
    "User Information","Device Information","Geo-location Data",
    "Firewall Logs","Proxy Information","Log Source"
]
df.drop(columns=to_drop, inplace=True)

# Fill common NaNs
df["Malware Indicators"].fillna("None Detected", inplace=True)
df["Alerts/Warnings"].fillna("No Alert",       inplace=True)

# ─── 2) EXPLORATORY VISUALS (saved) ────────────────────────
# a) Anomaly Scores distribution
plt.figure(figsize=(8,6))
sns.boxplot(x=df["Anomaly Scores"].dropna())
plt.title("Anomaly Scores (Box Plot)")
plt.savefig("boxplot_anomaly_scores.png", dpi=300); plt.close()

plt.figure(figsize=(8,6))
sns.histplot(df["Anomaly Scores"].dropna(), kde=True, bins=30)
plt.title("Anomaly Scores (Histogram)")
plt.savefig("hist_anomaly_scores.png", dpi=300); plt.close()

# b) Packet Length vs Anomaly
plt.figure(figsize=(8,6))
sns.scatterplot(x=df["Packet Length"], y=df["Anomaly Scores"], alpha=0.4)
plt.xlabel("Packet Length")
plt.ylabel("Anomaly Scores")
plt.title("Packet Length vs Anomaly Scores")
plt.savefig("scatter_packet_anomaly.png", dpi=300); plt.close()

# c) Packet Length by Severity 
plt.figure(figsize=(8,6))
sns.boxplot(x="Severity Level", y="Packet Length", data=df)
plt.title("Packet Length by Severity Level")
plt.savefig("boxplot_severity_packet.png", dpi=300); plt.close()

# d) Severity vs Attack Type
plt.figure(figsize=(8,6))
sns.pointplot(x="Severity Level", y="Attack Type", data=df,
              hue="Attack Type", dodge=True, linestyles="")
plt.title("Severity Level vs Attack Type")
plt.savefig("pointplot_severity_attack.png", dpi=300); plt.close()

# e) Severity vs Action Taken
plt.figure(figsize=(8,6))
sns.pointplot(x="Severity Level", y="Action Taken", data=df,
              hue="Action Taken", dodge=True, linestyles="")
plt.title("Severity Level vs Action Taken")
plt.savefig("pointplot_severity_action.png", dpi=300); plt.close()

# ─── 3) ENCODE TARGET ──────────────────────────────────────
le = LabelEncoder()
df["Severity_enc"] = le.fit_transform(df["Severity Level"])
# Now le.classes_ == array of original labels in correct order

# ─── 4) PREPARE FEATURES & SPLIT ───────────────────────────
cat_feats = [
    "Protocol","Packet Type","Traffic Type","Malware Indicators",
    "Attack Type","Attack Signature","Action Taken","Network Segment","Alerts/Warnings"
]
num_feats = ["Packet Length","Anomaly Scores"]

X = df[cat_feats + num_feats]
y = df["Severity_enc"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ─── 5) BUILD PIPELINE ────────────────────────────────────
preprocessor = ColumnTransformer([
    ("num", StandardScaler(),     num_feats),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_feats),
])

model = Pipeline([
    ("prep", preprocessor),
    ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
])

# ─── 6) TRAIN & CV ─────────────────────────────────────────
model.fit(X_tr, y_tr)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_validate(
    model, X_tr, y_tr, cv=cv,
    scoring=["accuracy","f1_macro","f1_micro"],
    return_train_score=True
)
print("CV accuracy:", scores["test_accuracy"].mean())
print("CV f1_macro:", scores["test_f1_macro"].mean())

# ─── 7) EVALUATE ON TEST ───────────────────────────────────
y_pred = model.predict(X_te)
report = classification_report(
    y_te, y_pred,
    target_names=le.classes_
)
print(report)

# Save confusion matrix
cm = confusion_matrix(y_te, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=le.classes_, yticklabels=le.classes_
)
plt.ylabel("True")
plt.xlabel("Predicted")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png", dpi=300); plt.close()

# ─── 8) FEATURE IMPORTANCES ────────────────────────────────
importances = model.named_steps["clf"].feature_importances_
feat_names = (
    model.named_steps["prep"]
         .named_transformers_["cat"]
         .get_feature_names_out(cat_feats)
    .tolist()
    + num_feats
)

imp_df = pd.DataFrame({
    "feature": feat_names,
    "importance": importances
}).sort_values("importance", ascending=False).head(15)

plt.figure(figsize=(8,6))
sns.barplot(x="importance", y="feature", data=imp_df)
plt.title("Top 15 Feature Importances")
plt.savefig("feature_importances.png", dpi=300); plt.close()

print("✅ Done! All plots and metrics have been saved.")
