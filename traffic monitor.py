import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# 1. Simulate dataset with dependent vehicle_count
np.random.seed(42)
n = 5000

hour_of_day = np.random.randint(0, 24, n)
day_of_week = np.random.randint(0, 7, n)
is_holiday = np.random.randint(0, 2, n)
weather_condition = np.random.randint(0, 3, n)  # 0=sunny,1=rainy,2=foggy

# vehicle_count depends on hour_of_day and weather_condition + noise
vehicle_count = (
    50 +
    hour_of_day * 12 +               # more traffic during later hours
    weather_condition * 40 +        # worse weather increases traffic count
    np.random.randint(0, 40, n)     # some randomness
)

df = pd.DataFrame({
    'hour_of_day': hour_of_day,
    'day_of_week': day_of_week,
    'is_holiday': is_holiday,
    'weather_condition': weather_condition,
    'vehicle_count': vehicle_count
})

# 2. Create congestion_level target based on vehicle_count thresholds
conditions = [
    (df['vehicle_count'] <= 200),
    (df['vehicle_count'] > 200) & (df['vehicle_count'] <= 400),
    (df['vehicle_count'] > 400)
]
choices = ['Low', 'Medium', 'High']

df['congestion_level'] = np.select(conditions, choices, default='Low')
df['congestion_level'] = df['congestion_level'].astype(
    'category').cat.codes  # Encode Low=0, Medium=1, High=2

# 3. Prepare features and target
X = df.drop('congestion_level', axis=1)
y = df['congestion_level']

# 4. Train/test split with fixed seed
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 5. Train XGBoost classifier with fixed random_state
model = XGBClassifier(use_label_encoder=False,
                      eval_metric='mlogloss', random_state=42)
model.fit(X_train, y_train)

# 6. Predict & evaluate
y_pred = model.predict(X_test)

print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 7. Feature importance
xgb_feature_importance = model.feature_importances_
features = X.columns

print("\nFeature Importances:")
for f, imp in zip(features, xgb_feature_importance):
    print(f"{f}: {imp:.6f}")

# 8. Plot feature importance with log scale for visibility
plt.figure(figsize=(8, 4))
plt.barh(features, xgb_feature_importance, color='skyblue')
plt.xlabel("Feature Importance")
plt.title("XGBoost Feature Importance (log scale)")
plt.xscale('log')
plt.tight_layout()
plt.show()
