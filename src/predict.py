import os
import joblib
import pandas as pd

# Load the trained model and cleaned data
model = joblib.load("models/best_model.pkl")
df = pd.read_csv("data/cleaned-data/cleaned_data_modeling.csv")

# Sample 20 records to simulate upcoming inspections
X_new = df.drop(columns=["target"]).sample(
    n=20, random_state=42
).copy()

# Calculate model-estimated risk scores
X_new["risk_score"] = model.predict_proba(X_new)[:, 1] * 100


# Assign inspection priority based on risk score
def assign_priority(score):
    if score >= 60:
        return "High"
    elif score >= 30:
        return "Medium"
    return "Low"


X_new["priority"] = X_new["risk_score"].apply(assign_priority)


# Convert one-hot encoded features into readable categories
def get_category(row, prefix):
    columns = [col for col in row.index if col.startswith(prefix)]

    for col in columns:
        if row[col] == 1:
            return col.replace(prefix, "")

    return "Reference/Other"


# Create a readable results table
results = pd.DataFrame(index=X_new.index)

results["neighborhood"] = X_new.apply(
    lambda row: get_category(row, "analysis_neighborhood_"), axis=1
)

results["permit_type"] = X_new.apply(
    lambda row: get_category(row, "permit_type_"), axis=1
)

results["inspection_type"] = X_new.apply(
    lambda row: get_category(row, "inspection_type_"), axis=1
)

results["risk_score"] = X_new["risk_score"].round(2)
results["priority"] = X_new["priority"]

# Rank inspections from highest to lowest risk
results = results.sort_values("risk_score", ascending=False)

# Add simulated inspection IDs
results.insert(0, "inspection_id", range(1, len(results) + 1))

# Save results
os.makedirs("outputs", exist_ok=True)
results.to_csv("outputs/inspection_priorities.csv", index=False)

# Display results
print("\nSIMULATED UPCOMING INSPECTION PRIORITIES")
print(results.to_string(index=False))
