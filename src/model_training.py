# Read 'data/processed/cleaned_data.csv' and split into X (predictors) and y (target).

# Perform an 80/20 train/test split with stratify=y and random_state=any.

# Train the following Models
# -  Logistic Regression(class_weight='balanced')
# -  DecisionTreeClassifier(max_depth=5, class_weight='balanced')
# -  RandomForestClassifier(n_estimators=100, class_weight='balanced')

# Calculate Accuracy, Precision, Recall (Sensitivity), and ROC-AUC on test set for each model.

# Extract feature importance weights from the best model and export plot to 'figures/feature_importance.png'.

# Save trained model artifact to disk using joblib.dump(model, 'models/best_health_model.pkl').