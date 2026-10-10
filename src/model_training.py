import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score, average_precision_score, classification_report

# Read in the cleaned dataset and perform 80/20 stratified train/test split
def read_and_split_data(filepath: str):
    df = pd.read_csv(filepath)

    # Separate features (X) and target (y)
    X = df.drop(columns=['target'])
    y = df['target']

    # Stratified 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=83)
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")

    return X_train, X_test, y_train, y_test

def model_training(X_train, y_train, X_test):
    # Define the classification models to be trained (mapping model names to their scikit-learn objects)
    classification_models = {
        "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=83),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=83),
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=83)
    }

    trained_models = {}
    predictions = {}
    for name, model in classification_models.items():
        print(f"Training {name}...")
        pipeline = Pipeline([('classifier', model)])
        pipeline.fit(X_train, y_train)

        # generate binary predictions on the test set (pass or fail --> 0 or 1)
        y_pred = pipeline.predict(X_test)
        # probability scores for the positive class(Fail) --> how confident the model is about the prediction
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        trained_models[name] = pipeline
        predictions[name] = {'y_pred': y_pred, 'y_prob': y_prob}

    print("Training of all models complete.")

    return trained_models, predictions




if __name__ == "__main__":
    file_path = os.path.join('data', 'cleaned-data', 'cleaned_data_modeling.csv')
    X_train, X_test, y_train, y_test = read_and_split_data(file_path)
    trained_models, predictions = model_training(X_train, y_train, X_test)



# Calculate Accuracy, Precision, Recall (Sensitivity), and ROC-AUC on test set for each model.

# Extract feature importance weights from the best model and export plot to 'figures/feature_importance.png'.

# Save trained model artifact to disk using joblib.dump(model, 'models/best_health_model.pkl').