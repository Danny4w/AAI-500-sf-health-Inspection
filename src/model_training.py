import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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

def model_evaluation(predictions, y_test):
    evaluation_results = []

    # Loop through eah models saved predictions
    for name, preds in predictions.items():
        y_pred = preds['y_pred']
        y_prob = preds['y_prob']
        
        # Calculate evaluation metrics
        recall = recall_score(y_test, y_pred, zero_division=0)
        precision = precision_score(y_test, y_pred, zero_division=0)
        pr_auc = average_precision_score(y_test, y_prob)

        evaluation_results.append({
            'model': name,
            'recall': round(recall, 3),
            'precision': round(precision, 3),
            'pr_auc': round(pr_auc, 3)
        })

        print(f"\n--- Classification Report: {name} ---")
        print(classification_report(y_test, y_pred, zero_division=0))

    # Convert evaluation results to a DataFrame 
    metrics_df = pd.DataFrame(evaluation_results)
    print("\n" + "="*45)
    print("         MODEL PERFORMANCE COMPARISON        ")
    print("="*45)
    print(metrics_df.to_string(index=False))
    print("="*45)

    return metrics_df


def save_best_model(trained_models, metrics_df, X_train):
    # Automatically select the bet model based on PR AUC
    best_auc = metrics_df.loc[metrics_df['pr_auc'].idxmax()]
    best_model = best_auc['model']
    print(f"Best model chosen based on PR AUC: {best_model}")

    best_pipeline = trained_models[best_model]

    # Save the best model to 'models' directory
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'best_model.pkl')
    joblib.dump(best_pipeline, model_path)
    print(f"Best model saved to {model_path}")

    # Extract and Plot Logistic Regression Coefficients coefficients (weights)
    classifier = best_pipeline.named_steps['classifier']
    if hasattr(classifier, 'coef_'):
        coefficients = np.abs(classifier.coef_[0])
        feature_names = X_train.columns

        coef_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': coefficients
        }).sort_values(by='Importance', ascending=False)

        top_features = coef_df.head(15).sort_values(by='Importance', ascending=True)\

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.barh(top_features['Feature'], top_features['Importance'], color='teal')
        plt.xlabel('Importance (coefficient magnitude)', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.title(f'Top 15 Feature Importances ({best_model})', fontsize=14, fontweight='bold')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()

        figure_dir = 'figures'
        os.makedirs(figure_dir, exist_ok=True)
        figure_path = os.path.join(figure_dir, 'feature_importance.png')
        plt.savefig(figure_path)
        print(f"Feature importance plot saved to {figure_path}")
        plt.close()



if __name__ == "__main__":
    file_path = os.path.join('data', 'cleaned-data', 'cleaned_data_modeling.csv')
    X_train, X_test, y_train, y_test = read_and_split_data(file_path)
    trained_models, predictions = model_training(X_train, y_train, X_test)
    metrics_df = model_evaluation(predictions, y_test)
    save_best_model(trained_models, metrics_df, X_train)

