
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    recall_score,
    precision_score,
    average_precision_score,
    classification_report
)


# Read the cleaned dataset and perform an 80/20 stratified split
def read_and_split_data(filepath: str):
    df = pd.read_csv(filepath)

    X = df.drop(columns=['target'])
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=83
    )

    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")

    return X_train, X_test, y_train, y_test


# Train and compare the three classification models
def model_training(X_train, y_train, X_test):

    classification_models = {
        "Logistic Regression": LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=83
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            class_weight='balanced',
            random_state=83
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=83
        )
    }

    trained_models = {}
    predictions = {}
    cv_scores = {}

    # Use 5-fold stratified cross-validation on training data
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=83
    )

    for name, model in classification_models.items():
        print(f"\nTraining {name}...")

        # Evaluate each model using five validation folds
        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring='average_precision'
        )

        cv_scores[name] = scores.mean()
        print(f"Mean CV Average Precision: {scores.mean():.3f}")

        # Train the model on the full training set
        model.fit(X_train, y_train)

        # Generate predictions on the test set
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        trained_models[name] = model
        predictions[name] = {
            'y_pred': y_pred,
            'y_prob': y_prob
        }

    print("\nTraining of all models complete.")

    return trained_models, predictions, cv_scores


# Evaluate each model on the test set
def model_evaluation(predictions, y_test):
    evaluation_results = []

    for name, preds in predictions.items():
        y_pred = preds['y_pred']
        y_prob = preds['y_prob']

        recall = recall_score(y_test, y_pred, zero_division=0)
        precision = precision_score(y_test, y_pred, zero_division=0)
        pr_auc = average_precision_score(y_test, y_prob)

        evaluation_results.append({
            'model': name,
            'recall': recall,
            'precision': precision,
            'pr_auc': pr_auc
        })

        print(f"\n--- Classification Report: {name} ---")
        print(classification_report(
            y_test, y_pred, zero_division=0
        ))

    # Display model performance comparison
    metrics_df = pd.DataFrame(evaluation_results)

    print("\n" + "=" * 45)
    print("MODEL PERFORMANCE COMPARISON")
    print("=" * 45)
    print(metrics_df.round(3).to_string(index=False))
    print("=" * 45)

    return metrics_df


# Save the best model and plot feature importance
def save_best_model(trained_models, cv_scores, X_train):

    # Select the best model based on cross-validation results
    best_model = max(cv_scores, key=cv_scores.get)

    print(f"\nBest model based on CV Average Precision: {best_model}")

    model = trained_models[best_model]

    # Save the selected model
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, 'best_model.pkl')
    joblib.dump(model, model_path)

    print(f"Best model saved to {model_path}")

    # Plot absolute coefficients if Logistic Regression is selected
    if hasattr(model, 'coef_'):

        coefficients = np.abs(model.coef_[0])

        coef_df = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': coefficients
        }).sort_values(
            by='Importance',
            ascending=False
        )

        # Select the 15 largest coefficients
        top_features = coef_df.head(15).sort_values(
            by='Importance',
            ascending=True
        )

        plt.figure(figsize=(10, 6))

        plt.barh(
            top_features['Feature'],
            top_features['Importance'],
            color='teal'
        )

        plt.xlabel('Importance (coefficient magnitude)', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.title(
            f'Top 15 Feature Importances ({best_model})',
            fontsize=14,
            fontweight='bold'
        )

        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()

        # Save the visualization
        figure_dir = 'figures'
        os.makedirs(figure_dir, exist_ok=True)

        figure_path = os.path.join(
            figure_dir, 'feature_importance.png'
        )

        plt.savefig(figure_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Feature importance plot saved to {figure_path}")


if __name__ == "__main__":

    file_path = os.path.join(
        'data', 'cleaned-data', 'cleaned_data_modeling.csv'
    )

    X_train, X_test, y_train, y_test = read_and_split_data(
        file_path
    )

    trained_models, predictions, cv_scores = model_training(
        X_train, y_train, X_test
    )

    metrics_df = model_evaluation(predictions, y_test)

    save_best_model(trained_models, cv_scores, X_train)