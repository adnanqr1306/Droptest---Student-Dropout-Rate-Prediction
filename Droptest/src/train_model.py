import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from data_processing import load_data, preprocess_data, get_project_root

def calculate_specificity_sensitivity(y_true, y_pred):
    """Calculate specificity and sensitivity from confusion matrix."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
    return specificity, sensitivity

def train_and_evaluate_models():
    """Train models, evaluate them, and save as .pkl files."""

    # Load and preprocess data
    df = load_data()
    df = preprocess_data(df)
    
    # Separate features and target
    X = df.drop('Dropped_Out', axis=1)
    y = df['Dropped_Out']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    # Save test set for app evaluation
    test_set_path = Path(get_project_root()) / 'data' / 'processed' / 'test_set.csv'
    test_set_path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([X_test, y_test], axis=1).to_csv(test_set_path, index=False)

    # Models with descriptions
    models = [
        ('Logistic Regression', 
         Pipeline([
             ('scaler', StandardScaler()),
             ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
         ]),
         "Logistic Regression with class weighting for imbalance"),

        ('Random Forest', 
         RandomForestClassifier(class_weight='balanced', n_estimators=100, max_depth=10, random_state=42),
         "Random Forest with balanced class weights"),

        ('Gradient Boosting', 
         GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42),
         "Gradient Boosting Machines (no native class weighting)"),

        ('Support Vector Machine', 
         Pipeline([
             ('scaler', StandardScaler()),
             ('classifier', SVC(class_weight='balanced', probability=True, random_state=42))
         ]),
         "SVM with RBF kernel and class balancing"),

        ('K-Nearest Neighbors', 
         Pipeline([
             ('scaler', StandardScaler()),
             ('classifier', KNeighborsClassifier(n_neighbors=5))
         ]),
         "KNN with standardized features")
    ]

    # Cross-validation setup
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Store results
    results = []

    # Train and evaluate each model
    for name, model, description in models:
        print(f"\n\033[1mTraining {name}:\033[0m")
        print(f"Algorithm: {description}")

        # Cross-validation
        cv_results = cross_validate(
            model, X_train, y_train, cv=cv,
            scoring=['accuracy', 'precision', 'recall', 'f1'],
            n_jobs=-1,
            verbose=1
        )

        # Full training
        print("Final training on full dataset...")
        model.fit(X_train, y_train)
        print(f"{name} training completed!")

        # Test evaluation
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        specificity, sensitivity = calculate_specificity_sensitivity(y_test, y_pred)

        results.append({
            'Model': name,
            'CV Accuracy': cv_results['test_accuracy'].mean(),
            'CV F1 Score': cv_results['test_f1'].mean(),
            'Test Accuracy': accuracy,
            'Test Precision': precision,
            'Test Recall': recall,
            'Test F1': f1,
            'Specificity': specificity,
            'Sensitivity': sensitivity
        })

        # Save individual model
        model_path = Path(get_project_root()) / 'all_models' / f'{name.lower().replace(" ", "_")}.pkl'
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Select best model
    results_df['Test_F1_Rank'] = results_df['Test F1'].rank(ascending=False)
    results_df['Test_Accuracy_Rank'] = results_df['Test Accuracy'].rank(ascending=False)
    results_df['Composite_Rank'] = (results_df['Test_F1_Rank'] + results_df['Test_Accuracy_Rank']) / 2
    best_model_idx = results_df['Composite_Rank'].idxmin()
    best_model_name = results_df.loc[best_model_idx, 'Model']

    # Save best model
    best_model = [model for name, model, desc in models if name == best_model_name][0]
    best_model_dir = Path(get_project_root()) / 'best_model'
    best_model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, best_model_dir / 'best_model.pkl')

    # Save results
    results_dir = Path(get_project_root()) / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_dir / 'model_metrics.csv', index=False)

    print("\n🏁 Model Performance Comparison:")
    print(results_df[['Model', 'Test Accuracy', 'Test F1']].sort_values(by='Test F1', ascending=False))
    print(f"\n🎯 Best Model Selected: {best_model_name}")

if __name__ == "__main__":
    train_and_evaluate_models()