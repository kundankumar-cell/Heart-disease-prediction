import json
import joblib

evaluation_data = {
    "project_name": "Heart Disease Prediction",
    "dataset": "heart.csv",
    "target": "HeartDisease",
    "test_split_ratio": 0.2,
    "stratify": True,
    "random_state": 42,
    "feature_count": 15,
    "features": [
        "Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak",
        "Sex_M", "ChestPainType_ATA", "ChestPainType_NAP", "ChestPainType_TA",
        "RestingECG_Normal", "RestingECG_ST", "ExerciseAngina_Y",
        "ST_Slope_Flat", "ST_Slope_Up"
    ],
    "models_evaluated": [
        {"Model": "Logistic Regression", "Accuracy": 0.8859, "F1 Score": 0.9005, "Rank": 3},
        {"Model": "KNN", "Accuracy": 0.8859, "F1 Score": 0.8995, "Rank": 4},
        {"Model": "Naive Bayes", "Accuracy": 0.9130, "F1 Score": 0.9208, "Rank": 1},
        {"Model": "Decision Tree", "Accuracy": 0.7663, "F1 Score": 0.7962, "Rank": 5},
        {"Model": "SVM (RBF Kernel)", "Accuracy": 0.8967, "F1 Score": 0.9100, "Rank": 2}
    ],
    "best_performing_model": {
        "Model": "Gaussian Naive Bayes",
        "Type": "GaussianNB",
        "Accuracy": 0.9130,
        "Accuracy_Percentage": "91.30%",
        "F1_Score": 0.9208,
        "F1_Score_Percentage": "92.08%",
        "Precision": 0.9300,
        "Precision_Percentage": "93.00%",
        "Recall": 0.9118,
        "Recall_Percentage": "91.18%",
        "Confusion_Matrix": {
            "TN": 75,
            "FP": 7,
            "FN": 9,
            "TP": 93
        },
        "Total_Test_Samples": 184,
        "Artifact_File": "Naive Bayes_heart.pkl",
        "Scaler_File": "scaler.pkl",
        "Columns_File": "columns.pkl"
    },
    "baseline_knn_model": {
        "Model": "KNN",
        "Type": "KNeighborsClassifier",
        "Accuracy": 0.8859,
        "F1_Score": 0.8995,
        "Artifact_File": "knn_heart_model.pkl",
        "Scaler_File": "heart_scaler.pkl",
        "Columns_File": "heart_columns.pkl"
    }
}

with open('model_evaluation_results.json', 'w', encoding='utf-8') as f:
    json.dump(evaluation_data, f, indent=4)

joblib.dump(evaluation_data, 'model_evaluation_results.pkl')
print("Updated model_evaluation_results.json and model_evaluation_results.pkl with detailed metrics.")
