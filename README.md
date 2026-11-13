# Heart Disease Prediction (Machine Learning)

This project contains the machine learning pipeline, trained models, preprocessing artifacts, and evaluation benchmarks for predicting heart disease risk based on clinical parameters.

## Project Structure

```text
Machine-Learning-Part-3/
├── heart.csv                     # Original clinical dataset (918 samples, 12 features)
├── HeartdiseaseFinal.ipynb       # Jupyter notebook with EDA, training, and evaluations
├── app.py                        # Streamlit inference application
├── Naive Bayes_heart.pkl         # Best-performing trained model (GaussianNB, 91.30% Acc, 92.08% F1)
├── knn_heart_model.pkl           # Baseline trained KNN model (KNeighborsClassifier)
├── scaler.pkl                    # Feature scaler for Naive Bayes pipeline (StandardScaler)
├── columns.pkl                   # One-hot encoded feature column list (15 features)
├── heart_scaler.pkl              # Feature scaler for KNN pipeline (StandardScaler)
├── heart_columns.pkl             # Feature column list for KNN pipeline
├── model_evaluation_results.json # Serialized evaluation metrics (JSON format)
├── model_evaluation_results.pkl  # Serialized evaluation metrics (joblib format)
├── requirements.txt              # Project Python dependencies
├── verify_all.py                 # Automated 8-point verification and integrity suite
└── backups/                      # Immutable archive copies of original files
```

## Model Benchmark & Evaluation Results

All models were evaluated on an identical stratified test split (80% train, 20% test, `random_state=42`):

| Model | Accuracy | F1 Score | Status |
| :--- | :---: | :---: | :--- |
| **Naive Bayes (GaussianNB)** | **91.30%** | **92.08%** | **Best Performing Model** |
| SVM (RBF Kernel) | 89.67% | 91.00% | Preserved Benchmark |
| Logistic Regression | 88.59% | 90.05% | Preserved Benchmark |
| KNN (KNeighborsClassifier) | 88.59% | 89.95% | Preserved Baseline |
| Decision Tree | 76.63% | 79.62% | Preserved Benchmark |

## Expected Feature Ordering (15 Features)

When passing input features into the scaler and model, ensure the following order:

1. `Age`
2. `RestingBP`
3. `Cholesterol`
4. `FastingBS`
5. `MaxHR`
6. `Oldpeak`
7. `Sex_M`
8. `ChestPainType_ATA`
9. `ChestPainType_NAP`
10. `ChestPainType_TA`
11. `RestingECG_Normal`
12. `RestingECG_ST`
13. `ExerciseAngina_Y`
14. `ST_Slope_Flat`
15. `ST_Slope_Up`

## Verification

To run the automated verification and health check suite:

```bash
python verify_all.py
```

