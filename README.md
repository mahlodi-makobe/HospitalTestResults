# Predictive Modeling for Test Results

## Objective
The goal of this project is to build a **multi-class classification model** to predict **Test Results** (Normal, Abnormal, Inconclusive) based on patient demographics, medical conditions, and other features in a synthetic healthcare dataset. This model can help healthcare providers prioritize patient care and improve diagnostic accuracy.

## Dataset
The dataset used in this project is a **synthetic healthcare dataset** containing information about patients, including:
- Demographics (e.g., Age, Gender)
- Medical details (e.g., Medical Condition, Blood Type)
- Administrative data (e.g., Insurance Provider, Billing Amount)
- Test Results (Target Variable: Normal, Abnormal, Inconclusive)

## Exploratory Data Analysis (EDA)

### Key Insights
1. **Target Variable Distribution**:
   - The dataset is balanced, with roughly equal numbers of Normal, Abnormal, and Inconclusive test results.

2. **Feature Relationships**:
   - **Age**: Older patients tend to have more Abnormal test results.
   - **Billing Amount**: Higher billing amounts are associated with Abnormal or Inconclusive results.
   - **Length of Stay**: Patients with longer hospital stays are more likely to have Abnormal results.

### Visualizations
- Bar charts and boxplots were used to explore relationships between features and the target variable.
- A correlation heatmap was created to identify relationships between numerical features.

## Data Preprocessing

### Steps
1. **Handling Missing Values**:
   - No missing values were found in the dataset.

2. **Encoding Categorical Variables**:
   - Categorical variables (e.g., Gender, Blood Type, Medical Condition) were encoded using **one-hot encoding**.

3. **Scaling Numerical Features**:
   - Numerical features (e.g., Age, Billing Amount) were normalized using **StandardScaler**.

4. **Train-Test Split**:
   - The dataset was split into training (80%) and testing (20%) sets.

## Feature Engineering

### New Features
1. **Length of Stay**:
   - Calculated as the difference between `Discharge Date` and `Date of Admission`.

2. **Age × Medical Condition**:
   - Created to capture interactions between age and medical conditions.

### Feature Selection
- Features with low importance scores (e.g., Medication, Insurance Provider) were dropped to reduce noise.
- The most important features were:
  1. **Billing Amount**
  2. **Room Number**
  3. **Age**
  4. **Length of Stay**

## Model Training

### Models Trained
1. **Random Forest**: Best accuracy (0.442).
2. **XGBoost**: Underperformed (accuracy: 0.378).
3. **LightGBM**: Underperformed (accuracy: 0.389).
4. **CatBoost**: Underperformed (accuracy: 0.375).
5. **Logistic Regression**: Underperformed (accuracy: 0.345).
6. **k-Nearest Neighbors (k-NN)**: Underperformed (accuracy: 0.372).
7. **Neural Network**: Underperformed (accuracy: 0.343).

### Ensemble Methods
- **VotingClassifier**: Accuracy = 0.438.
- **Stacking**: Accuracy = 0.435.

## Model Evaluation

### Evaluation Metrics
- **Accuracy**: 0.442 (Random Forest with feature selection).
- **Precision**: 0.43 (weighted average).
- **Recall**: 0.43 (weighted average).
- **F1-Score**: 0.43 (weighted average).

### Confusion Matrix
- The confusion matrix shows that the model performs similarly across all classes (Normal, Abnormal, Inconclusive).

## Next Steps
1. **Collect More Data**:
   - Include additional domain-specific features (e.g., lab test results, patient history) to improve predictive power.
2. **Advanced Models**:
   - Experiment with more sophisticated models (e.g., deep learning) if additional data is available.
3. **Deploy the Model**:
   - Deploy the best-performing model (Random Forest) as an API using Flask or FastAPI.
   - Operationalize the model on a cloud platform (e.g., AWS, Azure).

## Tools and Libraries

### Python Libraries
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Machine Learning**: Scikit-learn, XGBoost, LightGBM, CatBoost, TensorFlow/Keras
- **Model Evaluation**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- **Ensemble Methods**: VotingClassifier, StackingClassifier

## Conclusion
This project demonstrates the use of machine learning to predict test results based on patient data. The **Random Forest model** achieved the best performance, with an accuracy of **0.442**. While this is an improvement over random guessing (baseline accuracy: ~0.33), the model is still a **work in progress**, and the current accuracy level is not yet sufficient for real-world deployment.
