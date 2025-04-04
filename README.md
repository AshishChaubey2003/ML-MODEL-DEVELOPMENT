# ML-MODEL-DEVELOPMENT
🧠 Alzheimer's Disease Prediction Using Random Forest
This project is focused on predicting the presence of Alzheimer's Disease using clinical and behavioral data through a machine learning model (Random Forest Classifier). The workflow includes EDA, preprocessing, feature engineering, model training, and evaluation.

📁 Dataset Overview
File: Data.csv

Number of records: 2149

Number of features: 34 (including diagnosis label)

Target column: Diagnosis (0 = No Alzheimer's, 1 = Alzheimer's)

⚙️ Features
Some key features include:

Demographic: Age, Gender, Ethnicity, EducationLevel

Lifestyle: Smoking, AlcoholConsumption, PhysicalActivity, DietQuality, SleepQuality

Medical History: Diabetes, Hypertension, CardiovascularDisease, HeadInjury

Cognitive and Behavioral Assessments: MMSE, ADL, MemoryComplaints, Forgetfulness

Vitals & Lab Results: BMI, SystolicBP, DiastolicBP, Cholesterol (LDL, HDL, Triglycerides)

🧹 Data Preprocessing Steps
Missing Values: Checked – no missing data found.

Univariate Analysis: Histograms and boxplots plotted for numerical features.

Categorical Exploration: Bar charts for gender, smoking, and alcohol consumption.

Outlier Handling: IQR clipping applied to Age, BMI, MMSE.

Feature Scaling: Standardization using StandardScaler.

Encoding Categorical Data: Label Encoding and One-Hot Encoding.

Dropped Columns: PatientID, DoctorInCharge.

🔍 Exploratory Data Analysis
Distribution plots for Age, BMI, and MMSE.

Correlation heatmap for numerical features.

Boxplots and count plots to analyze relation with Diagnosis.

🧠 Model Building
Model Used: Random Forest Classifier

Cross Validation: 5-fold Cross Validation

Best CV Accuracy: 93.8%

Hyperparameter Tuning: GridSearchCV used with tuning on:

n_estimators, max_depth, min_samples_split, min_samples_leaf

📊 Evaluation Metrics
ROC Curve and AUC

Optimal Threshold (Based on F1 Score)

Confusion Matrix

Accuracy, Sensitivity (Recall), Specificity

Metric	Value
Accuracy	~93.8%
Sensitivity	calculated from test set
Specificity	calculated from test set
🛠️ Requirements
bash
Copy
Edit
pip install pandas numpy matplotlib seaborn scikit-learn
📂 Project Structure
Copy
Edit
├── Data.csv
├── Alzheimer_Prediction.ipynb
├── README.md
💡 Future Work
Try other classifiers like XGBoost or SVM.

Perform SHAP/LIME for model interpretability.

Build a web app using Streamlit or Flask.

🙌 Author
Ashish Kumar Chaubey
B.Tech CSE | Data Engineering & ML Enthusiast
📍 Lucknow | 📧 [Your Email] | 💼 [Your GitHub or LinkedIn]
