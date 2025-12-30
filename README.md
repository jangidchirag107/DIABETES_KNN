# DIABETES_KNN


🩺 Diabetes Prediction & Analysis Using Machine Learning
📌 Project Overview

Diabetes is a chronic medical condition that occurs when the body is unable to effectively regulate blood glucose levels due to insufficient insulin production or improper insulin utilization. Over time, unmanaged diabetes can lead to severe complications such as cardiovascular diseases, kidney failure, nerve damage, and vision loss. Early detection and diagnosis play a crucial role in preventing these long-term complications.

With the growth of healthcare data and advancements in machine learning, predictive models can be used to analyze patient data and assist in early disease detection. This project focuses on building a machine learning-based diabetes prediction system using medical attributes, combined with extensive data visualization (2D and 3D) to understand patterns and relationships within the data.

The project demonstrates a complete end-to-end workflow, starting from data exploration and visualization to model training, evaluation, and interpretation.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
📊 Dataset Description

***The dataset used in this project contains medical records of patients with the following features:***

> Pregnancies

> Glucose

> Blood Pressure

> Skin Thickness

> Insulin

> BMI (Body Mass Index)

> Diabetes Pedigree Function

> Age

> Outcome (Target Variable)

***Target Variable:***

> *0 → Non-Diabetic*

> *1 → Diabetic*

****This dataset is widely used for learning and benchmarking diabetes prediction models.****

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

🔍 Exploratory Data Analysis (EDA)

Before applying machine learning algorithms, Exploratory Data Analysis (EDA) was performed to understand the dataset, detect patterns, and identify important features.

🔹 Outcome Distribution
df['Outcome'].value_counts().plot(kind='bar')
plt.title("Diabetes Outcome Distribution")
plt.xlabel("Outcome (0 = No Diabetes, 1 = Diabetes)")
plt.ylabel("Count")
plt.show()

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

📌 Visualization Insight:

Shows the number of diabetic vs non-diabetic patients

Reveals slight class imbalance

Highlights why accuracy alone is not sufficient

🔹 Glucose vs Outcome
plt.hist(df[df['Outcome']==0]['Glucose'], bins=20, alpha=0.6, label='No Diabetes')
plt.hist(df[df['Outcome']==1]['Glucose'], bins=20, alpha=0.6, label='Diabetes')
plt.xlabel("Glucose Level")
plt.ylabel("Frequency")
plt.title("Glucose Distribution by Outcome")
plt.legend()
plt.show()

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

📌 Medical Insight:

Diabetic patients generally have higher glucose levels

Glucose is the most influential feature in diabetes prediction

🔹 BMI vs Outcome
plt.hist(df[df['Outcome']==0]['BMI'], bins=20, alpha=0.6, label='No Diabetes')
plt.hist(df[df['Outcome']==1]['BMI'], bins=20, alpha=0.6, label='Diabetes')
plt.xlabel("BMI")
plt.ylabel("Frequency")
plt.title("BMI Distribution by Outcome")
plt.legend()
plt.show()

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

📌 Insight:

Higher BMI correlates with increased diabetes risk

Supports medical understanding of obesity and insulin resistance

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

📐 3D Visualization (Advanced Analysis)

To understand the combined effect of multiple features, 3D visualization was used.

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(
    df['Glucose'],
    df['BMI'],
    df['Age'],
    c=df['Outcome']
)

ax.set_xlabel("Glucose")
ax.set_ylabel("BMI")
ax.set_zlabel("Age")
ax.set_title("3D Visualization of Diabetes Data")
plt.show()

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

📌 What this shows:

How glucose, BMI, and age interact together

Diabetic points cluster at higher glucose & BMI

Improves understanding beyond 2D plots

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

⚙️ Data Preprocessing

Machine learning models require properly prepared data.

🔹 Feature Scaling

Since distance-based models depend on numerical ranges, StandardScaler was applied.

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

📌 Why scaling is important:

Prevents features like glucose from dominating distance calculations

Essential for KNN performance

🔹 Train-Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

📌 Ensures fair evaluation on unseen data.

🤖 Machine Learning Models Implemented
🔹 K-Nearest Neighbors (KNN)

KNN predicts outcomes based on similarity between patients.

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

📌 Concept:

“Patients with similar medical attributes tend to have similar diabetes outcomes.”

🔹 Logistic Regression (Explainable Model)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)


Feature importance visualization:

coefficients = pd.Series(model.coef_[0], index=X.columns).sort_values()
coefficients.plot(kind='barh')
plt.title("Logistic Regression Feature Importance")
plt.show()

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

📌 Insight:

Positive coefficients increase diabetes risk

Glucose and BMI show strong positive influence

Provides interpretability for medical analysis

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

📈 Model Evaluation
🔹 Confusion Matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()
plt.show()

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

📌 Helps identify:

True Positives

False Negatives (critical in healthcare)

🔹 Precision–Recall Curve
from sklearn.metrics import precision_recall_curve

precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve")
plt.show()

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

📌 Why important:

Better than accuracy for imbalanced medical datasets

Emphasizes recall (detecting diabetic patients)

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

✅ Key Learnings & Outcomes

Glucose and BMI are the most influential features

Visualization plays a crucial role in medical data understanding

Feature scaling is mandatory for KNN

Recall is more important than accuracy in healthcare

Logistic Regression provides explainability

KNN serves as a strong baseline model

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

🎯 Final Conclusion

This project demonstrates a complete machine learning pipeline for diabetes prediction, integrating data exploration, 2D and 3D visualizations, preprocessing, model building, and evaluation. The use of both KNN and Logistic Regression highlights the trade-off between predictive power and interpretability. The project emphasizes responsible model evaluation in healthcare and shows how machine learning can assist in early disease detection while supporting medical decision-making.
