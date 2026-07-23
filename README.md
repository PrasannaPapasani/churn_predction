# 📉 Telecom Customer Churn Prediction (Classification)

## 📌 Project Overview

Customer churn is one of the biggest challenges faced by telecom companies. Acquiring new customers is more expensive than retaining existing ones. This project builds a **Machine Learning Classification Model** to predict whether a customer is likely to leave the telecom service based on customer demographics, account information, and service usage.

The model helps businesses identify high-risk customers and take proactive retention measures such as personalized offers, discounts, or customer support.

---

## 🎯 Problem Statement

The objective of this project is to predict whether a telecom customer will **Churn** (leave the company) or **Stay** using historical customer data.

This is a **Supervised Machine Learning Classification** problem where:

- **Target Variable:** Churn
- **Classes:**
  - Yes → Customer will churn
  - No → Customer will stay

---

# 📂 Dataset

**Dataset:** Telecom Customer Churn Dataset

### Features

| Feature | Description |
|----------|-------------|
| customerID | Unique customer ID |
| gender | Male/Female |
| SeniorCitizen | Whether customer is senior citizen |
| Partner | Has partner or not |
| Dependents | Has dependents or not |
| tenure | Number of months with company |
| PhoneService | Phone service subscription |
| MultipleLines | Multiple phone lines |
| InternetService | DSL/Fiber/No |
| OnlineSecurity | Online security service |
| OnlineBackup | Online backup |
| DeviceProtection | Device protection |
| TechSupport | Technical support |
| StreamingTV | TV streaming |
| StreamingMovies | Movie streaming |
| Contract | Contract type |
| PaperlessBilling | Paperless billing |
| PaymentMethod | Payment method |
| MonthlyCharges | Monthly bill |
| TotalCharges | Total bill |
| Churn | Target Variable |

---

# 🛠️ Technologies Used

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost (Optional)

---

# 📁 Project Structure

```
Telecom-Customer-Churn-Prediction/
│
├── data/
│   ├── telecom_churn.csv
│
├── notebooks/
│   ├── Customer_Churn_Prediction.ipynb
│
├── models/
│   ├── trained_model.pkl
│
├── images/
│   ├── correlation_heatmap.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│
├── requirements.txt
├── README.md
└── app.py (Optional)
```

---

# 🔄 Project Workflow

### 1. Data Collection

- Load Telecom Customer Churn dataset
- Understand business problem

---

### 2. Data Exploration

- Dataset Shape
- Dataset Information
- Statistical Summary
- Missing Values
- Duplicate Records

---

### 3. Data Cleaning

- Handle missing values
- Remove duplicates
- Correct data types
- Handle inconsistent values

---

### 4. Exploratory Data Analysis (EDA)

Performed analysis using:

- Customer Churn Distribution
- Gender Distribution
- Contract Type Analysis
- Internet Service Analysis
- Monthly Charges Distribution
- Tenure Distribution
- Correlation Heatmap
- Countplots
- Histograms
- Boxplots

---

### 5. Data Preprocessing

- Label Encoding
- One-Hot Encoding
- Feature Scaling
- Train-Test Split

---

### 6. Feature Engineering

- Encode categorical variables
- Remove unnecessary columns
- Create numerical features if required

---

### 7. Model Building

Implemented multiple classification algorithms:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Gradient Boosting
- XGBoost
- AdaBoost

---

### 8. Model Evaluation

Evaluation Metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Confusion Matrix
- Classification Report

---

### 9. Hyperparameter Tuning

Performed using:

- GridSearchCV
- RandomizedSearchCV

---

### 10. Model Saving

```python
import joblib

joblib.dump(model, "customer_churn_model.pkl")
```

---

# 📊 Exploratory Data Analysis

Some important visualizations include:

- Customer Churn Count
- Contract Type Distribution
- Monthly Charges Distribution
- Tenure Distribution
- Correlation Heatmap
- Confusion Matrix
- ROC Curve

---

# 🤖 Machine Learning Models

| Model | Type |
|--------|------|
| Logistic Regression | Linear Classification |
| Decision Tree | Tree-Based |
| Random Forest | Ensemble |
| KNN | Distance-Based |
| SVM | Margin-Based |
| Gradient Boosting | Ensemble |
| XGBoost | Boosting |
| AdaBoost | Boosting |

---

# 📈 Evaluation Metrics

```
Accuracy Score

Precision

Recall

F1 Score

ROC-AUC Score

Confusion Matrix

Classification Report
```

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Telecom-Customer-Churn-Prediction.git
```

Move into project folder

```bash
cd Telecom-Customer-Churn-Prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the notebook

```bash
jupyter notebook
```

---

# ▶️ Usage

1. Load the dataset.
2. Perform preprocessing.
3. Train the machine learning model.
4. Evaluate model performance.
5. Save the trained model.
6. Predict customer churn for new customer data.

---

# 📌 Business Impact

This model helps telecom companies:

- Identify customers likely to churn.
- Improve customer retention strategies.
- Reduce revenue loss.
- Increase customer lifetime value.
- Support data-driven business decisions.

---

# 🔮 Future Improvements

- Deploy using Flask/FastAPI
- Streamlit Web Application
- Docker Containerization
- CI/CD Pipeline
- Cloud Deployment (AWS/Azure/GCP)
- Real-time Churn Prediction API
- Model Monitoring and Retraining

---

# 📷 Sample Output

```
Prediction:

Customer Status: Churn

Probability of Churn: 92%
```

---

# 📚 Skills Demonstrated

- Data Cleaning
- Exploratory Data Analysis
- Feature Engineering
- Data Visualization
- Classification Algorithms
- Hyperparameter Tuning
- Model Evaluation
- Machine Learning Pipeline
- Model Serialization
- Business Problem Solving

---

# 📄 License

This project is licensed under the MIT License.

---

# 👤 Author

**Prasanna Papasani**

Aspiring Data Scientist | Machine Learning Engineer | AI Enthusiast

- Python
- SQL
- Machine Learning
- Deep Learning
- NLP
- Data Visualization
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn

⭐ If you found this project helpful, please consider giving it a star!
