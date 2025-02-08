
# Healthcare Analytics Project

## Overview
This project focuses on building a healthcare analytics system using SQL, Python, and machine learning to analyze healthcare data and predict diseases, specifically diabetes, based on patient records. The data includes various health parameters like age, gender, BMI, blood glucose levels, and more. The goal is to extract useful insights and create a predictive model for diabetes using machine learning techniques.

## Requirements
### 1. SQL Database Setup
The project utilizes an SQL database (`healthcare_db`) where a table (`patient`) is created to store patient health information.

- **SQL Version**: MySQL
- **Libraries**: 
  - `mysql-connector-python`
  - `pandas`
  - `matplotlib`
  - `seaborn`
  
To set up the project database, use the following SQL commands:

```sql
CREATE database healthcare_db;
USE healthcare_db;

CREATE TABLE patient(
    id INT auto_increment PRIMARY KEY,
    gender ENUM('Male','Female'),
    age INT,
    hypertension tinyint, -- 1 for yes, 0 for no
    heart_diseases tinyint,
    smoking_history enum('never','No_info','current'),
    bmi float,
    Hba1c_level float,
    blood_glucose_level float,
    diabetes tinyint -- 1 for diabetes, 0 for no diabetes
);

SHOW DATABASES;
USE healthcare_db;
SHOW TABLES;
```

### 2. Data Exploration & Preparation
In this project, the data is first fetched using SQL and loaded into a Pandas DataFrame. Exploratory Data Analysis (EDA) is performed to better understand the data.

```python
import mysql.connector
import pandas as pd
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="dk123",
    database="healthcare_db"
)
query = "SELECT * FROM patient"
df = pd.read_sql(query, conn)
conn.close()
df.head()
```

Check for any missing values in the dataset:

```python
df.isnull().sum()
```

Perform data type checks and data transformation:

```python
df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
df['smoking_history'] = df['smoking_history'].map({'never': 0, 'No Info': 1, 'current': 2})
```

### 3. Data Visualization
Data visualization is done using `matplotlib` and `seaborn` to gain insights into the distribution and correlations of various health parameters:

- **Age Distribution**:
```python
sns.histplot(df['age'], kde=True)
plt.title('Age distribution')
plt.show()
```

- **BMI Distribution**:
```python
sns.boxplot(x='bmi', data=df)
plt.title('BMI distribution')
plt.show()
```

- **Correlation Heatmap**:
```python
import numpy as np
numeric_df = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap', fontsize=16)
plt.show()
```

### 4. Machine Learning Model: Predicting Diabetes
A Random Forest Classifier is used to predict diabetes based on health data. The model is trained on a subset of data and evaluated using accuracy.

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X = df[['age', 'gender', 'bmi', 'blood_glucose_level', 'smoking_history']]
y = df['diabetes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
```

The model achieved an accuracy of **91.83%**.

### 5. Libraries Used
- **pandas**: For data manipulation and analysis
- **matplotlib**: For creating visualizations
- **seaborn**: For statistical data visualization
- **mysql-connector-python**: For connecting to the MySQL database
- **scikit-learn**: For machine learning algorithms

### 6. Steps to Run the Project

1. **Set up MySQL database**:
   - Create the `healthcare_db` database and `patient` table.
   - Insert healthcare data into the `patient` table.

2. **Run the Python script**:
   - Install necessary Python packages:
     ```bash
     pip install pandas mysql-connector-python matplotlib seaborn
     ```
   - Execute the Python script to load data from the MySQL database, perform data analysis, and train a machine learning model.

3. **Analyze the results**:
   - View the exploratory data analysis (EDA) and visualize key health parameters.
   - Check the performance of the diabetes prediction model.

## Conclusion
This project provides insights into healthcare data through exploratory data analysis and visualizations. Additionally, a predictive model is built using machine learning to classify diabetes based on various health metrics. This can be extended further to analyze other health conditions and improve predictive accuracy by incorporating more features or using different machine learning algorithms.
