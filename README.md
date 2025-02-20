# Diabetes Prediction using Machine Learning

## Overview
Diabetes Mellitus is a critical disease affecting millions worldwide. Various factors such as age, obesity, lack of exercise, hereditary diabetes, lifestyle choices, poor diet, and high blood pressure contribute to its onset. 

Machine learning models provide the ability to analyze vast healthcare datasets, identify hidden patterns, and detect risk factors that traditional methods may overlook. This project aims to predict diabetes in individuals based on health records and lifestyle factors using machine learning.

## Dataset
The dataset consists of **768** patient records with the following health parameters:

| Column                     | Data Type  | Non-Null Count |
|----------------------------|------------|---------------|
| Pregnancies                | int64      | 768           |
| Glucose                    | int64      | 768           |
| BloodPressure              | int64      | 768           |
| SkinThickness              | int64      | 768           |
| Insulin                    | int64      | 768           |
| BMI                        | float64    | 768           |
| DiabetesPedigreeFunction   | float64    | 768           |
| Age                        | int64      | 768           |
| Outcome                    | int64      | 768           |

- The dataset has **9 columns** with numerical values.
- The **Outcome** column indicates whether the patient has diabetes (1) or not (0).
- The dataset uses **54.1 KB** of memory.

## Methodologies Used
1. **Data Loading and Exploration**
   - Load the dataset and inspect the features.
2. **Data Visualization**
   - Visualize data distribution and relationships among features.
3. **Data Preprocessing**
   - Handle missing values, normalize features, and encode categorical variables.
4. **Exploratory Data Analysis (EDA)**
   - Identify correlations and trends in the dataset.
5. **Model Training and Evaluation**
   - Train **Linear Regression** to predict diabetes risk.
   - Split data into **90% training** and **10% testing** sets.
   - Evaluate model performance.

## Model Performance
- The trained **Linear Regression model** achieved an accuracy of **85.7%**.

## Results
These machine learning models help predict diabetes by analyzing patient health records, lifestyle factors, and genetic history. By including additional parameters, such as cholesterol levels and dietary habits, the model can improve its predictive accuracy and assist in early diagnosis.

## Technologies Used
- Python
- Pandas & NumPy (Data Handling)
- Matplotlib & Seaborn (Data Visualization)
- Scikit-learn (Machine Learning)

## Installation and Usage
### Prerequisites
Ensure you have Python and required libraries installed:
```sh
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Running the Model
1. Clone this repository:
   ```sh
   git clone https://github.com/APPANAHARINI1234/DiabetesPredictionML.git
   ```

2. Run the script:
   ```sh
   python Project_Diabetes.ipynb
   ```

## Future Improvements
- Implement additional machine learning algorithms (e.g., Random Forest, SVM).
- Enhance data preprocessing and feature selection.
- Develop a web-based interface for user-friendly predictions.
