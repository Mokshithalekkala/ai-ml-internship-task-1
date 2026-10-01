# Task 1 - Data Cleaning & Preprocessing

## AI & ML Internship

This project completes **Task 1: Data Cleaning & Preprocessing**.

The task asks us to learn how to clean and prepare raw data for machine learning using Python, Pandas, NumPy, Matplotlib/Seaborn, including handling nulls, encoding and feature scaling.

## Dataset

A small **Student Performance dataset** is included as `raw_student_performance.csv`. It contains numeric and categorical columns, missing values, and an intentional salary outlier so that the preprocessing steps can be demonstrated clearly.

## What this project does

1. Imports and explores the dataset.
2. Checks data types and missing values.
3. Fills numeric missing values using the median.
4. Fills categorical missing values using the mode.
5. Visualizes numeric outliers using boxplots.
6. Removes outliers using the IQR method.
7. Converts categorical variables using one-hot encoding.
8. Standardizes numerical features using `StandardScaler`.
9. Saves the processed dataset and preprocessing summary.
10. Saves before/after boxplot images.

## Files

- `raw_student_performance.csv` - raw dataset
- `preprocess.py` - complete preprocessing program
- `requirements.txt` - required Python libraries
- `outputs/processed_student_performance.csv` - cleaned and transformed dataset
- `outputs/preprocessing_summary.csv` - summary of preprocessing
- `outputs/01_before_outliers.png` - outlier visualization before removal
- `outputs/02_after_preprocessing.png` - visualization after preprocessing
- `README.md` - project explanation

## How to run

Open the project folder in VS Code.

Create/activate a virtual environment if desired, then install packages:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run:

```powershell
python preprocess.py
```

The processed files will be created inside the `outputs` folder.

## Interview Questions - Short Answers

### 1. What are the different types of missing data?
MCAR means missing completely at random, MAR means missing at random based on observed variables, and MNAR means the missingness is related to the missing value itself.

### 2. How do you handle categorical variables?
Categorical variables can be converted to numerical form using methods such as one-hot encoding or label encoding.

### 3. Normalization vs standardization
Normalization usually scales values to a fixed range such as 0 to 1. Standardization transforms values so they have approximately mean 0 and standard deviation 1.

### 4. How do you detect outliers?
Common methods include boxplots, the IQR method, and z-scores.

### 5. Why is preprocessing important in ML?
Preprocessing improves data quality and puts features into a suitable form for machine-learning algorithms.

### 6. One-hot encoding vs label encoding
One-hot encoding creates separate binary columns for categories. Label encoding assigns integer labels to categories.

### 7. How do you handle data imbalance?
Possible approaches include resampling, class weights, oversampling/undersampling, and suitable evaluation metrics.

### 8. Can preprocessing affect model accuracy?
Yes. Incorrect preprocessing can reduce model performance, while appropriate preprocessing can improve the quality of the data supplied to a model.

## GitHub Submission

The internship instructions require a separate GitHub repository for each task and ask you to include code, datasets, screenshots if any, and a short README.

Suggested repository name:

`ai-ml-internship-task-1`

After creating the repository, upload this entire project folder.
