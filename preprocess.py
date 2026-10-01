import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_FILE = os.path.join(BASE_DIR, "raw_student_performance.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Load dataset
df = pd.read_csv(RAW_FILE)
print("\n--- First 5 rows ---")
print(df.head())

print("\n--- Dataset information ---")
print(df.info())

print("\n--- Missing values before cleaning ---")
print(df.isnull().sum())

# 2. Handle missing values
numeric_cols = ["age", "cgpa", "annual_salary", "internships", "assessment_score"]
categorical_cols = ["gender", "department"]

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# 3. Visualize outliers before removing them
plt.figure(figsize=(8, 5))
sns.boxplot(data=df[numeric_cols])
plt.title("Numeric Features Before Outlier Removal")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "01_before_outliers.png"), dpi=150)
plt.close()

# 4. Remove outliers using IQR
def remove_iqr_outliers(dataframe, columns):
    result = dataframe.copy()
    removed_rows = set()

    for col in columns:
        q1 = result[col].quantile(0.25)
        q3 = result[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        bad_index = result.index[(result[col] < lower) | (result[col] > upper)]
        removed_rows.update(bad_index.tolist())

    result = result.drop(index=list(removed_rows))
    return result, len(removed_rows)

df, removed_count = remove_iqr_outliers(df, ["annual_salary", "assessment_score", "cgpa"])

# 5. One-hot encode categorical columns
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)

# 6. Standardize numerical features
scaler = StandardScaler()
scale_cols = ["age", "cgpa", "annual_salary", "internships", "assessment_score"]
df[scale_cols] = scaler.fit_transform(df[scale_cols])

# 7. Save processed dataset
output_file = os.path.join(OUTPUT_DIR, "processed_student_performance.csv")
df.to_csv(output_file, index=False)

# 8. Save summary
summary = pd.DataFrame({
    "item": [
        "Original rows",
        "Rows removed as outliers",
        "Final rows",
        "Final columns",
        "Missing values after cleaning"
    ],
    "value": [
        len(pd.read_csv(RAW_FILE)),
        removed_count,
        len(df),
        len(df.columns),
        int(df.isnull().sum().sum())
    ]
})
summary.to_csv(os.path.join(OUTPUT_DIR, "preprocessing_summary.csv"), index=False)

# 9. Visualize final numerical distributions
plt.figure(figsize=(8, 5))
sns.boxplot(data=df[scale_cols])
plt.title("Numeric Features After Outlier Removal and Standardization")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "02_after_preprocessing.png"), dpi=150)
plt.close()

print("\n--- Missing values after cleaning ---")
print(df.isnull().sum())
print(f"\nRows removed as outliers: {removed_count}")
print(f"Processed dataset saved to: {output_file}")
print("\n--- Final data preview ---")
print(df.head())
