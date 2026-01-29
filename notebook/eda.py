import pandas as pd

df = pd.read_csv("C:\\Users\\ASUS\\Desktop\\ISHIKA DEPLOYMENT\\churn_pred_mine\\data\\Telco-Customer-Churn.csv")


# preview first five rows
print(df.head())

print("dataset info")
print(df.info(),"\n")

#check for missing values
print("missing values in each column")
print(df.isnull().sum(),"\n")

print("descriptive statistics")
print(df.describe())

# target varibale distribution
# Step 6: Check target variable distribution
if 'Churn' in df.columns:
    print("\n===== Target Variable Distribution (Churn) =====")
    print(df['Churn'].value_counts())
else:
    print("\nChurn column not found!")


# Step 7: Analyze categorical variables
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
print("\n Unique values in categorical coloumn")
for col in categorical_cols:
    print(f"{col}:{df[col].unique()}\n")

# Step 8: analyze numeric variables
numeric_cols=df.select_dtypes(include=['int64','float64']).columns
print("\n===== Skewnessof numeric fetaures =====")
print(df[numeric_cols].skew())


