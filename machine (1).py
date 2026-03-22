import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer

df = pd.read_csv(r"C:\Users\Mo\Desktop\tries\fifa_world_rankings_jan_2026.csv", sep=',')
df.columns = [c.strip().lower() for c in df.columns]
print("Columns available:", df.columns.tolist())

col = input("Enter categorical column to encode: ").strip().lower()
if col in df.columns:
    encoder = OneHotEncoder(sparse_output=False)
    encoded = encoder.fit_transform(df[[col]])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out([col]), index=df.index)
    df = df.drop(columns=[col]).join(encoded_df)

    missing_percent = df.isnull().sum() / len(df) * 100
    missing_percent = missing_percent[missing_percent > 0].sort_values(ascending=False)
    print(missing_percent)

    plt.figure(figsize=(10,5))
    plt.bar(missing_percent.index, missing_percent.values)
    plt.title("Missing Values Percentage")
    plt.xlabel("Columns")
    plt.ylabel("Percentage")
    plt.xticks(rotation=45)
    plt.show()

    num_cols = df.select_dtypes(include=['int64','float64']).columns
    for col in num_cols:
        missing = df[col].isnull().sum()
        if missing == 0:
            continue
        if missing / len(df) <= 0.1:
            df[col].fillna(df[col].mean(), inplace=True)
        elif missing / len(df) <= 0.3:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            knn = KNNImputer(n_neighbors=3)
            df[col] = knn.fit_transform(df[[col]])

    print(df.head())
else:
    print(f"Column '{col}' not found.")
    