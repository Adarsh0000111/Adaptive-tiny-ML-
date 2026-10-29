import pandas as pd

file_path = "dataset/raw/DNN-EdgeIIoT-dataset.csv"

print("Loading dataset...")

df = pd.read_csv(file_path)

print("\nDataset loaded successfully!")

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())