import pandas as pd

FILE = "dataset/raw/DNN-EdgeIIoT-dataset.csv"

print("Loading dataset...")
df = pd.read_csv(FILE, low_memory=False)

print("\n================================")
print("DATASET SHAPE")
print("================================")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\n================================")
print("ALL COLUMNS")
print("================================")

for i, column in enumerate(df.columns):
    print(i, "->", column)

print("\n================================")
print("DATA TYPES")
print("================================")
print(df.dtypes)

print("\n================================")
print("TARGET DISTRIBUTION")
print("================================")
print(df["Attack_label"].value_counts())

print("\n================================")
print("ATTACK TYPES")
print("================================")
print(df["Attack_type"].value_counts())