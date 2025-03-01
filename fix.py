import pandas as pd

df = pd.read_csv("file2.csv")  # Make sure file path is correct

# Count how many times each disease appears in the dataset
disease_counts = df["prognosis"].value_counts()
print(disease_counts)


