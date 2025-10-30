import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("Chart data.csv")
print(df)
row = df["view","date"]
print(row)

