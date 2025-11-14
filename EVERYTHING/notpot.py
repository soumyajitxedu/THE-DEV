import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
# Load the dataset

student_data = [
    {'StudentID': 101, 'Gender': 'Male', 'Math_Score': 85, 'Science_Score': 78},
    {'StudentID': 102, 'Gender': 'Female', 'Math_Score': 92, 'Science_Score': 95},
    {'StudentID': 103, 'Gender': 'Female', 'Math_Score': 68, 'Science_Score': 70},
    {'StudentID': 104, 'Gender': 'Male', 'Math_Score': 75, 'Science_Score': 65},
    {'StudentID': 105, 'Gender': 'Male', 'Math_Score': None, 'Science_Score': 88}, # Missing value (None)
    {'StudentID': 106, 'Gender': 'Female', 'Math_Score': 98, 'Science_Score': 99},
    {'StudentID': 107, 'Gender': 'Male', 'Math_Score': 55, 'Science_Score': 60},
    {'StudentID': 108, 'Gender': 'Female', 'Math_Score': 72, 'Science_Score': 80},
    {'StudentID': 109, 'Gender': 'Male', 'Math_Score': 80, 'Science_Score': 75},
    {'StudentID': 110, 'Gender': 'Female', 'Math_Score': np.nan, 'Science_Score': 77}, # Missing value (np.nan)
    {'StudentID': 111, 'Gender': 'Male', 'Math_Score': 62, 'Science_Score': 68},
    {'StudentID': 112, 'Gender': 'Female', 'Math_Score': 88, 'Science_Score': 90}
]

df = pd.DataFrame(student_data)
print(df)
# Data Cleaning: Handle missing values by filling them with the mean score
df["Math_Score"].fillna(df["Math_Score"].mean(), inplace=True)
df["Science_Score"].fillna(df["Science_Score"].mean(), inplace=True)
#determine the fail and pass
df["result"] = np.where((df["Math_Score"] < 60) | (df["Science_Score"] < 60), "Fail", "Pass")
print(df)
#stats
mean_math = df['Math_Score'].mean()
total_students = df.shape[0]
pass_count = df[df['result'] == 'Pass'].shape[0]
fail_count = df[df['result'] == 'Fail'].shape[0]
print(f"Mean Math Score: {mean_math}")
print(f"Total Students: {total_students}")  
print(f"Pass Count: {pass_count}")
print(f"Fail Count: {fail_count}")
#visuals
x = ['Pass', 'Fail']
y = [pass_count, fail_count]
plt.scatter(x,y,color="red",marker="*")
plt.show()