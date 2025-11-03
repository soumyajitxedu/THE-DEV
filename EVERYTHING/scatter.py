import matplotlib.pyplot as plt
import numpy as np 

Study_Hours =np.array([2, 5, 1, 8, 4, 3, 7, 6, 9, 1])
Exam_Score = np.array([65, 85, 50, 95, 75, 70, 90, 88, 98, 55])
plt.scatter(Study_Hours, Exam_Score, color="green", marker='o',label="study")
plt.title("Correlation: Study Hours vs. Exam Score")
plt.xlabel("study")
plt.ylabel("exam")
plt.legend()
plt.show()