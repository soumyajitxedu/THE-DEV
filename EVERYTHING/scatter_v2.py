import matplotlib.pyplot as plt
import numpy as np

x1 = np.array([2, 5, 1, 8, 4, 3, 7, 6, 9, 1])
y1 = np.array([65, 85, 50, 95, 75, 70, 90, 88, 98, 55])
x2 = np.array([3, 7, 1, 5, 2, 4, 6, 8, 3, 5])
y2 = np.array([70, 92, 45, 80, 55, 68, 85, 90, 60, 75])
x3 = np.array([1, 9, 5, 2, 8, 4, 7, 3, 6, 10])
y3 = np.array([80, 50, 65, 90, 55, 75, 40, 95, 70, 45])
plt.scatter(x1,y1,color="skyblue",alpha=0.5,label="class1")
plt.scatter(x2,y2,color="green",alpha=0.7, label="class2")
plt.scatter(x3,y3,color="purple",alpha=0.9,label = "class3")
plt.title("student datasets", size =23,color="blue")
plt.legend()
plt.xlabel("study hours", size=15, color="red")
plt.ylabel("exam score", size=15, color="red")
plt.show()