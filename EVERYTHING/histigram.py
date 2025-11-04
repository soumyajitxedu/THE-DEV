import numpy as np 
import matplotlib.pyplot as plt 
scores = np.random.normal(loc= 80, scale=10,size=1000)
scores = np.clip(scores,0,100)
plt.hist(scores,color="green",alpha=0.8,bins=100,edgecolor="black")
plt.xlabel("exam scores",size=15,color="red")
plt.ylabel("number of students",size=15,color="red")
plt.title("im done", size=22, color="blue")

plt.show()