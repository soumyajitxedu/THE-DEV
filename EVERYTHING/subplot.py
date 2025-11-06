import matplotlib.pyplot as plt 
import numpy as np 
x = np.array([1,2,3,4,5,6,7,8,9,10,11])
y = np.array([12,21,34,43,54,45,56,65,78.87,89,98])
figure,axes = plt.subplots(3,2)
axes[0,0].plot(x+34,x*2+54)
axes[0,1].hist(x+37,x*2+54)
axes[2,0].scatter(x+34,x*2+4)
axes[1,1].plot(y,x)
plt.show()
