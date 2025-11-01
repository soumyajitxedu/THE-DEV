import matplotlib.pyplot as plt 
import numpy as np 
story = np.array(["minecraft","hacking","bakochodi","python","edit","typograpy"])
times = np.array([4,2,3,1,3,1])

plt.barh(story , times)
plt.title("MY INSTA AVG")
plt.xlabel("data")
plt.ylabel("counts", color = "green",
           size=30)
plt.show()