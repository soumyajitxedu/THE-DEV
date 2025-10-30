import matplotlib.pyplot as plt
import numpy as np 

x = np.array([1,2,3,48,5,6,7])
y = np.array([12,37,112,3,4,5,6])
y1 = np.array([20, 25, 30, 45, 50, 55, 60])  # steadily increasing values
plt.title("my data ?", fontsize=16,
          family="arial",
          fontweight="bold",
          color="#256ac5")
plt.xlabel("baby", fontsize=20,
           family ="Arial",
           fontweight="bold",
           color="#5f5eaa")
plt.ylabel("just", fontsize=41,
           family="arial",
           fontweight="bold")

plt.plot(x,y, marker =".",
         markersize=10,
         markerfacecolor="#51aa5d",
         markeredgecolor="#000000",
         linestyle="solid",
         linewidth=4,
         color="#841FB3")
line_style = dict( marker =".",
         markersize=10,
         markerfacecolor="#51aa5d",
         markeredgecolor="#8B3939",
         linestyle="solid",
         linewidth=4,
         color="#D83BB6")
plt.tick_params(axis="both",
                colors="#00ff22")

plt.plot(x, y, **line_style) 
plt.plot(x, y1, **line_style)

plt.xticks

plt.show()