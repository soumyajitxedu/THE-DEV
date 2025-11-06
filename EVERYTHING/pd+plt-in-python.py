import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
data = pd.read_csv("pokemon.csv")
tq = (data["abilities"].value_counts()) 
plt.barh(tq.index, tq)
plt.show()
