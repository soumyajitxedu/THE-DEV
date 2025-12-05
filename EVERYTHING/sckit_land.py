import pandas as pd 
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
data = np.array({
    'StudyHours' : [1,2,3,4,5,6,7,8,9,10],
    'testScore' : [20,21,22,23,42,43,33,32,12]
})
df = pd.DataFrame(data)