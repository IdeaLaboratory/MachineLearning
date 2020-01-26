import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

# READ DATA 
file_content=pd.read_csv('cancerReport.csv')

# PRINT DATA 
# print(file_content.head(10))
# print(file_content.tail())

sns.heatmap(file_content.iloc[:,:10].corr(), annot=True, square=True, cmap='coolwarm')
plt.show()
