import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('training_log.csv')
plt.plot(data['Epoch'], data['Train Loss'], label='Train Loss')
plt.plot(data['Epoch'], data['Val Loss'], label='Val Loss')
plt.legend()
plt.show()
