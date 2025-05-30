import numpy as np
import pandas as pd
from jg import jg

# Load food data from Excel sheet
df = pd.read_excel('Option6_Proportion1.xlsx')  

# Remove first column (food item names)
df = df.drop(df.columns[0], axis=1)
df = df.T

# Convert to numpy array
data = df.to_numpy()

# Define nutritional requirements
requirements = [[2100.0, 52.5, 40.0, 989.0, 1.1, 138.0, 32.0, 201.0, 27.6,
    12.4, 550, 1.1, 1.1, 13.8, 4.6, 1.2, 363, 2.2, 41.6, 6.1, 8.0, 48.2]]

# Run jg algorithm
f, g = jg(data,requirements)
print(f,g)