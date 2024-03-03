import pickle
import pandas as pd

with open('pv.pkl', 'rb') as file:
    pv = pickle.load(file)

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Example solar exposure data
x = np.arange(0, 288)
y = pv  # Make sure 'pv' is defined before this line

transition_start, transition_end = 100, 188

def sigmoid(x, k, x0):
    return 1 / (1 + np.exp(-k * (x - x0)))

def gaussian(x, d, e, f):
    return d * np.exp(-((x - e)**2) / (2 * f**2))

def quadratic(x, a, b, c):
    return a * x**2 + b * x + c

def combined_model(x, a, b, c, d, e, f, k, x0):
    S = sigmoid(x, k, x0)
    return (1 - S) * gaussian(x, d, e, f) + S * quadratic(x, a, b, c)


# Fit the curve
params, _ = curve_fit(combined_model, x, y, maxfev=100000)
# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(x, y, label='Actual Solar Exposure')
plt.plot(x, combined_model(x, *params), color='red', label='Fitted Curve')
plt.xlabel('Time of Day')
plt.ylabel('Solar Exposure')
plt.title('Solar Exposure Fitting')
plt.legend()
plt.show()
