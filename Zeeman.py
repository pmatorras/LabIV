import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

data = np.loadtxt('Zeeman.csv', delimiter='\t', skiprows=2)
headers = np.genfromtxt('Zeeman.csv', delimiter='\t', max_rows=1, dtype=str)
x = data[:, 2]
y = data[:, 5]
y_err_low = data[:, 3]  # Column containing lower error values
y_err_high = data[:, 4]  # Column containing upper error values

def cubic_func(x, a, b, c):
    return a * x**3 + b * x**2 + c * x

popt, pcov = curve_fit(cubic_func, x, y)
a_fit, b_fit, c_fit = popt

x_fit = np.linspace(min(x)-0.5, max(x)+0.5, 100)
y_fit = cubic_func(x_fit, a_fit, b_fit, c_fit)

plt.errorbar(x, y, yerr=[y-y_err_low, y_err_high-y], fmt='o', label='Data')
plt.plot(x_fit, y_fit, color='red', label='Fit')
plt.xlabel(headers[2])
plt.ylabel(headers[3])
plt.xlim(min(x_fit), max(x_fit))
plt.legend()
plt.grid()
plt.minorticks_on()  # Enable minor ticks
plt.yticks(np.arange(0.5, 0.85, step=0.05))  # Adjust the number of ticks on the x-axis
plt.xticks(np.arange(6.5,11.5, step=0.5))  # Adjust the number of ticks on the y-axis
plt.grid(which='minor', linestyle='dashed', linewidth=0.5, alpha=0.75)  # Customize minor gridlines
plt.savefig('example_plot.png')

