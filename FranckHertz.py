import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

data = np.loadtxt('FranckHertz.csv', delimiter='\t', skiprows=4)
headers = np.genfromtxt('FranckHertz.csv', delimiter='\t', max_rows=1, dtype=str)
units   = np.genfromtxt('FranckHertz.csv', delimiter='\t', max_rows=2, dtype=str)
print(units)
x      = data[:, 0]
y      = data[:, 1]

def linear_func(x, a):#, b, c):
    return a * x# + b * x**2 + c * x

popt1, pcov1 = curve_fit(linear_func, x, y)
a1_fit  = popt1

x_fit = np.linspace(min(x)-0.001, max(x)+0.001, 100)
y_fit = linear_func(x_fit, a1_fit)#, b_fit, c_fit)

plt.plot(x, y, label='Data')
#plt.plot(x_fit, y_fit, color='red', label='Fit')
plt.xlabel('U1 [V]')
plt.ylabel('I [nA]')
plt.xlim(0,50)#min(x_fit), max(x_fit))
plt.ylim(0,50)
plt.legend()
plt.grid()
plt.minorticks_on()  # Enable minor ticks
#plt.xticks(np.arange(0.014, 0.021, step=0.001))  # Adjust the number of ticks on the x-axis
#plt.xticks(np.arange(6.5,11.5, step=0.5))  # Adjust the number of ticks on the y-axis
plt.grid(which='minor', linestyle='dashed', linewidth=0.5, alpha=0.75)  # Customize minor gridlines
plt.savefig('plot_FranckHertz.png')

