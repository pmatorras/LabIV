import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

data = np.loadtxt('Diffraction.csv', delimiter='\t', skiprows=1)
headers = np.genfromtxt('Diffraction.csv', delimiter='\t', max_rows=1, dtype=str)
x      = data[:, 1]
y1     = data[:, 5]
y1_err = data[:, 6]  # Column containing lower error values
y2     = data[:, 10]
y2_err = data[:, 11]  # Column containing lower error values

def linear_func(x, a):#, b, c):
    return a * x# + b * x**2 + c * x

popt1, pcov1 = curve_fit(linear_func, x, y1)
a1_fit  = popt1
popt2, pcov2 = curve_fit(linear_func, x, y2)
a2_fit  = popt2

x_fit = np.linspace(min(x)-0.001, max(x)+0.001, 100)
y1_fit = linear_func(x_fit, a1_fit)#, b_fit, c_fit)
y2_fit = linear_func(x_fit, a2_fit)#, b_fit, c_fit)

fit_errors1 = np.sqrt(np.diag(pcov1))
fit_errors2 = np.sqrt(np.diag(pcov2))

# Print the fit errors
for param, error in zip(a1_fit, fit_errors1):
    print(f"Parameter: {param:.4f} ± {error:.4f}")
for param, error in zip(a2_fit, fit_errors2):
    print(f"Parameter: {param:.4f} ± {error:.4f}")

# D = k/d Va⁻1/2, k=\frac{\sqrt{2} nhL}{\sqrt{em}
# -> h = k\frac{\sqrt{em}{\sqrt{2} nL} and slope = k/d -> k = slope *d
# h = slope * d * \frac{\sqrt{em}{\sqrt{2} nL}
d10 = 0.213E-9 #m
d11 = 0.123E-9 #m
L    = 13.0E-2 #m
errL = 0.2E-2
n    = 1
qe   = 1.60217663E-19
me   = 9.1093837E-31
h1   = a1_fit * 0.01* d10 * np.sqrt(qe*me)/(np.sqrt(2)*n*L)
errh1= h1*np.sqrt((errL/L)**2+(fit_errors1/a1_fit)**2)
h2   = a2_fit * 0.01* d11 * np.sqrt(qe*me)/(np.sqrt(2)*n*L)
errh2= h2*np.sqrt((errL/L)**2+(fit_errors2/a2_fit)**2)

print("planck is", h1,'+-',errh1, h2,'+-',errh2)
plt.errorbar(x, y1, yerr=y1_err, fmt='o', color='red')
plt.plot(x_fit, y1_fit, color='red', label='Dmin')
plt.errorbar(x, y2, yerr=y1_err, fmt='o', color='blue')
plt.plot(x_fit, y2_fit, color='blue', label='Dmax')
plt.xlabel('V^1/2 [V^1/2]')
plt.ylabel('D [cm]')
plt.xlim(min(x_fit), max(x_fit))
plt.legend()
plt.grid()
plt.minorticks_on()  # Enable minor ticks
plt.xticks(np.arange(0.014, 0.021, step=0.001))  # Adjust the number of ticks on the x-axis
#plt.yticks(np.arange(2,6, step=0.5))  # Adjust the number of ticks on the y-axis
plt.grid(which='minor', linestyle='dashed', linewidth=0.5, alpha=0.75)  # Customize minor gridlines
plt.savefig('plot_Diffraction.png')

