# program fitting the exchange interaction
# model curve: Bethe-Slater function
import numpy as np, pylab, tkinter
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from decimal import *

print("Loop begin")

# definition of the Bethe-Slater function
def func(x,a,b,c):
    return 4*a*((x/c)**2)*(1-b*(x/c)**2)*np.exp(-(x/c)**2)

# exchange coeff table (data to fit)
rdata, Jdata = np.loadtxt('exchange_fcc_ni.dat', usecols=(0,1), unpack=True)
plt.plot(rdata, Jdata, 'b-', label='data')

# perform the fit
popt, pcov = curve_fit(func, rdata, Jdata, bounds=([0.0,-1.0,0.0], [100.,5.,5.]))
plt.plot(rdata, func(rdata, *popt), 'r--', label='fit')

# print the fitted parameters
print("Parameters: a={:.10} (in meV), b={:.10} (adim), c={:.10} (in Ang)".format(*popt))

# ploting the result
plt.xlabel('r_ij')
pylab.xlim([0.0,6.5])
#pylab.ylim([-2.0,10.0])
plt.ylabel('J_ij')
plt.legend()
plt.show()

print("Loop end")
