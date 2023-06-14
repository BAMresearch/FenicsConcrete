import numpy as np
from scipy.stats import expon
import math
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)

x = np.linspace(expon.ppf(0.01),
                expon.ppf(0.99), 100)


rv = expon(scale=1/(6*math.log(10)))
print(1/(6*math.log(10)))
print(rv.stats(moments='mvsk')[0])

ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen1 pdf')



""" rv2 = expon(loc = 2, scale=1/(6*math.log(10)))
ax.plot(x, rv2.pdf(x), 'k-', lw=2, label='frozen2 pdf')
print(rv2.stats(moments='mvsk')[0]) """
plt.show()