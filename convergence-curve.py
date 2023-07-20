import sys
from matplotlib import pyplot as plt
from matplotlib import style
import numpy as np
import matplotlib
matplotlib.use('Agg')
style.use('fast')

arrs = []
labels = []
files = sys.argv[1:]
for f in files:
    res = np.loadtxt(f, skiprows=0)
    print(res.shape)
    arrs.append(res)
    labels.append(f)

fix, ax = plt.subplots(1, 1)
for arr, f in zip(arrs, labels):
    ax.plot(arr, label=f, rasterized=True)
ax.legend()
ax.set_xlabel('iteration')
ax.set_ylabel('residual')
ax.set_yscale('log')
plt.savefig('convergence-plot.pdf')
