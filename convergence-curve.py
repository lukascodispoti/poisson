import pandas as pd
import matplotlib.pyplot as plt
import sys

files = sys.argv[1:]
fix, ax = plt.subplots(1, 1)
for f in files:
    df = pd.read_csv(f, names=['iter', 'res'])
    ax.plot(df['iter'], df['res'], label=f)
ax.legend()
ax.set_xlabel('iteration')
ax.set_ylabel('residual')
ax.set_yscale('log')
plt.savefig('convergence-plot.png', dpi=500)
