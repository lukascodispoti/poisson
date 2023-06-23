import sys
from matplotlib import pyplot as plt
from matplotlib import style
import pandas as pd
import matplotlib
matplotlib.use('Agg')
style.use('fast')

dfs = []
labels = []
files = sys.argv[1:]
for f in files:
    df = pd.read_csv(f)
    print(df.head())
    dfs.append(df)
    labels.append(f)

fix, ax = plt.subplots(1, 1)
for df, f in zip(dfs, labels):
    ax.plot(df['iter'], df['residual'], label=f, rasterized=True)
ax.legend()
ax.set_xlabel('iteration')
ax.set_ylabel('residual')
ax.set_yscale('log')
plt.savefig('convergence-plot.png')
