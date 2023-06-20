import pandas as pd
import matplotlib.pyplot as plt
import sys

sor = pd.read_csv('residual-sor.csv', names=['iter', 'res'])
gs = pd.read_csv('residual-gs.csv', names=['iter', 'res'])
jac = pd.read_csv('residual-jacobi.csv', names=['iter', 'res'])

fix, ax = plt.subplots(1, 1)
files = sys.argv[1:]
for f in files:
    df = pd.read_csv(f, names=['iter', 'res'])
    ax.plot(df['iter'], df['res'], label=f)
ax.legend()
ax.set_xlabel('iteration')
ax.set_ylabel('residual')
ax.set_yscale('log')
plt.savefig('convergence-plot.pdf')
