# `poisson`

An `mpi`-parallel poisson equation solver written in `C++`.

The poisson equation is solved on an `M x M x M` grid in a cubic domain with
periodic boundary conditions using a central difference scheme.

Parallel `hdf5` is used for i/o.

## Available solvers are:

- Jacobi
- Gauss-Seidel
- Successive over-relaxation (SOR)

<p align="center">
    <img src="convergence-plot.png"  width="500">
</p>

## Dependencies

- `make`
- `mpi`
- `hdf5-mpi`
