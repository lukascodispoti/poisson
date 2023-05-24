/**
 * @file poisson.cc
 * @author lukascodispoti
 * @brief Solve the poisson equation in a 3D periodic domain using a central
 * finite difference scheme and the Jacobi method. File i/o is done using HDF5.
 * The program is parallelized using MPI. The domain is decomposed into slices
 * along the x axis. The boundaries are exchanged after each iteration.
 * @version 0.1
 * @date 2023-05-22
 *
 * @copyright Copyright (c) 2023
 *
 */

#include <hdf5.h>
#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "poisson.h"

int main(int argc, char **argv) {
    /* set non-buffered stdout */
    setbuf(stdout, NULL);

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int M = 1024;

    hsize_t Nloc = M / size;
    hsize_t offset = rank * Nloc;
    rank == size - 1 ? Nloc += M % size : Nloc;
    printf("rank %d, Nloc %" PRIdHSIZE " offset %" PRIdHSIZE "\n", rank, Nloc,
           offset);

    /* get input filename and datatset to read from the command line */
    char inputfile[100], inputdset[100];
    bool restart = false;
    char restartfile[100], restartdset[100] = "phi";
    if (argc != 3 && argc != 4 && argc != 5) {
        printf("Usage: %s <inputfile> <dataset> [<restartfile> <restartdset>]\n", argv[0]);
        exit(1);
    }
    strcpy(inputfile, argv[1]);
    strcpy(inputdset, argv[2]);
    if (argc > 3) {
        restart = true;
        strcpy(restartfile, argv[3]);
    }
    if (argc > 4) {
        strcpy(restartdset, argv[4]);
    }

    std::vector<float> f(Nloc * M * M);
    read1D(f, inputfile, inputdset, Nloc, offset, M);

    std::vector<float> phi(Nloc * M * M, 0);
    if (restart) {
        char dset[] = "phi";
        read1D(phi, restartfile, dset, Nloc, offset, M);
    }
    std::vector<float> phinew(Nloc * M * M, 0);
    std::vector<float> left(M * M);
    std::vector<float> right(M * M);

    const float tol = 1e-5;
    const int max_iter = 10000;
    const int dump_interval = 100;
    int iter = 0;
    float res = 1e10;
    while (res > tol && iter < max_iter) {
        update(f, phi, phinew, left, right, Nloc, M);
        res = residual(f, phi, left, right, Nloc, M);
        exchange(phi, left, right, Nloc, M);
        iter++;
        if (!rank) printf("iter: %d, residual: %f\n", iter, res);
        if (iter % dump_interval == 0) {
            char fname[100] = "phi.h5";
            char dset[100] = "phi";
            write1D(phi, fname, dset, Nloc, offset, M);
        }
    }

    MPI_Finalize();
    return 0;
}