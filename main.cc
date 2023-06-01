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

    /* get command line arguments */
    char inputfile[100], inputdset[100];
    bool restart = false;
    char restartfile[100], restartdset[100] = "phi";
    int method = 2;
    if (argc < 3 || argc > 6) {
        if (!rank)
            printf(
                "Usage: %s <inputfile> <dataset> [<method: 0 "
                "jacobi, 1 "
                "gauss-seidel, 2 sor> <restartfile> <restartdset>]\n",
                argv[0]);
        MPI_Finalize();
    }
    strcpy(inputfile, argv[1]);
    strcpy(inputdset, argv[2]);
    if (argc > 3) method = atoi(argv[3]);
    if (argc > 4) {
        restart = true;
        strcpy(restartfile, argv[4]);
    }
    if (argc > 5) strcpy(restartdset, argv[5]);

    if (!rank) {
        printf("inputfile: %s\n", inputfile);
        printf("inputdset: %s\n", inputdset);
        printf("method: %d\n", method);
        if (restart) {
            printf("restartfile: %s\n", restartfile);
            printf("restartdset: %s\n", restartdset);
        }
    }

    /* get the gridsize from the first dimension of the first datatset */
    hsize_t M;
    hsize_t dims[3];
    hid_t file_id, dset_id, dataspace;
    file_id = H5Fopen(inputfile, H5F_ACC_RDONLY, H5P_DEFAULT);
    dset_id = H5Dopen(file_id, inputdset, H5P_DEFAULT);
    dataspace = H5Dget_space(dset_id);
    H5Sget_simple_extent_dims(dataspace, dims, NULL);
    M = dims[0];
    H5Sclose(dataspace);
    H5Dclose(dset_id);
    H5Fclose(file_id);
    if (!rank) printf("Gridsize M = %" PRIdHSIZE "\n", M);

    hsize_t Nloc = M / size;
    hsize_t offset = rank * Nloc;
    rank == size - 1 ? Nloc += M % size : Nloc;

    /* select method */
    void (*update)(std::vector<float> &, std::vector<float> &,
                   std::vector<float> &, std::vector<float> &, hsize_t,
                   const hssize_t);
    if (method == 0) {
        update = &Jacobi;
        if (!rank) printf("Jacobi method\n");
    } else if (method == 1) {
        update = &GaussSeidel;
        if (!rank) printf("Gauss-Seidel method\n");
    } else if (method == 2) {
        update = &SOR;
        if (!rank) printf("SOR method\n");
    } else {
        printf("Invalid method\n");
        exit(1);
    }

    /* read input */
    std::vector<float> f(Nloc * M * M);
    read1D(f, inputfile, inputdset, Nloc, offset, M);

    std::vector<float> phi(Nloc * M * M, 0);
    if (restart) {
        char dset[] = "phi";
        read1D(phi, restartfile, dset, Nloc, offset, M);
    }
    std::vector<float> left(M * M);
    std::vector<float> right(M * M);

    const float tol = 1e-5;
    const int max_iter = 10000;
    const int dump_interval = 100;
    int iter = 0;
    float res = 1e10;

    /* output file and dataset */
    char outfname[100] = "phi.h5";
    char outdset[100] = "phi";

    /* create the residual file */
    FILE *fp;
    char fname[100] = "residual.csv";
    fp = fopen(fname, "w");
    fprintf(fp, "iter,residual\n");
    fclose(fp);

    /* main loop */
    while (res > tol && iter < max_iter) {
        update(f, phi, left, right, Nloc, M);
        res = residual(f, phi, left, right, Nloc, M);
        exchange(phi, left, right, Nloc, M);
        iter++;
        if (!rank) printf("iter: %d, residual: %f\n", iter, res);
        if (iter % dump_interval == 0)
            write1D(phi, outfname, outdset, Nloc, offset, M);
        if (!rank) {
            fp = fopen(fname, "a");
            fprintf(fp, "%d,%f\n", iter, res);
            fclose(fp);
        }
    }
    write1D(phi, outfname, outdset, Nloc, offset, M);

    MPI_Finalize();
    return 0;
}