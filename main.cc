/**
 * @file poisson.cc
 * @author lukascodispoti
 * @brief Solve the poisson equation in a 3D periodic domain using a central
 * finite difference scheme.
 * The program is parallelized using MPI. The domain is decomposed into slices
 * along the x axis. The boundaries are exchanged after each iteration.
 * File i/o is done using parallel HDF5.
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

    /* command line arguments */
    char inputfile[100], inputdset[100];
    char outputfile[100] = "phi.h5", outputdset[100] = "phi";
    bool restart = false;
    char restartfile[100], restartdset[100] = "phi";
    int method = 2;
    float omega = 1.8;
    strcpy(inputfile, argv[1]);
    strcpy(inputdset, argv[2]);
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h") || argc < 2) {
            if (!rank) {
                printf("Usage: %s <inputfile> <inputdset> [options]\n",
                       argv[0]);
                printf("Options:\n");
                printf("  -h, --help\t\t\tPrint this message\n");
                printf(
                    "  -m, --method <method>\t\t"
                    "0: Jacobi, 1: Gauss-Seidel, 2: SOR\n");
                printf(
                    "  -r, --restart <file>\t\tRestart from "
                    "restartfile\n");
                printf(
                    "  -d, --restartdset <dset>\tRestart from "
                    "restartdset\n");
                printf("  -o, --outputfile <file>\tOutput to outputfile\n");
                printf(
                    "  -t, --outputdset <dset>\tOutput to "
                    "outputdset\n");
                printf("  -w, --omega <omega>\t\tSOR parameter\n");
            }
            MPI_Finalize();
            return 0;
        }
        if (!strcmp(argv[i], "-r") || !strcmp(argv[i], "--restart")) {
            restart = true;
            strcpy(restartfile, argv[i + 1]);
            if (!endswith(restartfile, ".h5")) strcat(restartfile, ".h5");
        }
        if (!strcmp(argv[i], "-d") || !strcmp(argv[i], "--restartdset"))
            strcpy(restartdset, argv[i + 1]);
        if (!strcmp(argv[i], "-o") || !strcmp(argv[i], "--outputfile")) {
            strcpy(outputfile, argv[i + 1]);
            if (!endswith(outputfile, ".h5")) strcat(outputfile, ".h5");
        }
        if (!strcmp(argv[i], "-t") || !strcmp(argv[i], "--outputdset"))
            strcpy(outputdset, argv[i + 1]);
        if (!strcmp(argv[i], "-m") || !strcmp(argv[i], "--method"))
            method = atoi(argv[i + 1]);
        if (!strcmp(argv[i], "-w") || !strcmp(argv[i], "--omega"))
            omega = atof(argv[i + 1]);
    }

    if (!rank) {
        printf("inputfile: %s\n", inputfile);
        printf("inputdset: %s\n", inputdset);
        printf("method: %d\n", method);
        if (restart) {
            printf("restartfile: %s\n", restartfile);
            printf("restartdset: %s\n", restartdset);
        }
        printf("outputfile: %s\n", outputfile);
        printf("outputdset: %s\n", outputdset);
    }
    if (method == 1) omega = 1.0;
    set_omega(omega);

    /* get the gridsize from the first dimension of the first datatset */
    hsize_t M = get_gridsize(inputfile, inputdset);
    if (!rank) printf("Gridsize M = %" PRIdHSIZE "\n", M);

    hsize_t Nloc = M / size;
    hsize_t offset = rank * Nloc;
    rank == size - 1 ? Nloc += M % size : Nloc;

    /* select method */
    void (*update)(std::vector<float> &, std::vector<float> &, hsize_t,
                   const hssize_t);
    if (method == 0) {
        update = &Jacobi;
        if (!rank) printf("Jacobi method\n");
    } else if (method == 1) {
        update = &SOR;
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

    std::vector<float> phi((Nloc + 2) * M * M, 0);
    if (restart) read1D(phi, restartfile, restartdset, Nloc, offset, M);

    /* pad phi with M x M zeros at the beginning */
    int pad = M * M;
    phi.insert(phi.begin(), pad, 0);

    const float tol = 1e-5;
    const int max_iter = 10000;
    const int dump_interval = 100;
    int iter = 0;
    float res = 1e10;

    /* create the residual file */
    FILE *fp;
    char fname[100] = "residual.csv";
    fp = fopen(fname, "w");
    fprintf(fp, "iter,residual\n");
    fclose(fp);

    /* main loop */
    while (res > tol && iter < max_iter) {
        update(f, phi, Nloc, M);
        res = residual(f, phi, Nloc, M);
        iter++;
        if (!rank) printf("iter: %d, residual: %f\n", iter, res);
        if (iter % dump_interval == 0)
            write1D(phi, outputfile, outputdset, Nloc, offset, M, pad);
        if (!rank) {
            fp = fopen(fname, "a");
            fprintf(fp, "%d,%f\n", iter, res);
            fclose(fp);
        }
    }
    write1D(phi, outputfile, outputdset, Nloc, offset, M, pad);

    MPI_Finalize();
    return 0;
}