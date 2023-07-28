/**
 * @file main.cc
 * @author lukascodispoti (lukas.codispoti@gmail.com)
 * @brief Solve the Poisson equation using Jacobi, Gauss-Seidel or SOR method on
 * a cubid domain with periodic boundary conditions.
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

    if (!rank) {
        printf("\n   P O I S S O N\n\n");
        printf("Solving Poisson equation on %d processes\n", size);
    }

    // TODO: implement "welcome()" function?
    /* command line arguments */
    char inputfile[100] = "", inputdset[100] = "";
    char outputfile[100] = "phi.h5", outputdset[100] = "phi";
    bool restart = false;
    char restartfile[100], restartdset[100] = "phi";
    int method = 2;
    float omega = 1.8;
    int dump_interval = 100;
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
                printf("  -i, --interval <interval>\tDump interval\n");
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
        if (!strcmp(argv[i], "-i") || !strcmp(argv[i], "--interval"))
            dump_interval = atoi(argv[i + 1]);
    }
    strcpy(inputfile, argv[1]);
    strcpy(inputdset, argv[2]);

    if (!rank) {
        printf("Inputfile: %s\n", inputfile);
        printf("Inputdset: %s\n", inputdset);
        if (restart) {
            printf("Restartfile: %s\n", restartfile);
            printf("Restartdset: %s\n", restartdset);
        }
        printf("Outputfile: %s\n", outputfile);
        printf("Outputdset: %s\n", outputdset);
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
                   const hssize_t, MPI_Request *, MPI_Status *);
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
    float norm_f = 0.f;
    for (size_t i = 0; i < Nloc * M * M; i++) norm_f += f[i] * f[i];
    MPI_Allreduce(MPI_IN_PLACE, &norm_f, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    norm_f = sqrt(norm_f);
    if (!rank) printf("norm_f = %f\n", norm_f);

    std::vector<float> phi((Nloc + 2) * M * M, 0);
    if (restart) read1D(phi, restartfile, restartdset, Nloc, offset, M);

    /* pad phi with M x M zeros at the beginning */
    int pad = M * M;
    phi.insert(phi.begin(), pad, 0);

    if (restart) exchange(phi, Nloc, M);


    /* residual file */
    FILE *fp;
    const char fname[100] = "residual.csv";
    if (!rank) {
        if (!restart && !access(fname, F_OK)) remove(fname);
        fp = fopen(fname, "a");
        fclose(fp);
    }

    /* array to collect residuals, will be written to file at dump interval */
    std::vector<float> residuals(dump_interval);

    uint64_t iter = 0;
    const uint64_t max_iter = 1000000;
    const float tol = 1e-7;
    float res = 1e10, res_rel = 1e10;
    MPI_Request req[4];
    MPI_Status stat[4];
    /* main loop */
    while (res_rel > tol && iter < max_iter) {
        update(f, phi, Nloc, M, req, stat);
        res = residual(f, phi, Nloc, M);
        res_rel = res / norm_f;
        residuals[iter % dump_interval] = res_rel;
        iter++;
        if (!rank)
            printf("%07" PRIu64 ": res = %e, rel = %e\n", iter, res, res_rel);
        if (method != 0) MPI_Waitall(4, req, stat);
        if (iter % dump_interval == 0) {
            write1D(phi, outputfile, outputdset, Nloc, offset, M, pad);
            if (!rank) {
                fp = fopen(fname, "a");
                for (int i = 0; i < dump_interval; i++)
                    fprintf(fp, "%e\n", residuals[i]);
                fclose(fp);
            }
        }
    }
    write1D(phi, outputfile, outputdset, Nloc, offset, M, pad);

    MPI_Finalize();
    return 0;
}