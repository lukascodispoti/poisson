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

const int M = 1024;
const float h = 2 * M_PI / M;
const float tol = 1e-5;
const int max_iter = 10000;

size_t loc_idx(hssize_t i, hssize_t j, hssize_t k);
/**
 * @brief Read the right hand side of the poisson equation from a hdf5 file.
 *
 * @param f The right hand side the poisson equation.
 * @param fname
 * @param Nloc
 * @param offset
 */
void read_rhs(std::vector<float> &f, char *fname, hsize_t Nloc, hsize_t offset);

float residual(std::vector<float> &f, std::vector<float> &phi,
               std::vector<float> &left, std::vector<float> &right,
               hsize_t Nloc);
/**
 * @brief We solve the poisson equation of the form:
 *
 *  ∂ 2 φ
 *  ----- = f
 *  ∂x 2
 *
 * using the centered difference scheme. This function updates the solution.
 *
 * @param f
 * @param phi
 * @param phinew
 * @param left
 * @param right
 * @param Nloc
 */
void update(std::vector<float> &f, std::vector<float> &phi,
            std::vector<float> &phinew, std::vector<float> &left,
            std::vector<float> &right, hsize_t Nloc);

/**
 * @brief Exchange the boundaries between the mpi ranks.
 *
 * @param phi
 * @param left
 * @param right
 * @param Nloc
 */
void exchange(std::vector<float> &phi, std::vector<float> &left,
              std::vector<float> &right, hsize_t Nloc);

void write_phi(std::vector<float> &phi, char *fname, hsize_t Nloc,
               hsize_t offset);

int main(int argc, char **argv) {
    /* set non-buffered stdout */
    setbuf(stdout, NULL);

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    hsize_t Nloc = M / size;
    hsize_t offset = rank * Nloc;
    rank == size - 1 ? Nloc += M % size : Nloc;
    printf("rank %d, Nloc %" PRIdHSIZE " offset %" PRIdHSIZE "\n", rank, Nloc,
           offset);

    /* get inputfilename from command line argument */
    char inputfile[100];
    if (argc != 2) {
        printf("Usage: %s <inputfile>\n", argv[0]);
        exit(1);
    }
    strcpy(inputfile, argv[1]);

    std::vector<float> f(Nloc * M * M);
    read_rhs(f, inputfile, Nloc, offset);

    std::vector<float> phi(Nloc * M * M, 0);
    std::vector<float> phinew(Nloc * M * M, 0);
    std::vector<float> left(M * M);
    std::vector<float> right(M * M);

    int iter = 0;
    float res = 1e10;
    while (res > tol && iter < max_iter) {
        update(f, phi, phinew, left, right, Nloc);
        res = residual(f, phi, left, right, Nloc);
        exchange(phi, left, right, Nloc);
        iter++;
        if (!rank) printf("iter: %d, residual: %f\n", iter, res);
        if (iter % 100 == 0) {
            char fname[100] = "phi.h5";
            write_phi(phi, fname, Nloc, offset);
        }
    }

    MPI_Finalize();
    return 0;
}

size_t loc_idx(hssize_t i, hssize_t j, hssize_t k) {
    i >= M ? i -= M : i;
    j >= M ? j -= M : j;
    k >= M ? k -= M : k;
    i < 0 ? i += M : i;
    j < 0 ? j += M : j;
    k < 0 ? k += M : k;
    return (i * M + j) * M + k;
}

void read_rhs(std::vector<float> &f, char *fname, hsize_t Nloc,
              hsize_t offset) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    hid_t file, dataset, dataspace, memspace;
    /* open the file */
    hid_t fapl_par = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(fapl_par, MPI_COMM_WORLD, MPI_INFO_NULL);
    file = H5Fopen(fname, H5F_ACC_RDONLY, fapl_par);
    H5Pclose(fapl_par);

    dataset = H5Dopen2(file, "div", H5P_DEFAULT);
    dataspace = H5Dget_space(dataset);

    hsize_t dims[3] = {M, M, M};
    H5Sget_simple_extent_dims(dataspace, dims, NULL);

    const hsize_t count[3] = {Nloc, M, M};
    const hsize_t start[3] = {offset, 0, 0};
    const hsize_t stride[3] = {1, 1, 1};
    const hsize_t block[3] = {1, 1, 1};
    H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, start, stride, count, block);

    memspace = H5Screate_simple(3, count, NULL);

    H5Dread(dataset, H5T_NATIVE_FLOAT, memspace, dataspace, H5P_DEFAULT,
            f.data());
    H5Sclose(memspace);
    H5Sclose(dataspace);
    H5Dclose(dataset);
    H5Fclose(file);

    if (!rank) printf("Finished reading file %s\n", fname);
}

float residual(std::vector<float> &f, std::vector<float> &phi,
               std::vector<float> &left, std::vector<float> &right,
               hsize_t Nloc) {
    /* Calculate the residual: res = \nabla^2 phi - f */
    float res = 0.f;
    float nabla2f = 0.f;
    ssize_t i, j, k;
    for (i = 1; i < (long long)Nloc - 1; i++)
        for (j = 0; j < M; j++)
            for (k = 0; k < M; k++) {
                nabla2f =
                    phi[loc_idx(i + 1, j, k)] + phi[loc_idx(i - 1, j, k)] +
                    phi[loc_idx(i, j + 1, k)] + phi[loc_idx(i, j - 1, k)] +
                    phi[loc_idx(i, j, k + 1)] + phi[loc_idx(i, j, k - 1)] -
                    6 * phi[loc_idx(i, j, k)];
                nabla2f /= h * h;
                res += (nabla2f - f[loc_idx(i, j, k)]) *
                       (nabla2f - f[loc_idx(i, j, k)]);
            }
    /* left boundary */
    i = 0;
    for (j = 0; j < M; j++)
        for (k = 0; k < M; k++) {
            nabla2f = phi[loc_idx(i + 1, j, k)] + left[loc_idx(0, j, k)] +
                      phi[loc_idx(i, j + 1, k)] + phi[loc_idx(i, j - 1, k)] +
                      phi[loc_idx(i, j, k + 1)] + phi[loc_idx(i, j, k - 1)] -
                      6 * phi[loc_idx(i, j, k)];
            nabla2f /= h * h;
            res += (nabla2f - f[loc_idx(i, j, k)]) *
                   (nabla2f - f[loc_idx(i, j, k)]);
        }
    /* right boundary */
    i = Nloc - 1;
    for (j = 0; j < M; j++)
        for (k = 0; k < M; k++) {
            nabla2f = right[loc_idx(0, j, k)]   + phi[loc_idx(i - 1, j, k)] +
                      phi[loc_idx(i, j + 1, k)] + phi[loc_idx(i, j - 1, k)] +
                      phi[loc_idx(i, j, k + 1)] + phi[loc_idx(i, j, k - 1)] -
                      6 * phi[loc_idx(i, j, k)];
            nabla2f /= h * h;
            res += (nabla2f - f[loc_idx(i, j, k)]) *
                   (nabla2f - f[loc_idx(i, j, k)]);
        }

    MPI_Allreduce(MPI_IN_PLACE, &res, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    return sqrt(res / M / M / M);
}

void update(std::vector<float> &f, std::vector<float> &phi,
            std::vector<float> &phinew, std::vector<float> &left,
            std::vector<float> &right, hsize_t Nloc) {
    ssize_t i, j, k;
    for (i = 1; i < (long long)Nloc - 1; i++)
        for (j = 0; j < M; j++)
            for (k = 0; k < M; k++) {
                phinew[loc_idx(i, j, k)] =
                    (phi[loc_idx(i + 1, j, k)] + phi[loc_idx(i - 1, j, k)] +
                     phi[loc_idx(i, j + 1, k)] + phi[loc_idx(i, j - 1, k)] +
                     phi[loc_idx(i, j, k + 1)] + phi[loc_idx(i, j, k - 1)] -
                     h * h * f[loc_idx(i, j, k)]) /
                    6.f;
            }
    /* left boundary */
    i = 0;
    for (j = 0; j < M; j++)
        for (k = 0; k < M; k++) {
            phinew[loc_idx(i, j, k)] =
                (phi[loc_idx(i + 1, j, k)] + left[loc_idx(0, j, k)] +
                 phi[loc_idx(i, j + 1, k)] + phi[loc_idx(i, j - 1, k)] +
                 phi[loc_idx(i, j, k + 1)] + phi[loc_idx(i, j, k - 1)] -
                 h * h * f[loc_idx(i, j, k)]) /
                6.f;
        }
    /* right boundary */
    i = Nloc - 1;
    for (j = 0; j < M; j++)
        for (k = 0; k < M; k++) {
            phinew[loc_idx(i, j, k)] =
                (right[loc_idx(0, j, k)] + phi[loc_idx(i - 1, j, k)] +
                 phi[loc_idx(i, j + 1, k)] + phi[loc_idx(i, j - 1, k)] +
                 phi[loc_idx(i, j, k + 1)] + phi[loc_idx(i, j, k - 1)] -
                 h * h * f[loc_idx(i, j, k)]) /
                6.f;
        }

    std::swap(phi, phinew);
}

void exchange(std::vector<float> &phi, std::vector<float> &left,
              std::vector<float> &right, hsize_t Nloc) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int tag = 0;
    int prev = (rank == 0) ? size - 1 : rank - 1;
    int next = (rank == size - 1) ? 0 : rank + 1;
    MPI_Request req[4];
    MPI_Status stat[4];

    MPI_Isend(&phi[(Nloc - 1) * M * M], M * M, MPI_FLOAT, next, tag,
              MPI_COMM_WORLD, &req[0]);
    MPI_Isend(&phi[0], M * M, MPI_FLOAT, prev, tag, MPI_COMM_WORLD, &req[1]);
    MPI_Irecv(left.data(), M * M, MPI_FLOAT, prev, tag, MPI_COMM_WORLD,
              &req[2]);
    MPI_Irecv(right.data(), M * M, MPI_FLOAT, next, tag, MPI_COMM_WORLD,
              &req[3]);
    MPI_Waitall(4, req, stat);
}
void write_phi(std::vector<float> &phi, char *fname, hsize_t Nloc,
               hsize_t offset) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    hid_t cparms;
    cparms = H5Pcreate(H5P_DATASET_CREATE);
    cparms = H5P_DEFAULT;
    // hsize_t chunk_dims[3] = {256, 256, 256};
    // H5Pset_chunk(cparms, 3, chunk_dims);
    // H5Pset_deflate(cparms, 6);
    hid_t xp = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(xp, H5FD_MPIO_INDEPENDENT);

    hid_t fapl_par, file, dataset, dataspace, memspace;
    /* open the file */
    fapl_par = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(fapl_par, MPI_COMM_WORLD, MPI_INFO_NULL);
    file = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, fapl_par);
    H5Pclose(fapl_par);

    hid_t attr_id, attr_type, attribute;
    attr_id = H5Screate(H5S_SCALAR);
    attr_type = H5Tcopy(H5T_NATIVE_INT);
    H5Tset_size(attr_type, sizeof(int));
    attribute =
        H5Acreate2(file, "M", attr_type, attr_id, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attribute, attr_type, &M);
    H5Aclose(attribute);
    H5Sclose(attr_id);
    H5Tclose(attr_type);

    /* create the dataspace */
    const hsize_t dims[3] = {M, M, M};
    dataspace = H5Screate_simple(3, dims, NULL);
    dataset = H5Dcreate(file, "phi", H5T_NATIVE_FLOAT, dataspace, H5P_DEFAULT,
                        cparms, H5P_DEFAULT);

    /* select the hyperslab */
    dataspace = H5Dget_space(dataset);
    const hsize_t start[3] = {offset, 0, 0};
    const hsize_t count[3] = {Nloc, M, M};
    H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, start, NULL, count, NULL);

    memspace = H5Screate_simple(3, count, NULL);
    H5Dwrite(dataset, H5T_NATIVE_FLOAT, memspace, dataspace, xp, phi.data());
    H5Sclose(memspace);
    H5Sclose(dataspace);
    H5Dclose(dataset);
    H5Fclose(file);
};
