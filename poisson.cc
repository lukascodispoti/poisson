#include "poisson.h"

static float OMEGA;

void set_omega(float omega) { OMEGA = omega; }

size_t loc(hssize_t i, hssize_t j, hssize_t k, const hssize_t M) {
    i >= M ? i -= M : i;
    j >= M ? j -= M : j;
    k >= M ? k -= M : k;
    i < 0 ? i += M : i;
    j < 0 ? j += M : j;
    k < 0 ? k += M : k;
    return (i * M + j) * M + k;
}

void read1D(std::vector<float> &f, char *fname, char *dsetname, hsize_t Nloc,
            hsize_t offset, const hsize_t M) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    hid_t file, dataset, dataspace, memspace;
    /* open the file */
    hid_t fapl_par = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(fapl_par, MPI_COMM_WORLD, MPI_INFO_NULL);
    file = H5Fopen(fname, H5F_ACC_RDONLY, fapl_par);
    H5Pclose(fapl_par);

    dataset = H5Dopen2(file, dsetname, H5P_DEFAULT);
    dataspace = H5Dget_space(dataset);
    hsize_t ndims;
    ndims = H5Sget_simple_extent_ndims(dataspace);
    /* warn the user if the dataset in the file is 3D instead of 1D */
    if (ndims == 4)
        if (!rank) printf("Warning: Reading 1D field from 3d dataset!\n");

    /* mpi-local dimension of the hyperslab */
    const hsize_t count[3] = {Nloc, M, M};
    const hsize_t start[3] = {offset, 0, 0};
    H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, start, NULL, count, NULL);

    memspace = H5Screate_simple(3, count, NULL);
    H5Dread(dataset, H5T_NATIVE_FLOAT, memspace, dataspace, H5P_DEFAULT,
            f.data());
    H5Sclose(memspace);
    H5Sclose(dataspace);
    H5Dclose(dataset);
    H5Fclose(file);

    if (!rank) printf("Finished reading file %s\n", fname);
}

void read3D(std::vector<float> &f, char *fname, char *dsetname, hsize_t Nloc,
            hsize_t offset, const hsize_t M) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    hid_t file, dataset, dataspace, memspace;
    /* open the file */
    hid_t fapl_par = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(fapl_par, MPI_COMM_WORLD, MPI_INFO_NULL);
    file = H5Fopen(fname, H5F_ACC_RDONLY, fapl_par);
    H5Pclose(fapl_par);

    dataset = H5Dopen2(file, dsetname, H5P_DEFAULT);
    dataspace = H5Dget_space(dataset);

    /* mpi-local dimension of the hyperslab */
    const hsize_t count[4] = {Nloc, M, M, 3};
    const hsize_t start[4] = {offset, 0, 0, 0};
    H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, start, NULL, count, NULL);

    memspace = H5Screate_simple(4, count, NULL);
    H5Dread(dataset, H5T_NATIVE_FLOAT, memspace, dataspace, H5P_DEFAULT,
            f.data());
    H5Sclose(memspace);
    H5Sclose(dataspace);
    H5Dclose(dataset);
    H5Fclose(file);

    if (!rank) printf("Finished reading file %s\n", fname);
}

void write1D(std::vector<float> &f, char *fname, char *dsetname, hsize_t Nloc,
             hsize_t offset, const hsize_t M, int pad) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    hid_t cparms;
    cparms = H5Pcreate(H5P_DATASET_CREATE);
    cparms = H5P_DEFAULT;
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
    attr_type = H5Tcopy(H5T_NATIVE_HSIZE);
    H5Tset_size(attr_type, sizeof(hsize_t));
    attribute =
        H5Acreate2(file, "M", attr_type, attr_id, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attribute, attr_type, &M);
    H5Aclose(attribute);
    H5Sclose(attr_id);
    H5Tclose(attr_type);

    /* create the dataspace */
    const hsize_t dims[3] = {M, M, M};
    dataspace = H5Screate_simple(3, dims, NULL);
    dataset = H5Dcreate(file, dsetname, H5T_NATIVE_FLOAT, dataspace,
                        H5P_DEFAULT, cparms, H5P_DEFAULT);

    /* select the hyperslab */
    dataspace = H5Dget_space(dataset);
    const hsize_t start[3] = {offset, 0, 0};
    const hsize_t count[3] = {Nloc, M, M};
    H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, start, NULL, count, NULL);

    memspace = H5Screate_simple(3, count, NULL);
    H5Dwrite(dataset, H5T_NATIVE_FLOAT, memspace, dataspace, xp, &f[pad]);
    H5Sclose(memspace);
    H5Sclose(dataspace);
    H5Dclose(dataset);
    H5Fclose(file);

    if (!rank) printf("Finished writing file %s\n", fname);
}

void write3D(std::vector<float> &f, char *fname, char *dsetname, hsize_t Nloc,
             hsize_t offset, const hsize_t M) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    hid_t cparms;
    cparms = H5Pcreate(H5P_DATASET_CREATE);
    const hsize_t chunksize = 32;
    const hsize_t chunk[4] = {chunksize, chunksize, chunksize, 3};
    H5Pset_chunk(cparms, 4, chunk);
    hid_t xp = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(xp, H5FD_MPIO_INDEPENDENT);

    hid_t file, dataset, dataspace, memspace;
    /* open the file */
    hid_t fapl_par = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(fapl_par, MPI_COMM_WORLD, MPI_INFO_NULL);
    file = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, fapl_par);
    H5Pclose(fapl_par);

    hid_t attr_id, attr_type, attribute;
    attr_id = H5Screate(H5S_SCALAR);
    attr_type = H5Tcopy(H5T_NATIVE_HSIZE);
    H5Tset_size(attr_type, sizeof(hsize_t));
    attribute =
        H5Acreate2(file, "M", attr_type, attr_id, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attribute, attr_type, &M);
    H5Aclose(attribute);
    H5Sclose(attr_id);
    H5Tclose(attr_type);

    /* warn the user if vector to write is 1D instead of 3D */
    if (f.size() / (Nloc * M * M) != 3)
        if (!rank) printf("Warning: vector to write is not 3D!\n");

    /* create the dataspace */
    const hsize_t dims[4] = {M, M, M, 3};
    dataspace = H5Screate_simple(4, dims, NULL);
    dataset = H5Dcreate(file, dsetname, H5T_NATIVE_FLOAT, dataspace,
                        H5P_DEFAULT, cparms, H5P_DEFAULT);

    /* select the hyperslab */
    dataspace = H5Dget_space(dataset);
    const hsize_t start[4] = {offset, 0, 0, 0};
    const hsize_t count[4] = {Nloc, M, M, 3};
    H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, start, NULL, count, NULL);

    memspace = H5Screate_simple(4, count, NULL);
    H5Dwrite(dataset, H5T_NATIVE_FLOAT, memspace, dataspace, xp, f.data());
    H5Sclose(memspace);
    H5Sclose(dataspace);
    H5Dclose(dataset);
    H5Fclose(file);

    if (!rank) printf("Finished writing file %s\n", fname);
}

hsize_t get_gridsize(char *file, char *dset) {
    hsize_t dims[4];
    hid_t file_id, dset_id, dataspace;
    file_id = H5Fopen(file, H5F_ACC_RDONLY, H5P_DEFAULT);
    dset_id = H5Dopen(file_id, dset, H5P_DEFAULT);
    dataspace = H5Dget_space(dset_id);
    H5Sget_simple_extent_dims(dataspace, dims, NULL);
    H5Sclose(dataspace);
    H5Dclose(dset_id);
    H5Fclose(file_id);
    return dims[0];
}

float residual(std::vector<float> &f, std::vector<float> &phi, hsize_t Nloc,
               const hssize_t M) {
    /* Calculate the residual: res = norm(\nabla^2 phi - f) */
    float h = 2 * M_PI / M;
    float res = 0.f;
    float nabla2f = 0.f;
    ssize_t i, j, k, ii;
    for (i = 0; i < (long long)Nloc; i++) {
        ii = i + 1;
        for (j = 0; j < M; j++)
            for (k = 0; k < M; k++) {
                nabla2f =
                    phi[loc(ii + 1, j, k, M)] + phi[loc(ii - 1, j, k, M)] +
                    phi[loc(ii, j + 1, k, M)] + phi[loc(ii, j - 1, k, M)] +
                    phi[loc(ii, j, k + 1, M)] + phi[loc(ii, j, k - 1, M)] -
                    6 * phi[loc(ii, j, k, M)];
                nabla2f /= h * h;
                res += (nabla2f - f[loc(i, j, k, M)]) *
                       (nabla2f - f[loc(i, j, k, M)]);
            }
    }
    MPI_Allreduce(MPI_IN_PLACE, &res, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    return sqrt(res);
}

void Jacobi(std::vector<float> &f, std::vector<float> &phi, hsize_t Nloc,
            const hssize_t M, MPI_Request *req, MPI_Status *stat) {
    std::vector<float> phinew((Nloc + 2) * M * M, 0);
    float h = 2 * M_PI / M;
    ssize_t i, j, k, ii;
    for (i = 0; i < (long long)Nloc; i++) {
        ii = i + 1;
        for (j = 0; j < M; j++)
            for (k = 0; k < M; k++) {
                phinew[loc(ii, j, k, M)] =
                    (phi[loc(ii + 1, j, k, M)] + phi[loc(ii - 1, j, k, M)] +
                     phi[loc(ii, j + 1, k, M)] + phi[loc(ii, j - 1, k, M)] +
                     phi[loc(ii, j, k + 1, M)] + phi[loc(ii, j, k - 1, M)] -
                     h * h * f[loc(i, j, k, M)]) /
                    6.f;
            }
    }
    std::swap(phi, phinew);
    exchange(phi, Nloc, M);
}

static void sweep(ssize_t i, const hssize_t M, std::vector<float> &phi,
                  std::vector<float> &f, int mode) {
    float h = 2 * M_PI / M;
    ssize_t j, k, ii;
    ii = i + 1;
    for (j = 0; j < M; j++)
        for (k = (i + j + mode) % 2; k < M; k += 2)
            if ((i + j + k) % 2 == mode)
                phi[loc(ii, j, k, M)] =
                    (1 - OMEGA) * phi[loc(ii, j, k, M)] +
                    OMEGA *
                        (phi[loc(ii + 1, j, k, M)] + phi[loc(ii - 1, j, k, M)] +
                         phi[loc(ii, j + 1, k, M)] + phi[loc(ii, j - 1, k, M)] +
                         phi[loc(ii, j, k + 1, M)] + phi[loc(ii, j, k - 1, M)] -
                         h * h * f[loc(i, j, k, M)]) /
                        6.f;
}

void SOR(std::vector<float> &f, std::vector<float> &phi, hsize_t Nloc,
         const hssize_t M) {
    ssize_t i;
    for (i = 0; i < (ssize_t)Nloc; i++) sweep(i, M, phi, f, 0);
    exchange(phi, Nloc, M);

    for (i = 0; i < (ssize_t)Nloc; i++) sweep(i, M, phi, f, 1);
    exchange(phi, Nloc, M);
}

/**
 * @brief Call this function from the update function to overlap communication
 * and computation.
 *
 * @param phi
 * @param Nloc
 * @param M
 * @param req
 */
static void exchange(std::vector<float> &phi, hsize_t Nloc, const int M,
                     MPI_Request *req) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int tag = 0;
    int prev = (rank == 0) ? size - 1 : rank - 1;
    int next = (rank == size - 1) ? 0 : rank + 1;
    MPI_Comm comm = MPI_COMM_WORLD;

    MPI_Isend(&phi[Nloc * M * M], M * M, MPI_FLOAT, next, tag, comm, &req[0]);
    MPI_Isend(&phi[M * M], M * M, MPI_FLOAT, prev, tag, comm, &req[1]);
    MPI_Irecv(&phi[0], M * M, MPI_FLOAT, prev, tag, comm, &req[2]);
    MPI_Irecv(&phi[(Nloc + 1) * M * M], M * M, MPI_FLOAT, next, tag, comm,
              &req[3]);
}

void SOR(std::vector<float> &f, std::vector<float> &phi, hsize_t Nloc,
         const hssize_t M, MPI_Request *req, MPI_Status *stat) {
    ssize_t i;
    /* sweep boundaries first */
    sweep(0, M, phi, f, 0);
    sweep((ssize_t)Nloc - 1, M, phi, f, 0);

    /* exchange the newly computed boundaries */
    exchange(phi, Nloc, M, req);

    /* sweep through rest of phi (overlapped with communication) */
    for (i = 1; i < (ssize_t)Nloc - 1; i++) sweep(i, M, phi, f, 0);

    /* wait for communication to finish */
    MPI_Waitall(4, req, stat);

    /* repeat steps above for odd fields, don't wait */
    sweep(0, M, phi, f, 1);
    sweep((ssize_t)Nloc - 1, M, phi, f, 1);
    exchange(phi, Nloc, M, req);
    for (i = 1; i < (ssize_t)Nloc - 1; i++) sweep(i, M, phi, f, 1);
}

void exchange(std::vector<float> &phi, hsize_t Nloc, const int M) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int tag = 0;
    int prev = (rank == 0) ? size - 1 : rank - 1;
    int next = (rank == size - 1) ? 0 : rank + 1;
    MPI_Request req[4];
    MPI_Status stat[4];
    MPI_Comm comm = MPI_COMM_WORLD;

    MPI_Isend(&phi[Nloc * M * M], M * M, MPI_FLOAT, next, tag, comm, &req[0]);
    MPI_Isend(&phi[M * M], M * M, MPI_FLOAT, prev, tag, comm, &req[1]);
    MPI_Irecv(&phi[0], M * M, MPI_FLOAT, prev, tag, comm, &req[2]);
    MPI_Irecv(&phi[(Nloc + 1) * M * M], M * M, MPI_FLOAT, next, tag, comm,
              &req[3]);
    MPI_Waitall(4, req, stat);
}

bool endswith(std::string const &value, std::string const &ending) {
    if (ending.size() > value.size()) return false;
    return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}