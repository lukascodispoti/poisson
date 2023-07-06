#ifndef POISSON_H
#define POISSON_H

#include <hdf5.h>

#include <cmath>
#include <string>
#include <vector>

void set_omega(float omega);

/**
 * @brief Return the serialized index of a point in a M * M * M grid.
 *
 * @return size_t
 */
size_t loc(hssize_t, hssize_t, hssize_t, const hssize_t);

/**
 * @brief Read a single component of a M * M * M field from an hdf5 file.
 *
 * @param f
 * @param fname
 * @param dsetname
 * @param Nloc
 * @param offset
 * @param M
 */
void read1D(std::vector<float> &f, char *fname, char *dsetname, hsize_t Nloc,
            hsize_t offset, const hsize_t M);

/**
 * @brief Read three components of a M * M * M field from an hdf5 file.
 *
 * @param f
 * @param fname
 * @param dsetname
 * @param Nloc
 * @param offset
 * @param M
 */
void read3D(std::vector<float> &f, char *fname, char *dsetname, hsize_t Nloc,
            hsize_t offset, const hsize_t M);
/**
 * @brief Write a single component of a M * M * M field to an hdf5 file.
 *
 * @param f
 * @param fname
 * @param Nloc
 * @param offset
 * @param M
 * @param pad
 */
void write1D(std::vector<float> &f, char *fname, char *dsetname, hsize_t Nloc,
             hsize_t offset, const hsize_t M, int pad  = 0);

/**
 * @brief Write three components of a M * M * M field to an hdf5 file.
 *
 * @param f
 * @param fname
 * @param dsetname
 * @param Nloc
 * @param offset
 * @param M
 */
void write3D(std::vector<float> &f, char *fname, char *dsetname, hsize_t Nloc,
             hsize_t offset, const hsize_t M);

/**
 * @brief Return the extent of the first dimension of the dataset `dset` in the
 * hdf5 file `file`.
 *
 * @param file
 * @param dset
 * @return hsize_t
 */
hsize_t get_gridsize(char *file, char *dset);

/**
 * @brief Compute the residual, the norm of the difference between the left and
 * right hand side of the poisson equation.
 *
 * @param f
 * @param phi
 * @param Nloc
 * @return float
 */
float residual(std::vector<float> &f, std::vector<float> &phi, hsize_t Nloc,
               const hssize_t M);

/**
 * @brief Update the solution of the poisson equation
 *
 *  d 2 φ
 *  ----- = f
 *  dx 2
 *
 * using the centered difference scheme and the Jacobi method.
 *
 * @param f
 * @param phi
 * @param phinew
 * @param Nloc
 */
void Jacobi(std::vector<float> &f, std::vector<float> &phi, hsize_t Nloc,
            const hssize_t M);

/**
 * @brief Use Gauss Seidel to update the solution of the poisson equation.
 *
 * @param f
 * @param phi
 * @param Nloc
 * @param M
 */
void GaussSeidel(std::vector<float> &f, std::vector<float> &phi, hsize_t Nloc,
                 const hssize_t M);

/**
 * @brief Use SOR to update the solution of the poisson equation.
 *
 * @param f
 * @param phi
 * @param Nloc
 * @param M
 */
void SOR(std::vector<float> &f, std::vector<float> &phi, hsize_t Nloc,
         const hssize_t M);

/**
 * @brief Exchange the ghost cells between the mpi ranks.
 *
 * @param phi
 * @param Nloc
 * @param M
 */
void exchange(std::vector<float> &phi, hsize_t Nloc, const int M);

/**
 * @brief Returns true if the string `value` ends with the string `ending`,
 * otherwise return false.
 *
 * @param value
 * @param ending
 * @return true
 * @return false
 */
bool endswith(std::string const &value, std::string const &ending);

#endif