#ifndef POISSON_H
#define POISSON_H

#include <hdf5.h>

#include <cmath>
#include <vector>

size_t loc_idx(hssize_t, hssize_t, hssize_t, const hssize_t);

/**
 * @brief Read a single component of a M * M * M field from a hdf5 file.
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

void read3D(std::vector<float> &f, char *fname, char *dsetname, hsize_t Nloc,
            hsize_t offset, const hsize_t M);

/**
 * @brief Write a single component of a M * M * M field to a hdf5 file.
 *
 * @param f
 * @param fname
 * @param Nloc
 * @param offset
 * @param M
 */
void write1D(std::vector<float> &f, char *fname, char *dsetname, hsize_t Nloc,
             hsize_t offset, const hsize_t M);

void write3D(std::vector<float> &f, char *fname, char *dsetname, hsize_t Nloc,
             hsize_t offset, const hsize_t M);

/**
 * @brief Compute the residual, the norm of the difference between the left and
 * right hand side of the poisson equation.
 *
 * @param f
 * @param phi
 * @param left
 * @param right
 * @param Nloc
 * @return float
 */
float residual(std::vector<float> &f, std::vector<float> &phi,
               std::vector<float> &left, std::vector<float> &right,
               hsize_t Nloc, const hssize_t M);

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
            std::vector<float> &right, hsize_t Nloc, const hssize_t M);

/**
 * @brief Exchange the boundaries between the mpi ranks.
 *
 * @param phi
 * @param left
 * @param right
 * @param Nloc
 */
void exchange(std::vector<float> &phi, std::vector<float> &left,
              std::vector<float> &right, hsize_t Nloc, const int);

#endif