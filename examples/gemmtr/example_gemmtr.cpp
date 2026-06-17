/// @file example_gemmtr.cpp
/// @author Henricus Bouwmeester, University of Colorado Denver, USA
/// @author Benicio Ayala, Metropolitan State University of Denver, USA
/// @author James Barton, Metropolitan State University of Denver, USA
/// @author Hunter Hagerman, Metropolitan State University of Denver, USA
/// @author Sandra Swartz, Metropolitan State University of Denver, USA
//
// Copyright (c) 2026, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.
//
// DGEQRT3 recursively computes a QR factorization of a real M-by-N
// matrix A, using the compact WY representation of Q.

// Plugins for <T>LAPACK (must come before <T>LAPACK headers)
#include <tlapack/plugins/legacyArray.hpp>

// <T>LAPACK
#include <tlapack/blas/gemm.hpp>
#include <tlapack/blas/trmm.hpp>
#include <tlapack/lapack/gemmtr.hpp>
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/lange.hpp>
#include <tlapack/lapack/larfb.hpp>
#include <tlapack/lapack/laset.hpp>

// C++ headers
#include <chrono>  // for high_resolution_clock
#include <iostream>
#include <memory>
#include <vector>

using namespace tlapack;

//------------------------------------------------------------------------------
/// Print matrix A in the standard output
template <typename matrix_t>
void printMatrix(const matrix_t& A)
{
    using idx_t = tlapack::size_type<matrix_t>;
    const idx_t m = tlapack::nrows(A);
    const idx_t n = tlapack::ncols(A);

    for (idx_t i = 0; i < m; ++i) {
        std::cout << std::endl;
        for (idx_t j = 0; j < n; ++j)
            std::cout << A(i, j) << " ";
    }
}

//------------------------------------------------------------------------------
template <typename T>
void run(size_t m, size_t n, size_t k)
{
    using std::size_t;
    using matrix_t = tlapack::LegacyMatrix<T>;
    using idx_t = tlapack::size_type<matrix_t>;
    using range = tlapack::pair<idx_t, idx_t>;

    // Functors for creating new matrices
    tlapack::Create<matrix_t> new_matrix;

    // Turn it off if m or n are large
    bool verbose = false;

    // Matrices
    // triangular matrix
    std::vector<T> A_;
    auto A = new_matrix(A_, m, k);
    std::vector<T> B_;
    auto B = new_matrix(B_, k, n);
    std::vector<T> C_;
    auto C = new_matrix(C_, m, n);
    std::vector<T> C_copy_;
    auto C_copy = new_matrix(C_copy_, m, n);

    // Fill A
    for (idx_t j = 0; j < k; ++j) {
        for (idx_t i = 0; i < m; ++i) {
            A(i, j) = T(static_cast<float>(0xDEADBEEF));
        }
    }

    // Fill B
    for (idx_t j = 0; j < n; ++j) {
        for (idx_t i = 0; i < k; ++i) {
            B(i, j) = T(static_cast<float>(0xDEADBEEF));
        }
    }

    // Fill C
    for (idx_t j = 0; j < n; ++j) {
        for (idx_t i = 0; i < m; ++i) {
            C(i, j) = T(static_cast<float>(0xDEADBEEF));
        }
    }
    // Make A upper triangular
    for (idx_t j = 0; j < k; ++j) {
        for (idx_t i = 0; i < m; ++i) {
            if (i > j) A(i, j) = T(0);
        }
    }
    lacpy(Uplo::General, C, C_copy);
    // c gets overloaded
    gemm(Op::NoTrans, Op::NoTrans, static_cast<T>(1.0), A, B,
         static_cast<T>(1.0), C);

    std::cout << std::endl << "gemm matrix: " << std::endl;
    printMatrix(C);
    std::cout << std::endl;

    gemmtr(Side::Left, Uplo::Upper, Op::NoTrans, Diag::NonUnit, Op::NoTrans,
           static_cast<T>(1.0), A, B, static_cast<T>(1.0), C_copy);
    std::cout << std::endl << "gemmtr matrix: " << std::endl;
    printMatrix(C_copy);
    std::cout << std::endl;

    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = 0; i < m; ++i)
            C_copy(i, j) -= C(i, j);

    auto error = lange(tlapack::FROB_NORM, C_copy);

    std::cout << "||gemm - gemmtr||_F = " << std::real(error) << std::endl;

    lacpy(Uplo::General, C_copy, C);
    // Reset C only
    for (idx_t j = 0; j < n; ++j) {
        for (idx_t i = 0; i < m; ++i) {
            C(i, j) = T(static_cast<float>(0xDEADBEEF));
        }
    }

    trmm(Side::Left, Uplo::Upper, Op::NoTrans, Diag::NonUnit,
         static_cast<T>(1.0), A, B);

    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = 0; i < m; ++i)
            C(i, j) += B(i, j);

    std::cout << std::endl << "trmm matrix: " << std::endl;
    printMatrix(C);
    std::cout << std::endl;

    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = 0; i < m; ++i)
            C_copy(i, j) -= C(i, j);

    error = lange(tlapack::FROB_NORM, C_copy);

    std::cout << "||trmm - gemmtr||_F = " << std::real(error) << std::endl;
}
//================================================================================
//================================================================================
int main(int argc, char** argv)
{
    int m, n, k;

    // Default arguments
    m = (argc < 2) ? 1 : atoi(argv[1]);
    n = (argc < 3) ? 1 : atoi(argv[2]);

    srand(3);  // Init random seed

    m = 4;
    n = 5;
    k = 4;

    std::cout.precision(5);
    std::cout << std::scientific << std::showpos;

    printf("run< float  >( %d, %d )", m, n);
    run<float>(m, n, k);
    printf("-----------------------\n");

    // printf("run< double >( %d, %d )", m, n);
    // run<double>(m, n);
    // printf("-----------------------\n");

    // printf("run< long double >( %d, %d )", m, n);
    // run<long double>(m, n);
    // printf("-----------------------\n");

    // printf("run< complex<float> >( %d, %d )", m, n);
    // run<std::complex<float>>(m, n);
    // printf("-----------------------\n");

    // printf("run< complex<double> >( %d, %d )", m, n);
    // run<std::complex<double>>(m, n);
    // printf("-----------------------\n");

    // printf("run< complex<long double> >( %d, %d )", m, n);
    // run<std::complex<long double>>(m, n);
    // printf("-----------------------\n");

    return 0;
}
