//// @file example_rgeqrf.cpp
/// @author Weslley S Pereira, University of Colorado Denver, USA
//
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Plugins for <T>LAPACK (must come before <T>LAPACK headers)
#include <tlapack/plugins/legacyArray.hpp>

// <T>LAPACK
#include <tlapack/blas/gemm.hpp>
#include <tlapack/blas/syrk.hpp>
#include <tlapack/blas/trmm.hpp>
#include <tlapack/lapack/geqr2.hpp>
#include <tlapack/lapack/geqrt3.hpp>
#include <tlapack/lapack/lacpy.hpp>
#include <tlapack/lapack/lange.hpp>
#include <tlapack/lapack/lansy.hpp>
#include <tlapack/lapack/laset.hpp>
#include <tlapack/lapack/ung2r.hpp>

// C++ headers
#include <chrono>  // for high_resolution_clock
#include <iostream>
#include <memory>
#include <vector>

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
void run(size_t m, size_t n, size_t nb)
{
    using std::size_t;
    using matrix_t = tlapack::LegacyMatrix<T>;
    using idx_t = tlapack::size_type<matrix_t>;
    using range = tlapack::pair<idx_t, idx_t>;

    // Functors for creating new matrices
    tlapack::Create<matrix_t> new_matrix;

    // Turn it off if m or n are large
    bool verbose = true;

    // Arrays
    std::vector<T> tau(n);

    // Matrices
    std::vector<T> A_;
    auto A = new_matrix(A_, m, n);
    std::vector<T> R_;
    auto R = new_matrix(R_, n, n);
    std::vector<T> Q_;
    auto Q = new_matrix(Q_, m, n);
    std::vector<T> Tmatrix_;
    auto Tmatrix = new_matrix(Tmatrix_, m, n);

    // Initialize arrays with junk
    for (idx_t j = 0; j < n; ++j) {
        for (idx_t i = 0; i < m; ++i) {
            A(i, j) = T(static_cast<float>(0xDEADBEEF));
            Q(i, j) = T(static_cast<float>(0xCAFED00D));
        }
        for (idx_t i = 0; i < n; ++i) {
            Tmatrix(i, j) = T(static_cast<float>(0XFEE1DEAD));
            R(i, j) = T(static_cast<float>(0xFEE1DEAD));
        }
        tau[j] = T(static_cast<float>(0xFFBADD11));
    }

    // Generate a random matrix in A
    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = 0; i < m; ++i)
            if constexpr (tlapack::is_complex<T>)
                A(i, j) = T(
                    static_cast<float>(rand()) / static_cast<float>(RAND_MAX),
                    static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
            else
                A(i, j) = T(static_cast<float>(rand()) /
                            static_cast<float>(RAND_MAX));

    // Frobenius norm of A
    auto normA = tlapack::lange(tlapack::FROB_NORM, A);

    // Print A
    if (verbose) {
        std::cout << std::endl << "A = ";
        printMatrix(A);
    }

    // Copy A to Q
    tlapack::lacpy(tlapack::GENERAL, A, Q);

    // 1) Compute A = QR (Stored in the matrix Q)

    // Record start time
    auto startQR = std::chrono::high_resolution_clock::now();
    {
        for (idx_t k = 0; k < n; k += nb) {
            std::cout << "k = " << k << " k + nb = " << k + nb << " n = " << n
                      << std::endl;
            std::cout << "loop!" << std::endl;
            std::cout << std::endl;

            if (k + nb >= n) {
                nb = n - 1 - k;
                std::cout << "Change! k + nb = " << k + nb << std::endl;
            }

            // Slice Q and T
            auto A1 = tlapack::slice(Q, range(k, m), range(k, k + nb));
            auto A11 = tlapack::slice(Q, range(k, k + nb), range(k, k + nb));
            auto A21 = tlapack::slice(Q, range(k + nb, m), range(k, k + nb));
            auto A12 = tlapack::slice(Q, range(k, k + nb), range(k + nb, n));
            auto A22 = tlapack::slice(Q, range(k + nb, m), range(k + nb, n));
            auto T11 =
                tlapack::slice(Tmatrix, range(k, k + nb), range(k, k + nb));
            auto T12 =
                tlapack::slice(Tmatrix, range(k, k + nb), range(k + nb, n));

            // QR factorization
            tlapack::geqrt3<T>(A1, T11);

            tlapack::lacpy(tlapack::Uplo::General, A12, T12);

            tlapack::trmm(tlapack::Side::Left, tlapack::Uplo::Lower,
                          tlapack::Op::ConjTrans, tlapack::Diag::Unit,
                          static_cast<T>(1.0), A11, T12);

            tlapack::gemm(tlapack::Op::ConjTrans, tlapack::Op::NoTrans,
                          static_cast<T>(1.0), A21, A22, static_cast<T>(1.0),
                          T12);

            tlapack::trmm(tlapack::Side::Left, tlapack::Uplo::Upper,
                          tlapack::Op::ConjTrans, tlapack::Diag::NonUnit,
                          static_cast<T>(1.0), T11, T12);

            tlapack::trmm(tlapack::Side::Left, tlapack::Uplo::Lower,
                          tlapack::Op::NoTrans, tlapack::Diag::Unit,
                          static_cast<T>(1.0), A11, T12);

            for (idx_t i = 0; i < nb; ++i) {
                for (idx_t j = 0; j < n - k - nb; ++j) {
                    A12(i, j) -= T12(i, j);
                }
            }

            tlapack::gemm(tlapack::Op::NoTrans, tlapack::Op::NoTrans,
                          static_cast<T>(-1.0), A21, T12, static_cast<T>(1.0),
                          A22);
        }
        // After this point is creating final products of Q and R
        //
        //
        //
        //
        //
        //
        //
        //
        //
        // Save the R matrix
        tlapack::lacpy(tlapack::UPPER_TRIANGLE, Q, R);

        // Generates Q = H_1 H_2 ... H_n
        tlapack::ung2r(Q, tau);
    }
    // Record end time
    auto endQR = std::chrono::high_resolution_clock::now();

    // Compute elapsed time in nanoseconds
    auto elapsedQR =
        std::chrono::duration_cast<std::chrono::nanoseconds>(endQR - startQR);

    // Compute FLOPS
    double flopsQR =
        (4.0e+00 * ((double)m) * ((double)n) * ((double)n) -
         4.0e+00 / 3.0e+00 * ((double)n) * ((double)n) * ((double)n)) /
        (elapsedQR.count() * 1.0e-9);

    // Print Q and R
    if (verbose) {
        std::cout << std::endl << "Q = ";
        printMatrix(Q);
        std::cout << std::endl << "R = ";
        printMatrix(R);
    }

    T norm_orth_1, norm_repres_1;

    // 2) Compute ||Q'Q - I||_F

    {
        std::vector<T> work_;
        auto work = new_matrix(work_, n, n);
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < n; ++i)
                work(i, j) = static_cast<float>(0xABADBABE);

        // work receives the identity n*n
        tlapack::laset(tlapack::UPPER_TRIANGLE, 0.0, 1.0, work);
        // work receives Q'Q - I
        tlapack::syrk(tlapack::Uplo::Upper, tlapack::Op::Trans, 1.0, Q, -1.0,
                      work);

        // Compute ||Q'Q - I||_F
        norm_orth_1 =
            tlapack::lansy(tlapack::FROB_NORM, tlapack::UPPER_TRIANGLE, work);

        if (verbose) {
            std::cout << std::endl << "Q'Q-I = ";
            printMatrix(work);
        }
    }

    // 3) Compute ||QR - A||_F / ||A||_F

    {
        std::vector<T> work_;
        auto work = new_matrix(work_, m, n);
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < m; ++i)
                work(i, j) = static_cast<float>(0xABADBABE);

        // Copy Q to work
        tlapack::lacpy(tlapack::GENERAL, Q, work);

        tlapack::trmm(tlapack::Side::Right, tlapack::Uplo::Upper,
                      tlapack::Op::NoTrans, tlapack::Diag::NonUnit, 1.0, R,
                      work);

        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < m; ++i)
                work(i, j) -= A(i, j);

        norm_repres_1 = tlapack::lange(tlapack::FROB_NORM, work) / normA;
    }

    // *) Output

    std::cout << std::endl;
    std::cout << "time = " << elapsedQR.count() * 1.0e-6 << " ms"
              << ",   GFlop/sec = " << flopsQR * 1.0e-9;
    std::cout << std::endl;
    std::cout << "||QR - A||_F/||A||_F  = " << norm_repres_1
              << ",        ||Q'Q - I||_F  = " << norm_orth_1;
    std::cout << std::endl;
}

//------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    int m, n, nb;

    // Default arguments
    m = (argc < 2) ? 7 : atoi(argv[1]);
    n = (argc < 3) ? 5 : atoi(argv[2]);

    m = 4;
    n = 4;
    nb = 2;

    srand(3);  // Init random seed

    std::cout.precision(5);
    std::cout << std::scientific << std::showpos;

    printf("run< float  >( %d, %d )", m, n);
    run<float>(m, n, nb);
    printf("-----------------------\n");

    printf("run< double >( %d, %d )", m, n);
    run<double>(m, n, nb);
    printf("-----------------------\n");

    printf("run< long double >( %d, %d )", m, n);
    run<long double>(m, n, nb);
    printf("-----------------------\n");

    return 0;
}