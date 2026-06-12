/// @file test_geqrt3.cpp
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

// Test utilities and definitions (must come before <T>LAPACK headers)
#include "testutils.hpp"

// Auxiliary routines
#include "tlapack/base/utils.hpp"
#include "tlapack/blas/gemm.hpp"
#include "tlapack/lapack/geqrt3.hpp"
#include "tlapack/lapack/lacpy.hpp"
#include "tlapack/lapack/lange.hpp"
#include "tlapack/lapack/lansy.hpp"
#include "tlapack/lapack/larf.hpp"
#include "tlapack/lapack/larfb.hpp"
#include "tlapack/lapack/larfg.hpp"
#include "tlapack/lapack/laset.hpp"
#include "tlapack/lapack/ung2r.hpp"

using namespace tlapack;

TEMPLATE_TEST_CASE("geqrt3 computes the QR factorization of a matrix",
                   "[geqrt3][qrt]",
                   TLAPACK_TYPES_TO_TEST)
{
    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    typedef real_type<T> real_t;

    // Functor
    Create<matrix_t> new_matrix;

    MatrixMarket mm;

    idx_t m, n;
    m = GENERATE(5, 7, 63, 199, 512);
    n = GENERATE(2, 3, 5, 8, 16, 21, 51, 128);

    const real_t eps = ulp<real_t>();
    const real_t tol = real_t(100 * n) * eps;

    // Matrices
    // original a matrix
    std::vector<T> A_;
    auto A = new_matrix(A_, m, n);
    // upper triangular
    std::vector<T> R_;
    auto R = new_matrix(R_, n, n);
    // household vectors
    std::vector<T> V_;
    auto V = new_matrix(V_, m, n);
    // A copy
    std::vector<T> Q_;
    auto Q = new_matrix(Q_, m, n);
    std::vector<T> T_;
    auto Tmatrix = new_matrix(T_, n, n);
    // Compute ||Qᴴ Q - I||ꜰ
    std::vector<T> work_;
    auto work = new_matrix(work_, n, n);
    // Compute ||QR - A||ᶠ / ||A||ᶠ
    std::vector<T> workQR_;
    auto workQR = new_matrix(workQR_, m, n);

    real_t norm_orth = real_t(0.0);
    real_t normA = real_t(0.0);
    real_t norm_repres = real_t(0.0);

    if (m > 0 && n > 0 && m >= n) {
        // Generate a random matrix in A & T
        mm.random(A);

        normA = tlapack::lange(tlapack::FROB_NORM, A);
        // Copy A to Q
        tlapack::lacpy(tlapack::GENERAL, A, Q);

        tlapack::geqrt3(Q, Tmatrix);

        // 2) Compute ||Qᴴ Q - I||ꜰ

        // Copy Upper Triangle of A into R
        tlapack::lacpy(tlapack::Uplo::Upper, Q, R);

        // Copy the Householder vectors into V
        tlapack::lacpy(tlapack::GENERAL, Q, V);

        // Make Q the indentity matrix
        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < m; ++i)
                Q(i, j) = static_cast<T>(0.0);

        for (idx_t j = 0; j < std::min(m, n); ++j)
            Q(j, j) = static_cast<T>(1.0);

        tlapack::larfb(tlapack::Side::Left, tlapack::Op::NoTrans,
                       tlapack::Direction::Forward, tlapack::StoreV::Columnwise,
                       V, Tmatrix, Q);
        // Compute ||Qᴴ Q - I||ꜰ

        // work receives the identity n*n
        tlapack::laset(tlapack::GENERAL, static_cast<T>(0.0),
                       static_cast<T>(1.0), work);
        // work receives Qᴴ Q - I
        tlapack::gemm(tlapack::Op::ConjTrans, tlapack::Op::NoTrans,
                      static_cast<T>(1.0), Q, Q, static_cast<T>(-1.0), work);

        norm_orth = tlapack::lange(tlapack::FROB_NORM, work);

        // 3) Compute ||QR - A||ᶠ / ||A||ᶠ

        // Copy Q to work
        tlapack::lacpy(tlapack::GENERAL, Q, workQR);

        tlapack::trmm(tlapack::Side::Right, tlapack::Uplo::Upper,
                      tlapack::Op::NoTrans, tlapack::Diag::NonUnit,
                      static_cast<T>(1.0), R, workQR);

        for (idx_t j = 0; j < n; ++j)
            for (idx_t i = 0; i < m; ++i)
                workQR(i, j) -= A(i, j);

        norm_repres = tlapack::lange(tlapack::FROB_NORM, workQR) / normA;
    }
    CHECK(norm_repres <= tol);
    CHECK(norm_orth <= tol);
}