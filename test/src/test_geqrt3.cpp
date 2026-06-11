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

work(i, j) = static_cast<float>(0xABADBABE);

// work receives the identity n*n
tlapack::laset(tlapack::UPPER_TRIANGLE,
               static_cast<T>(0.0),
               static_cast<T>(1.0),
               work);
// work receives QᴴQ - I
tlapack::gemm(tlapack::Op::ConjTrans,
              tlapack::Op::NoTrans,
              static_cast<T>(1.0),
              A,
              A,
              static_cast<T>(-1.0),
              work);

// Compute ||QᴴQ - I||ᶠ
norm_orth = tlapack::lansy(tlapack::FROB_NORM, tlapack::UPPER_TRIANGLE, work);

// 3) Compute ||QR - A||ᶠ / ||A||ᶠ

for (idx_t j = 0; j < n; ++j)
    for (idx_t i = 0; i < m; ++i)
        work(i, j) = static_cast<float>(0);
// Copy Q to work
tlapack::lacpy(tlapack::GENERAL, A, work);

tlapack::trmm(tlapack::Side::Right,
              tlapack::Uplo::Upper,
              tlapack::Op::NoTrans,
              tlapack::Diag::NonUnit,
              static_cast<T>(1.0),
              A,
              work);

for (idx_t j = 0; j < n; ++j)
    for (idx_t i = 0; i < m; ++i)
        work(i, j) -= A(i, j);

normA = tlapack::lange(tlapack::FROB_NORM, work) / normA;
}

CHECK(normA <= tol);
CHECK(norm_orth <= tol);
}
}

using matrix_t = TestType;
using T = type_t<matrix_t>;
using idx_t = size_type<matrix_t>;
typedef real_type<T> real_t;

// Functor
Create<matrix_t> new_matrix;

// Matrices
std::vector<T> A_;
auto A = new_matrix(A_, m, n);
std::vector<T> R_;
auto R = new_matrix(R_, n, n);
std::vector<T> V_;
auto V = new_matrix(V_, m, n);
std::vector<T> Q_;
auto Q = new_matrix(Q_, m, n);
std::vector<T> T_;
auto Tmatrix = new_matrix(T_, n, n);

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
}
// Generate a random matrix in A & T
mm.random(A);
mm.random(T);
mm.random(Q);
mm.random(R);
mm.random(Tmatrix);

// Copy A to Q
tlapack::lacpy(tlapack::GENERAL, A, Q);

tlapack::geqrt3(Q, Tmatrix);

T norm_orth, norm_repres;

// 2) Compute ||Qᴴ Q - I||ꜰ

{
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
                   tlapack::Direction::Forward, tlapack::StoreV::Columnwise, V,
                   Tmatrix, Q);

    std::vector<T> work_;
    auto work = new_matrix(work_, n, n);
    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = 0; i < n; ++i)
            work(i, j) = static_cast<float>(0xABADBABE);
    ;

    // work receives the identity n*n
    tlapack::laset(tlapack::GENERAL, static_cast<T>(0.0), static_cast<T>(1.0),
                   work);
    // work receives Qᴴ Q - I
    tlapack::gemm(tlapack::Op::ConjTrans, tlapack::Op::NoTrans,
                  static_cast<T>(1.0), Q, Q, static_cast<T>(-1.0), work);

    // Compute ||Qᴴ Q - I||ꜰ
    norm_orth = tlapack::lange(tlapack::FROB_NORM, work);

    // 3) Compute ||QR - A||ᶠ / ||A||ᶠ
    std::vector<T> work_;

    auto work = new_matrix(work_, m, n);
    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = 0; i < m; ++i)
            work(i, j) = static_cast<float>(0);
    // Copy Q to work
    tlapack::lacpy(tlapack::GENERAL, Q, work);

    tlapack::trmm(tlapack::Side::Right, tlapack::Uplo::Upper,
                  tlapack::Op::NoTrans, tlapack::Diag::NonUnit,
                  static_cast<T>(1.0), R, work);

    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = 0; i < m; ++i)
            work(i, j) -= A(i, j);

    norm_repres = tlapack::lange(tlapack::FROB_NORM, work) / normA;
}

CHECK(normA <= tol);
CHECK(norm_orth <= tol);
}