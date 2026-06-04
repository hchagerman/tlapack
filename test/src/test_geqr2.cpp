/// @file test_geqr2.cpp
/// @author
/// @brief Test GEQR2 routine
//
// Copyright?
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Test utilities and definitions (must come before <T>LAPACK headers)
#include "testutils.hpp"

// Auxiliary routines
#include "tlapack/base/utils.hpp"
#include "tlapack/lapack/geqr2.hpp"
#include "tlapack/lapack/larf.hpp"
#include "tlapack/lapack/larfg.hpp"

using namespace tlapack;

TEMPLATE_TEST_CASE("geqr2 computes the QR factorization of a matrix",
                   "[geqr2][qrt]",
                   TLAPACK_TYPES_TO_TEST)
{
    using matrix_t = TestType;
    using T = type_t<matrix_t>;
    using idx_t = size_type<matrix_t>;
    typedef real_type<T> real_t;

    // Functor
    Create<matrix_t> new_matrix;

    // MatrixMarket reader
    MatrixMarket mm;

    idx_t m, n;

    m = GENERATE(2, 6, 9);
    n = GENERATE(2, 4, 5);

    DYNAMIC_SECTION("m = " << m << " n = " << n)
    {
        const real_t eps = ulp<real_t>();
        const real_t tol = real_t(n) * eps;

        std::vector<T> A_;
        auto A = new_matrix(A_, m, n);
        std::vector<T> tau(n);

        // Generate m-by-n random matrix
        mm.random(A);

        // Compute the QR factorization of A
        geqr2(A, tau);

        // Check that Q is unitary and that A = Q*R
        for (idx_t i = 0; i < min(m, n); ++i) {
            REQUIRE(std::abs(tau[i]) <= tol);
        }
    }
}