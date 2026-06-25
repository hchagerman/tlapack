/// @file geqrt3.hpp
/// @author Henricus Bouwmeester, University of Colorado Denver, USA
/// @author Benicio Ayala, Metropolitan State University of Denver, USA
/// @author James Barton, Metropolitan State University of Denver, USA
/// @author Hunter Hagerman, Metropolitan State University of Denver, USA
/// @author Sandra Swartz, Metropolitan State University of Denver, USA
// Copyright (c) 2025, University of Colorado Denver. All rights reserved.
//
// This file is part of <T>LAPACK.
// <T>LAPACK is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TLAPACK_GELQT3_HH
#define TLAPACK_GELQT3_HH

#include "tlapack/blas/gemm.hpp"
#include "tlapack/blas/trmm.hpp"
#include "tlapack/lapack/lacpy.hpp"
#include "tlapack/lapack/larfg.hpp"

/**
 * Recursive QR factorization using compact WY Householder representation.
 *
 * @param[in,out] A
 *     On entry, the m-by-n matrix to be factorized, where m >= n.
 *     On exit, A contains \(R\) and the Householder vectors.
 *
 * @param[out] Tmatrix
 *     n-by-n matrix containing the triangular factor of the compact WY
 *     representation.
 *
 * @ingroup workspace_query
 */

namespace tlapack {
template <TLAPACK_MATRIX matrix_a, TLAPACK_MATRIX matrix_h>

void gelqt3(matrix_a& A, matrix_h& Tmatrix)
{
    using std::size_t;
    using idx_t = size_type<matrix_a>;
    using range = pair<idx_t, idx_t>;
    using T = type_t<matrix_a>;

    // constants
    const idx_t m = nrows(A);
    const idx_t n = ncols(A);

    auto info = 0;
    if (m > n) {
        std::cout << "Error: m < n" << std::endl;
        info = -1;
    }

    if (info != 0) {
        return;
    }

    if (m == 1) {
        // Turn the single row into a vector
        auto a_vector = row(A, 0);

        // Populate matrix T with an elementary reflector
        larfg(Direction::Forward, StoreV::Rowwise, a_vector, Tmatrix(0, 0));
    }
    else {
        // // Define slice sizes
        // idx_t m1 = m / 2;

        idx_t m1 = m / 2;
        idx_t m2 = m - m1;

        auto A1 = slice(A, range(0, m1), range(0, n));
        auto A2 = slice(A, range(m1, m), range(0, n));

        auto A11 = slice(A, range(0, m1), range(0, m1));
        auto A12 = slice(A, range(0, m1), range(m1, n));
        auto A21 = slice(A, range(m1, m), range(0, m1));
        auto A22 = slice(A, range(m1, m), range(m1, n));

        auto T11 = slice(Tmatrix, range(0, m1), range(0, m1));
        auto T22 = slice(Tmatrix, range(m1, m), range(m1, m));
        auto T12 = slice(Tmatrix, range(0, m1), range(m1, m));

        // step 1: Compute the LQ factorization of A1
        gelqt3(A1, T11);

        // step 2: Copy A21 into T21

        lacpy(Uplo::General, A21, T12);

        // step 3: T21 = A11ᴴ * T21

        trmm(Side::Right, Uplo::Upper, Op::ConjTrans, Diag::Unit,
             static_cast<T>(1.0), A11, T12);

        // step 4: T21 = T21 + (A12ᴴ * A22)

        // gemm(Op::NoTrans, Op::ConjTrans, T(1.0), A22, A12, T(1.0), T21);

        // T21 = T21 + A22 * A12^H
        gemm(Op::NoTrans, Op::ConjTrans, static_cast<T>(1.0), A22, A12,
             static_cast<T>(1.0), T12);

        // T21 = T21 * T11^H
        trmm(Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit,
             static_cast<T>(1.0), T11, T12);

        // step 6:  A22 = A22 - (A12 * T21)
        // gemm(Op::NoTrans, Op::NoTrans, T(-1.0), A12, T21, T(1.0), A22);

        // A22 = A22 - T21 * A12
        gemm(Op::NoTrans, Op::NoTrans, static_cast<T>(-1.0), T12, A12,
             static_cast<T>(1.0), A22);

        // step 7:T21 = A11 * T21
        trmm(Side::Right, Uplo::Upper, Op::NoTrans, Diag::Unit,
             static_cast<T>(1.0), A11, T12);

        // step 8: A21 = A21 - T21
        for (idx_t j = 0; j < m1; ++j) {
            for (idx_t i = 0; i < m2; ++i) {
                A21(i, j) -= T12(i, j);
            }
        }
        // step 9: Compute the LQ factorization of A22
        gelqt3(A22, T22);

        // // step 10: manually compute T21 = A12ᴴ
        // for (idx_t j = 0; j < m2; ++j) {
        //     for (idx_t i = 0; i < m1; ++i) {
        //         if constexpr (is_complex<T>)
        //             T21(i, j) = std::conj(A12(j, i));
        //         else
        //             T21(i, j) = A12(j, i);
        //     }
        // }
        lacpy(Uplo::General, A21, T12);

        // step 11: T21 = T21 * T22ᴴ
        trmm(Side::Right, Uplo::Upper, Op::ConjTrans, Diag::Unit,
             static_cast<T>(1.0), A11, T12);

        // step 12: T21 = T21 + A31ᴴ * A32
        gemm(Op::NoTrans, Op::ConjTrans, static_cast<T>(1.0), A22, A12,
             static_cast<T>(1.0), T12);

        // step 13: T21 = T21 * T11
        trmm(Side::Left, Uplo::Upper, Op::NoTrans, Diag::NonUnit,
             static_cast<T>(-1.0), T22, T12);

        // step 14: T21 = T21 * T22
        trmm(Side::Right, Uplo::Upper, Op::NoTrans, Diag::NonUnit,
             static_cast<T>(1.0), T11, T12);
    }
}
}  // namespace tlapack
#endif  // TLAPACK_GELQT3_HH