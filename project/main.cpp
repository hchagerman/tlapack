// Plugins for <T>LAPACK (must come before <T>LAPACK headers)
#include <tlapack/plugins/legacyArray.hpp>

// <T>LAPACK headers
#include <tlapack/base/utils.hpp>
#include <tlapack/blas/gemm.hpp>

// C++ headers
#include <chrono>  // for high_resolution_clock


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
    std::cout << std::endl;
}

//------------------------------------------------------------------------------
    template <typename T>
void run(size_t m, size_t n, size_t k)
{
    using std::size_t;
    using matrix_t = tlapack::LegacyMatrix<T>;
    using idx_t = tlapack::size_type<matrix_t>;
    using range = tlapack::pair<idx_t, idx_t>;
    using tlapack::conj;

    // Functors for creating new matrices
    tlapack::Create<matrix_t> new_matrix;

    // Turn it off if m or n are large
    bool verbose = false;

    // Matrices
    std::vector<T> A_(m * k, 0.0);
    auto A = new_matrix(A_, m, k);
    std::vector<T> B_(k * n, 0.0);
    auto B = new_matrix(B_, k, n);
    std::vector<T> C_(m * n, 0.0);
    auto C = new_matrix(C_, m, n);


    // Initialize arrays with junk
    for (idx_t j = 0; j < k; ++j) {
        for (idx_t i = 0; i < m; ++i)
            A(i, j) = T(static_cast<float>(0xDEADBEEF));
        for (idx_t i = 0; i < n; ++i)
            B(j, i) = T(static_cast<float>(0xDEADBEEF));
    }
    for (idx_t j = 0; j < n; ++j)
        for (idx_t i = 0; i < m; ++i)
            C(i, j) = T(static_cast<float>(0xDEADBEEF));


    // Generate a random matrix in A and B
    for (idx_t j = 0; j < k; ++j) {
        for (idx_t i = 0; i < m; ++i)
            A(i, j) = T(static_cast<float>(rand()) /
                    static_cast<float>(RAND_MAX));
        for (idx_t i = 0; i < n; ++i)
            B(j, i) = T(static_cast<float>(rand()) /
                    static_cast<float>(RAND_MAX));
    }

    // Record start time
    auto startGEMM = std::chrono::high_resolution_clock::now();
    {
        tlapack::gemm(tlapack::Op::NoTrans, tlapack::Op::NoTrans, T(1.0), A, B, (1.0), C);
    }   
    // Record end time
    auto endGEMM = std::chrono::high_resolution_clock::now();

    // Compute elapsed time in nanoseconds
    auto elapsedGEMM =
        std::chrono::duration_cast<std::chrono::nanoseconds>(endGEMM - startGEMM);

    // Compute FLOPS
    double flopsGEMM =
        (2.0e+00 * ((double)m) * ((double)n) * ((double)k)) / (elapsedGEMM.count() * 1.0e-9);

    // *) Output

    std::cout << std::endl;
    std::cout << "time = " << elapsedGEMM.count() * 1.0e-6 << " ms"
              << ",   GFlop/sec = " << flopsGEMM * 1.0e-9;
    std::cout << std::endl;
}

//------------------------------------------------------------------------------
int main(int argc, char** argv) {

    int m, n, k;

    // Default arguments
    m = (argc < 2) ? 10 : atoi(argv[1]);
    n = (argc < 3) ? 5 : atoi(argv[2]);
    k = (argc < 4) ? 3 : atoi(argv[3]);

    srand(3);  // Init random seed

    std::cout.precision(5);
    std::cout << std::scientific << std::showpos;

    printf("run< float  >( %d, %d, %d )", m, n, k);
    run<float>(m, n, k);
    printf("-----------------------\n");

    printf("run< double >( %d, %d, %d )", m, n, k);
    run<double>(m, n, k);
    printf("-----------------------\n");

    return 0;
}
