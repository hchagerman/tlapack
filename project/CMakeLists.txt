cmake_minimum_required(VERSION 3.14)
project(my_tlapack_project CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. Locate Ubuntu's OpenBLAS
find_package(OpenMP REQUIRED)
find_package(BLAS REQUIRED)

# 2. Configure wrappers to target OpenBLAS
set(blas "openblas" CACHE STRING "" FORCE)
set(lapack "openblas" CACHE STRING "" FORCE)

# Force TLAPACK's automatic package lookup OFF so CMake doesn't crash
set(TLAPACK_USE_LAPACKPP OFF CACHE BOOL "" FORCE)

set(blaspp_gpu_backend "none" CACHE STRING "" FORCE)
set(lapackpp_gpu_backend "none" CACHE STRING "" FORCE)
set(openmp OFF CACHE BOOL "" FORCE)

# 3. Suppress installation targets for fetched dependencies
set(BLASPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LAPACKPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)

# 4. Fetch the entire stack sequentially using FetchContent
include(FetchContent)

FetchContent_Declare(
    blaspp
    GIT_REPOSITORY "https://github.com/icl-utk-edu/blaspp.git"
    GIT_TAG        "master"
)
FetchContent_MakeAvailable(blaspp)

FetchContent_Declare(
    lapackpp
    GIT_REPOSITORY "https://github.com/icl-utk-edu/lapackpp.git"
    GIT_TAG        "master"
)
FetchContent_MakeAvailable(lapackpp)

FetchContent_Declare(
    tlapack
    GIT_REPOSITORY "https://github.com/tlapack/tlapack.git"
    GIT_TAG        "master"
)
FetchContent_MakeAvailable(tlapack)

# 5. Define your application
add_executable(app main.cpp)

# 6. Explicitly pass the preprocessor flag directly to your main compilation loop
target_compile_definitions(app PRIVATE TLAPACK_USE_LAPACKPP=1)

# 7. Bind dependencies 
target_link_libraries(app PRIVATE 
    tlapack 
    lapackpp 
    blaspp 
    BLAS::BLAS
)

