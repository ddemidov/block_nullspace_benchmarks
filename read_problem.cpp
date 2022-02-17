#include <iostream>
#include <vector>

#include <amgcl/io/binary.hpp>
#include <amgcl/util.hpp>
#include "read_problem.hpp"

ptrdiff_t read_problem(
        std::vector<ptrdiff_t> &ptr,
        std::vector<ptrdiff_t> &col,
        std::vector<double>    &val,
        std::vector<double>    &rhs,
        std::vector<double>    &coo
        )
{
    ptrdiff_t nrows, ncoo, n, m;

    amgcl::io::read_crs("A.bin", nrows, ptr, col, val);

    amgcl::io::read_dense("b.bin", n, m, rhs);
    amgcl::precondition(n == nrows && m == 1,
            "The RHS vector has wrong size");

    amgcl::io::read_dense("C.bin", n, ncoo, coo);
    amgcl::precondition(n * ncoo == nrows && ncoo == 3,
            "Coordinate matrix has wrong size");

    std::cout << "Matrix: " << nrows << "x" << nrows << std::endl;
    std::cout << "RHS: "    << nrows << "x" << 1     << std::endl;
    std::cout << "Coords: " << n     << "x" << ncoo  << std::endl;

    return nrows;
}
