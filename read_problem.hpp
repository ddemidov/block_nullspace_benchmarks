#ifndef CODE_READ_HPP
#define CODE_READ_HPP

#include <vector>

ptrdiff_t read_problem(
        std::vector<ptrdiff_t> &ptr,
        std::vector<ptrdiff_t> &col,
        std::vector<double>    &val,
        std::vector<double>    &rhs,
        std::vector<double>    &coo
        );

#endif
