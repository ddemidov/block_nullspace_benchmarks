#include <vector>
#include <iostream>
#include <fstream>

#include <amgcl/backend/builtin.hpp>
#include <amgcl/value_type/static_matrix.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/adapter/block_matrix.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/ilu0.hpp>
#include <amgcl/solver/cg.hpp>

#include <amgcl/profiler.hpp>

#include "read_problem.hpp"

namespace amgcl {
    profiler<> prof("block");
}
using amgcl::prof;

int main(int, char *argv[]) {
    // Read the system matrix, the RHS, and the coordinates:
    ptrdiff_t rows;
    std::vector<ptrdiff_t> ptr, col;
    std::vector<double> val, rhs, coo;

    prof.tic("read");
    rows = read_problem(ptr, col, val, rhs, coo);
    prof.toc("read");

    // Declare the solver type
    typedef amgcl::static_matrix<double, 3, 3> Block;
    typedef amgcl::backend::builtin<Block> Backend;

    typedef amgcl::make_solver<
        amgcl::amg<
            Backend,
            amgcl::coarsening::smoothed_aggregation,
            amgcl::relaxation::ilu0
            >,
        amgcl::solver::cg<Backend>
        > Solver;

    // Solver parameters:
    Solver::params prm;
    prm.solver.maxiter = 1000;

    // We use the tuple of CRS arrays to represent the system matrix.
    auto A = std::tie(rows, ptr, col, val);
    amgcl::backend::crs<Block> Ab(amgcl::adapter::block_matrix<Block>(A));

    // Initialize the solver with the system matrix.
    prof.tic("setup");
    Solver solve(Ab, prm);
    double tm_setup = prof.toc("setup");

    // Show the mini-report on the constructed solver:
    std::cout << solve << std::endl;

    // Solve the system with the zero initial approximation:
    int iters;
    double error;
    std::vector<double> x(rows, 0.0);
    auto F = amgcl::backend::reinterpret_as_rhs<Block>(rhs);
    auto X = amgcl::backend::reinterpret_as_rhs<Block>(x);

    prof.tic("solve");
    std::tie(iters, error) = solve(F, X);
    double tm_solve = prof.toc("solve");

    // Output the number of iterations, the relative error,
    // and the profiling data:
    std::cout << "Iters: " << iters << std::endl
              << "Error: " << error << std::endl
              << prof << std::endl;

    std::ofstream log(argv[0] + std::string(".log"));
    log << tm_setup << "\t"
        << tm_solve << "\t"
        << tm_setup + tm_solve << "\t"
        << iters << "\t"
        << solve.precond().bytes() / (1024. * 1024) << std::endl;
}
