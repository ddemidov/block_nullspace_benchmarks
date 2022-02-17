#include <vector>
#include <iostream>
#include <fstream>

#include <amgcl/backend/builtin.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/coarsening/rigid_body_modes.hpp>
#include <amgcl/relaxation/ilu0.hpp>
#include <amgcl/solver/cg.hpp>

#include <amgcl/profiler.hpp>

#include "read_problem.hpp"

namespace amgcl {
    profiler<> prof("ns_scalar");
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
    typedef amgcl::backend::builtin<double> Backend;

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
    prm.precond.coarsening.aggr.eps_strong = 0;

    // Convert the coordinates to the rigid body modes.
    // The function returns the number of near null-space vectors
    // (3 in 2D case, 6 in 3D case) and writes the vectors to the
    // std::vector<double> specified as the last argument:
    prm.precond.coarsening.nullspace.cols = amgcl::coarsening::rigid_body_modes(
            3, coo, prm.precond.coarsening.nullspace.B);

    // We use the tuple of CRS arrays to represent the system matrix.
    auto A = std::tie(rows, ptr, col, val);

    // Initialize the solver with the system matrix.
    prof.tic("setup");
    Solver solve(A, prm);
    double tm_setup = prof.toc("setup");

    // Show the mini-report on the constructed solver:
    std::cout << solve << std::endl;

    // Solve the system with the zero initial approximation:
    int iters;
    double error;
    std::vector<double> x(rows, 0.0);

    prof.tic("solve");
    std::tie(iters, error) = solve(rhs, x);
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
