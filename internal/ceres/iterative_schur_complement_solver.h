// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2015 Google Inc. All rights reserved.
// http://ceres-solver.org/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of Google Inc. nor the names of its contributors may be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: sameeragarwal@google.com (Sameer Agarwal)

#ifndef CERES_INTERNAL_ITERATIVE_SCHUR_COMPLEMENT_SOLVER_H_
#define CERES_INTERNAL_ITERATIVE_SCHUR_COMPLEMENT_SOLVER_H_

#include <memory>

#include "ceres/conjugate_gradients_solver.h"
#include "ceres/internal/disable_warnings.h"
#include "ceres/internal/eigen.h"
#include "ceres/internal/export.h"
#include "ceres/linear_solver.h"
#include "ceres/types.h"

namespace ceres::internal {

class BlockSparseMatrix;
class ImplicitSchurComplementBase;
class Preconditioner;

// This class implements an iterative solver for the linear least
// squares problems that have a bi-partite sparsity structure common
// to Structure from Motion problems.
//
// The algorithm used by this solver was developed in a series of
// papers - "Agarwal et al, Bundle Adjustment in the Large, ECCV 2010"
// and "Wu et al, Multicore Bundle Adjustment, submitted to CVPR
// 2011" at the University of Washington.
//
// The key idea is that one can run Conjugate Gradients on the Schur
// Complement system without explicitly forming the Schur Complement
// in memory. The heavy lifting for this is done by the
// ImplicitSchurComplement class. Not forming the Schur complement in
// memory and factoring it results in substantial savings in time and
// memory. Further, iterative solvers like this open up the
// possibility of solving the Newton equations in a non-linear solver
// only approximately and terminating early, thereby saving even more
// time.
//
// For the curious, running CG on the Schur complement is the same as
// running CG on the Normal Equations with an SSOR preconditioner. For
// a proof of this fact and others related to this solver please see
// the section on Domain Decomposition Methods in Saad's book
// "Iterative Methods for Sparse Linear Systems".
//
// Implementations of IterativeSchurComplementSolverBase interface are expected
// to provide methods for creating preconditioner and pre-solver that are
// compatible in terms of location of vector values (cpu / gpu)
class CERES_NO_EXPORT IterativeSchurComplementSolverBase
    : public BlockSparseMatrixSolver {
 public:
  explicit IterativeSchurComplementSolverBase(LinearSolver::Options options);
  IterativeSchurComplementSolverBase(
      const IterativeSchurComplementSolverBase&) = delete;
  void operator=(const IterativeSchurComplementSolverBase&) = delete;

  virtual ~IterativeSchurComplementSolverBase();

  static std::unique_ptr<IterativeSchurComplementSolverBase> Create(
      const LinearSolver::Options& options);

 protected:
  std::unique_ptr<ImplicitSchurComplementBase> schur_complement_;
  std::unique_ptr<Preconditioner> preconditioner_;
  std::unique_ptr<Preconditioner> pre_solver_;
  LinearSolver::Options options_;

 private:
  virtual void Initialize() = 0;
  // Both vectors b and x are in cpu memory
  LinearSolver::Summary SolveImpl(BlockSparseMatrix* A,
                                  const double* b,
                                  const LinearSolver::PerSolveOptions& options,
                                  double* x) final;
  // Create power-series preconditioner
  virtual void CreatePreSolver(const int max_num_spse_iterations,
                               const double spse_tolerance) = 0;
  // Back-substitution (x pointer is always in cpu memory)
  virtual void BackSubstitute(const double* reduced_system_solution,
                              double* x) = 0;
  // Run conjugate gradients solver
  virtual LinearSolver::Summary ReducedSolve(
      const ConjugateGradientsSolverOptions& cg_options) = 0;

  // Pointer to solution of reduced linear system (might be in cpu or gpu memory
  // depending on implementation)
  virtual double* reduced_linear_system_solution() = 0;

  virtual void CreatePreconditioner(const BlockSparseMatrix* A) = 0;
};

class CERES_NO_EXPORT IterativeSchurComplementSolver
    : public IterativeSchurComplementSolverBase {
 public:
  explicit IterativeSchurComplementSolver(LinearSolver::Options options);
  double* reduced_linear_system_solution();
  void CreatePreconditioner(const BlockSparseMatrix* A);
  void CreatePreSolver(const int max_num_spse_iterations,
                       const double spse_tolerance);
  void Initialize();
  void BackSubstitute(const double* reduced_system_solution, double* x);
  LinearSolver::Summary ReducedSolve(
      const ConjugateGradientsSolverOptions& cg_options);

 private:
  Vector reduced_linear_system_solution_;
  Vector scratch_[4];
};

}  // namespace ceres::internal

#include "ceres/internal/reenable_warnings.h"

#endif  // CERES_INTERNAL_ITERATIVE_SCHUR_COMPLEMENT_SOLVER_H_
