// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2012 Google Inc. All rights reserved.
// http://code.google.com/p/ceres-solver/
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

namespace ceres {

// The order in which variables are eliminated in a linear or
// non-linear solve has a lot of impact on the efficiency of the
// method. e.g., when doing sparse Cholesky factorization, a good
// ordering will give a Cholesky factor with O(n) storage, where as a
// bad ordering will result in an completely dense factor.
//
// Ceres allows the user to provide varying amounts of hints to the
// solver about the variable elimination ordering to use. This can
// range from no hints, where the solver is free to decide the best
// possible ordering based on the user's choices like the linear
// solver being used, to an exact order in which the variables should
// be eliminated, and a variety of possibilities in between. Instances
// of the Ordering class are used to communicating this infornation to
// Ceres.
//
// Formally an ordering is ordered partitioning of the parameter
// blocks, i.e, each parameter block belongs to exactly one group, and
// each group has a unique integer associated with it, that determines
// its order in the set of groups.
//
// Each group may also be associated with a type, which indicates
// extra knowledge about the group that the user wishes to communicate
// to the solver.
//
// Given such an ordering, Ceres ensures that the parameter blocks in
// the lowest numbered group are eliminated first, and then the
// parmeter blocks in the next lowest numbered group and so on. Within
// each group, Ceres is free to do as it chooses.
//
// So if the user wishes Ceres to take all decision, putting all
// parameter blocks in a single group.
//
// If the user wishes to indicate the order exactly, then if there are
// n parameter blocks, then she should create n groups with one
// parameter block each, with the appropriate ids.
//
// A particular case of interest is structure from motion
// problems. Where the user has two options. The default is that all
// the parameter blocks are in group 0, and the solver will see that
// the user wants to use a Schur type solver and will figure out the
// right elimination ordering.
//
// But if the user already knows what parameter blocks are points and
// what are cameras, she can save preprocessing time it would be
// useful to partition the parameter blocks into two groups - points
// and cameras, where the group containing the point parameter blocks
// has an id smaller than the group containing cameras.
class Ordering {
 public:
  Ordering(const Problem& problem);
  void SetGroupId(const double* parameter_block, int group_id);
  int GetGroupId(const double* parameter_block) const;
  int NumParameterBlocks() const;
  int NumGroups() const;
  const map<int, vector<const double*> >& group_to_parameter_blocks() const;
  const map<const double*, int>& parameter_block_to_group_;

 private:
  const Problem& problem;
  map<int, vector<const double*> > group_to_parameter_blocks_;
  map<const double*, int  parameter_block_to_group_;
};

}  // namespace ceres
