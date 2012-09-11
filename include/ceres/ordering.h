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

#include <map>
#include <set>
#include "ceres/internal/port.h"
#include "glog/logging.h"

namespace ceres {

// The order in which variables are eliminated in a linear solver can
// have a significant of impact on the efficiency and accuracy of the
// method. e.g., when doing sparse Cholesky factorization, there are
// matrices for which a good ordering will give a Cholesky factor with
// O(n) storage, where as a bad ordering will result in an completely
// dense factor.
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
// Formally an ordering is an ordered partitioning of the parameter
// blocks, i.e, each parameter block belongs to exactly one group, and
// each group has a unique integer associated with it, that determines
// its order in the set of groups.
//
// Given such an ordering, Ceres ensures that the parameter blocks in
// the lowest numbered group are eliminated first, and then the
// parmeter blocks in the next lowest numbered group and so on. Within
// each group, Ceres is free to order the parameter blocks as it
// chooses.
//
// e.g. Consider the linear system
//
//   x + y = 3
//   2x + 3y = 7
//
// There are two ways in which it can be solved. First eliminating x
// from the two equations, solving for y and then back substituting
// for x, or first eliminating y, solving for x and back substituting
// for y. The user can construct three orderings here.
//
// {0: x}, {1: y} - eliminate x first.
// {0: y}, {1: x} - eliminate y first.
// {0: x, y}      - Solver gets to decide the elimination order.
//
// Thus, yo have Ceres determine the ordering automatically using
// heuristics, put all the variables in group 0 and to control the
// ordering for every variable, create groups 0..N-1, one per
// variable, in the desired order.
//
// Bundle Adjustment
// -----------------
//
// A particular case of interest is bundle adjustment. Where the user
// has two options. The default is to not specify an ordering at all,
// the solver will see that the user wants to use a Schur type solver
// and figure out the right elimination ordering.
//
// But if the user already knows what parameter blocks are points and
// what are cameras, she can save preprocessing time by partitioning
// the parameter blocks into two groups, one for the points and one
// for the cameras, where the group containing the points has an id
// smaller than the group containing cameras.
class Ordering {
 public:
  // Add a parameter block to a group with id group_id. If a group
  // with this id does not exist, one is created. This method can be
  // called any number of times for a parameter block.
  void AddParameterBlock(double* parameter_block, int group_id) {
    const map<double*, int>::const_iterator it =
        parameter_block_to_group_id_.find(parameter_block);

    if (it != parameter_block_to_group_id_.end()) {
      const int current_group_id = it->second;
      group_id_to_parameter_blocks_[group_id].erase(parameter_block);
    }

    parameter_block_to_group_id_[parameter_block] = group_id;
    group_id_to_parameter_blocks_[group_id].insert(parameter_block);
  }

  // Return the group id for the parameter block. If the parameter
  // block is not known to the Ordering, calling this method results
  // in a crash.
  int GetGroupId(double* parameter_block) const {
    const map<double*, int>::const_iterator it =
        parameter_block_to_group_id_.find(parameter_block);
    CHECK(it !=  parameter_block_to_group_id_.end());
    return it->second;
  }

  int NumParameterBlocks() const {
    return parameter_block_to_group_id_.size();
  }

  int NumGroups() const { return group_id_to_parameter_blocks_.size(); }

  const map<int, set<double*> >& group_id_to_parameter_blocks() const {
    return group_id_to_parameter_blocks_;
  }

 private:
  map<int, set<double*> > group_id_to_parameter_blocks_;
  map<double*, int>  parameter_block_to_group_id_;
};

}  // namespace ceres
