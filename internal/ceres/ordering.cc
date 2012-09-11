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
#include "ceres/ordering.h"
#include "glog/logging.h"

namespace ceres {

void Ordering::AddParameterBlock(double* parameter_block, int group_id) {
  const map<double*, int>::const_iterator it =
      parameter_block_to_group_id_.find(parameter_block);

  if (it != parameter_block_to_group_id_.end()) {
    group_id_to_parameter_blocks_[it->second].erase(parameter_block);
  }

  parameter_block_to_group_id_[parameter_block] = group_id;
  group_id_to_parameter_blocks_[group_id].insert(parameter_block);
}

void Ordering::RemoveParameterBlock(double* parameter_block) {
  map<double*, int>::const_iterator it =
      parameter_block_to_group_id_.find(parameter_block);
  CHECK(it != parameter_block_to_group_id_.end());

  const int current_group_id = it->second;
  group_id_to_parameter_blocks_[current_group_id].erase(parameter_block);
  if (group_id_to_parameter_blocks_[current_group_id].size() == 0) {
    group_id_to_parameter_blocks_.erase(current_group_id);
  }

  parameter_block_to_group_id_.erase(parameter_block);
}

int Ordering::GroupIdForParameterBlock(double* parameter_block) const {
  const map<double*, int>::const_iterator it =
      parameter_block_to_group_id_.find(parameter_block);
  CHECK(it !=  parameter_block_to_group_id_.end());
  return it->second;
}

int Ordering::NumParameterBlocks() const {
  return parameter_block_to_group_id_.size();
}

int Ordering::NumGroups() const { return group_id_to_parameter_blocks_.size(); }

int Ordering::GroupSize(int group_id) const {
  map<int, set<double*> >::const_iterator it =
      group_id_to_parameter_blocks_.find(group_id);
  return (it ==  group_id_to_parameter_blocks_.end()) ? 0 : it->second.size();
}

const map<int, set<double*> >& Ordering::group_id_to_parameter_blocks() const {
  return group_id_to_parameter_blocks_;
}

}  // namespace ceres
