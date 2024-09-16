// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2023 Google Inc. All rights reserved.
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

#include "ceres/single_linkage_clustering.h"

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "ceres/graph.h"
#include "ceres/graph_algorithms.h"

namespace ceres::internal {

int ComputeSingleLinkageClustering(
    const SingleLinkageClusteringOptions& options,
    const WeightedGraph<int>& graph,
    absl::flat_hash_map<int, int>* membership) {
  CHECK(membership != nullptr);
  membership->clear();

  // Initially each vertex is in its own cluster.
  const absl::flat_hash_set<int>& vertices = graph.vertices();
  for (const int v : vertices) {
    (*membership)[v] = v;
  }

  for (const int vertex1 : vertices) {
    const absl::flat_hash_set<int>& neighbors = graph.Neighbors(vertex1);
    for (const int vertex2 : neighbors) {
      // Since the graph is undirected, only pay attention to one side
      // of the edge and ignore weak edges.
      if ((vertex1 > vertex2) ||
          (graph.EdgeWeight(vertex1, vertex2) < options.min_similarity)) {
        continue;
      }

      // Use a union-find algorithm to keep track of the clusters.
      const int c1 = FindConnectedComponent(vertex1, membership);
      const int c2 = FindConnectedComponent(vertex2, membership);

      if (c1 == c2) {
        continue;
      }

      if (c1 < c2) {
        (*membership)[c2] = c1;
      } else {
        (*membership)[c1] = c2;
      }
    }
  }

  // Make sure that every vertex is connected directly to the vertex
  // identifying the cluster.
  int num_clusters = 0;
  for (auto& m : *membership) {
    m.second = FindConnectedComponent(m.first, membership);
    if (m.first == m.second) {
      ++num_clusters;
    }
  }

  return num_clusters;
}

}  // namespace ceres::internal
