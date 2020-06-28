// Author: evanlevine138e@gmail.com (Evan Levine)

#ifndef CERES_PUBLIC_MARGINALIZATION_H_
#define CERES_PUBLIC_MARGINALIZATION_H_

#include <map>
#include <set>
#include <vector>

#include "ceres/problem.h"

namespace ceres {

// Background
// ==========
//
// Marginalization enables solving a problem for a subset of variables of
// interest at lower computational cost compared to solving the original
// problem. It requires making a linear approximation of the residuals with
// respect to the parameters to be marginalized out and the parameters that
// separate these variables from the rest of the graph, called the
// Markov blanket. The variables to marginalize out are replaced by an error
// term involving the variables in the Markov blanket. See [1], [2].
//
// The parameterization for the variables in the Markov blanket should be chosen
// judiciously. For example, accuracy may degrade if the second derivative of
// the residuals with respect to the tangent-space increment is large for the
// Markov blanket variables or for the variables to marginalize out.
//
// Consider a robustified non-linear least squares problem
//
// min_x 0.5 \sum_{i} ρ_i(\|f_i(x_i_1, ..., x_i_k)\|^2)
// s.t. l_j ≤ x_j ≤ u_j
//
// We can partition the variables into the variables to marginalize out,
// denoted x_m, the variables related to them by error terms (their Markov
// blanket), denoted x_b, and the remaining variables x_r. Suppose that the
// bounds l_j and u_j are trivial, -infinity and +infinity respectively, for
// x_m
//
// min_x 0.5 \sum_{i in ∂M} ρ_i(\|f_i(x_b, x_m)\|^2) +
//       0.5 \sum_{i not in ∂M} ρ_i(\|f_i(x_b, x_r)\|^2),
//
// where ∂M is the index set of all error terms involving x_m. Let x_b^0 and
// x_m^0 be linearization points for x_b and x_m to be respectively and [+] be
// the boxed plus operator. We can then make the following linear approximation
// for the first term.
//
// c(x_b, δm) = 0.5 \sum_{i in ∂M} ρ_i(\|f_i(x_b, x_m^0 [+] δm)\|^2)
//                 ~ 0.5 \sum_{i in ∂M} ρ_i(\|f_i(x_b^0, x_m^0) +
//                                            J_i [δb ; δm]\|^2),
// where J_i = [ ∂f_i/∂x_b ∂x_b/∂δb,  ∂f_i/∂x_m ∂x_m/∂δm], ";"
// denotes vertical concatenation, and δm is the error state for x_m =
// x_m^0 [+] δm.
//
// c(x_b,δm) = (g^T + [δb; δm]^T H) [δb; δm],
// where g = \sum_i ρ' J_i^T f_i(x_b^0, x_m^0),
//  H = \sum_i ρ' J_i^T J_i.
//
// Partition H into the block matrix
//  H = [ H_{mm}  H_{bm}^T ]
//      [ H_{bm}  H_{bb}   ].
// and g into the block vector g = [g_{mm}; g_{mb}].
//
// Minimize c(δb, δm) with respect to δm:
//
// argmin_{δm} c(δb, δm) =
//    H_{mm}^-1 (g_{mm} +  H_{mb}(δb))
//
// Substitution into c yields
//
// g_t^T(δb) + 0.5(δb) H_t(δb) + |f|^2,
//
// where  H_t = H_{bb} -  H_{bm} H_{mm}^{-1} H_{bm}^T
//        g_t = g_{mb} -  H_{bm} H_{mm}^{-1} g_{mm}.
//
// We can write this as
//
// \|D^(1/2) U^T δb + D^(-1/2) U^T g_t\|^2,
//
// where H_t = U * D * U^T is the eigen-decomposition of H_t with D
// containing only the nonzero eigenvalues on the diagonal. This is the cost
// function for the "marginalization prior" to be added to the graph with the
// marginalized parameter blocks removed.
//
// In this implementation, H_{mm} is assumed to be dense.
//
// [1] Carlevaris-Bianco, Nicholas, Michael Kaess, and Ryan M. Eustice. "Generic
// node removal for factor-graph SLAM." IEEE Transactions on Robotics 30.6
// (2014): 1371-1385.
//
// [2] Eckenhoff, Kevin, Liam Paull, and Guoquan Huang.
// "Decoupled, consistent node removal and edge sparsification for graph-based
// SLAM." 2016 IEEE/RSJ International Conference on Intelligent Robots and
// Systems (IROS). IEEE, 2016.

struct MarginalizationOptions {
  // Indicates whether to assume that the marginal information matrix is
  // full-rank. If set to false, an eigendecomposition step used to reduce the
  // dimensionality of the marginal information is used. Otherwise, a Cholesky
  // factorization of the marginal information is used.
  bool assume_marginal_information_is_full_rank = false;

  // Whether to assume that the block of the information matrix corresponding to
  // marginalized variables is full rank ( H_{mm}).
  bool assume_marginalized_block_is_full_rank = false;
};

// Marginalize out a set of variables. If the computation fails, returns false
// and does not modify the problem. If the computations succeeds, removes the
// variables to marginalize, adds a linear cost function for the marginalization
// prior and returns true. If marginalization_prior_id is not null, the residual
// block for the marginalization prior is returned in it. Optionally,
// linearization points used for Jacobians can be provided in
// parameter_block_linearization_states, a mapping from user pointers to the
// parameter blocks to pointers to values to be used for linearization.
bool MarginalizeOutVariables(
    const MarginalizationOptions& options,
    const std::set<double*>& parameter_blocks_to_marginalize,
    Problem* problem,
    std::vector<ResidualBlockId>* marginalization_prior_ids,
    const std::map<const double*, const double*>*
        parameter_block_linearization_states = nullptr);

}  // namespace ceres

#endif  // CERES_PUBLIC_MARGINALIZATION_H_
