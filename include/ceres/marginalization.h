// Author: evanlevine138e@gmail.com (Evan Levine)

#ifndef CERES_PUBLIC_MARGINALIZATION_H_
#define CERES_PUBLIC_MARGINALIZATION_H_

#include <set>

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
// min_x 0.5 \sum_{i} rho_i(\|f_i(x_i_1, ..., x_i_k)\|^2)
// s.t. l_j \leq x_j \leq u_j
//
// We can partition the variables into the variables to marginalize out,
// denoted x_m, the variables related to them by error terms (their Markov
// blanket), denoted x_b, and the remaining variables x_r.
//
// min_x 0.5 \sum_{i in dM} rho_i(\|f_i(x_b, x_m)\|^2) +
//       0.5 \sum_{i not in dM} rho_i(\|f_i(x_b, x_r)\|^2),
//
// where dM is the index set of all error terms involving x_m. Let x_b^0 and
// x_m^0 be linearization points for x_b and x_m to be respectively and (+) be
// the oplus operator. We can then make the following linear approximation for
// the first term.
//
// c(x_b, delta_m) = 0.5 \sum_{i in dM} rho_i(\|f_i(x_b, x_m^0(+)delta_m)\|^2)
//                 ~ 0.5 \sum_{i in dM} rho_i(\|f_i(x_b^0, x_m^0) +
//                                            J_i [delta_b ; delta_m]\|^2),
// where J_i = [ df_i/dx_b dx_b/d_delta_b,  df_i/dx_m dx_m/d_delta_m], ";"
// denotes vertical concatenation, and delta_m is the error state for x_m =
// x_m^0 (+) delta_m.
//
// c(x_b,delta_m) = (g^T + [delta_b; delta_m]^T\Lambda) [delta_b; delta_m],
// where g = \sum_i \rho^\prime J_i^T f_i(x_b^0, x_m^0),
// \Lambda = \sum_i \rho^\prime J_i^T J_i.
//
// Partition lambda into the block matrix
// \Lambda = [ \Lambda_{mm} \Lambda_{bm}^T ]
//           [ \Lambda_{bm} \Lambda_{bb}   ].
// and g into the block vector g = [g_{mm}; g_{mb}].
//
// Minimize c(delta_b, delta_m) with respect to delta_m:
//
// argmin_{delta_m} c(delta_b, delta_m) =
//   \Lambda_{mm}^-1 (g_{mm} + \Lambda_{mb}(delta_b))
//
// Substitution into c yields
//
// g_t^T(delta_b) + 0.5(delta_b)\Lambda_t(delta_b) + |f|^2,
//
// where \Lambda_t = \Lambda_{bb} - \Lambda_{bm}\Lambda_{mm}^{-1}\Lambda_{bm}^T
//             g_t = g_{mb} - \Lambda_{bm}\Lambda_{mm}^{-1}g_{mm}.
//
// We can write this as
//
// \|D^(1/2) U^T delta_b + D^(-1/2) U^T g_t\|^2,
//
// where Lambda_t = U * D * U^T is the eigen-decomposition of Lambda_t with D
// containing only the nonzero eigenvalues on the diagonal. This is the cost
// function for the "marginalization prior" to be added to the graph with the
// marginalized parameter blocks removed.
//
// In this implementation, Lambda_{mm} is assumed to be dense.
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
  // If set to true, the eigendecomposition step used to reduce the
  // dimensionality of the marginal information is skipped. Otherwise, the
  // marginal information is not assumed to be full-rank and a Cholesky factor
  // of the marginal information is used instead.
  bool assume_full_rank = false;
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
    const std::map<double*, const double*>*
        parameter_block_linearization_states = nullptr);

}  // namespace ceres

#endif  // CERES_PUBLIC_MARGINALIZATION_H_
