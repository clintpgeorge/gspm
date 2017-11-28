#include "gspm.h"

// Calcuate the gradient of a z vector (K-1 x 1 vector)
//
// Reference:
//   Betancourt (2010). Cruising The Simplex: Hamiltonian Monte Carlo and the
//   Dirichlet Distribution.
//
arma::vec calc_grad_z(
    double sigma,
    double alpha,
    const arma::vec & alpha_bar, // K-1 x 1
    const arma::vec & z, // K-1 x 1
    const arma::vec & theta, // K x 1
    const arma::mat & U, // M x K
    const arma::rowvec & y, // 1 x M
    unsigned int i) {
  unsigned int K = theta.size(); // number of subtypes
  unsigned int k, r;
  arma::vec mu = U * theta; // M x K * K x 1 -> M x 1
  arma::vec y_i = arma::trans(y); // M x 1

  arma::vec grad_f_theta = arma::zeros<arma::vec>(K);
  for (k = 0; k < K; k++) {
    grad_f_theta(k) = arma::accu((y_i - mu) % U.col(k)) / (sigma * sigma);
    if (!is_finite(grad_f_theta(k))){
      ::Rf_error("NaN in grad_f_theta (i = %d, k = %d)", i, k);
    }
  }

  arma::vec grad_f_z = arma::zeros<arma::vec>(K - 1);
  for (k = 0; k < (K - 1); k++) {
    grad_f_z += theta(k) * grad_f_theta(k) / (z(k) - 1.);
    for (r  = (k + 1); r < K; r++) {
      grad_f_z(k) += theta(r) * grad_f_theta(r) / z(k);
    }
    if (!is_finite(grad_f_z(k))){
      ::Rf_error("NaN in grad_f_z (i = %d, k = %d)", i, k);
    }
  }

  arma::vec grad_z = (alpha_bar / z) - (alpha / (1. - z)) + grad_f_z;
  return grad_z;
}


// Calcuate the gradient of a z vector (K-1 x 1 vector)
//
// Reference:
//   Betancourt (2010). Cruising The Simplex: Hamiltonian Monte Carlo and the
//   Dirichlet Distribution.
//
arma::vec calc_grad_z(
    const arma::vec & sigma, // diagonal sigma
    double alpha,
    const arma::vec & alpha_bar, // K-1 x 1
    const arma::vec & z, // K-1 x 1
    const arma::vec & theta, // K x 1
    const arma::mat & U, // M x K
    const arma::rowvec & y, // 1 x M
    unsigned int i) {
  unsigned int K = theta.size(); // number of subtypes
  unsigned int k, r;
  arma::vec mu = U * theta; // M x K * K x 1 -> M x 1
  arma::vec y_i = arma::trans(y); // M x 1

  arma::vec grad_f_theta = arma::zeros<arma::vec>(K);
  for (k = 0; k < K; k++) {
    grad_f_theta(k) = arma::accu( (y_i - mu) % U.col(k) / (sigma % sigma) );
    if (!is_finite(grad_f_theta(k))){
      ::Rf_error("NaN in grad_f_theta (i = %d, k = %d)", i, k);
    }
  }

  arma::vec grad_f_z = arma::zeros<arma::vec>(K - 1);
  for (k = 0; k < (K - 1); k++) {
    grad_f_z += theta(k) * grad_f_theta(k) / (z(k) - 1.);
    for (r  = (k + 1); r < K; r++) {
      grad_f_z(k) += theta(r) * grad_f_theta(r) / z(k);
    }
    if (!is_finite(grad_f_z(k))){
      ::Rf_error("NaN in grad_f_z (i = %d, k = %d)", i, k);
    }
  }

  arma::vec grad_z = (alpha_bar / z) - (alpha / (1. - z)) + grad_f_z;
  return grad_z;
}


// Calculate potential energy (P.E), i.e. -log(p (omega_i | U, y_i)) for a
// sample i
//
//
double calc_potential_energy(
    double sigma,
    double alpha,
    const arma::vec & alpha_bar, // K-1 x 1
    const arma::vec & z, // K-1 x 1
    const arma::vec & theta, // K x 1
    const arma::mat & U, // M x K
    const arma::rowvec & y) { // 1 x M
  arma::rowvec mu = arma::trans(U * theta);
  double pe = arma::accu(
    alpha_bar % arma::log(z)
    - alpha * arma::log(1. - z)
  );
  pe -= .5 * arma::accu(arma::pow((y - mu) / sigma, 2));
  return -pe;
}

// Calculate potential energy (P.E), i.e. -log(p (omega_i | U, y_i)) for a
// sample i
//
//
double calc_potential_energy(
    const arma::vec & sigma, // diagonal sigma
    double alpha,
    const arma::vec & alpha_bar, // K-1 x 1
    const arma::vec & z, // K-1 x 1
    const arma::vec & theta, // K x 1
    const arma::mat & U, // M x K
    const arma::rowvec & y) { // 1 x M
  arma::vec mu = U * theta; // M x 1
  double pe = arma::accu(
    alpha_bar % arma::log(z)
    - alpha * arma::log(1. - z)
  );
  pe -= .5 * arma::accu( arma::pow((trans(y) - mu) / sigma, 2) );
  return -pe;
}

// Map \code{K - 1} z's that are in [0, 1] space to a \code{K - 1} simplex
//
// Reference:
//   Betancourt (2010). Cruising The Simplex: Hamiltonian Monte Carlo and the
//   Dirichlet Distribution.
//
// @param z_vec a \code{K - 1} vector of beta samples
//
// @return a \code{K - 1} vector of Dirichlet probabilities
//
// @note
// Author: Clint P. George
//
// License: GPL-3
//
//
arma::vec map_betas_to_dirichlet(arma::vec z_vec) {
  unsigned int K = z_vec.size() + 1;
  arma::vec cs_z = arma::cumsum(arma::log(z_vec));
  arma::vec log_theta = arma::zeros<arma::vec>(K);
  for (unsigned int k = 0; k < K; k++) {
    if (k == 0) {
      log_theta[k] = std::log(1. - z_vec[k]);
    } else if (k == K) {
      log_theta[k] = cs_z[k - 1];
    } else {
      log_theta[k] = cs_z[k - 1] + std::log(1. - z_vec[k]);
    }
  }
  return arma::exp(log_theta);
}
