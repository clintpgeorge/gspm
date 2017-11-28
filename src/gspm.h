///////////////////////////////////////////////////////////////////////////////
// Utility functions for the GSPM MCMC algorithms
///////////////////////////////////////////////////////////////////////////////

#include "utils.h"

extern
  arma::vec calc_grad_z(
      double sigma,
      double alpha,
      const arma::vec & alpha_bar, // K-1 x 1
      const arma::vec & z, // K-1 x 1
      const arma::vec & theta, // K x 1
      const arma::mat & U, // M x K
      const arma::rowvec & y, // 1 x M
      unsigned int i);

extern
  arma::vec calc_grad_z(
      const arma::vec & sigma, // diagonal sigma
      double alpha,
      const arma::vec & alpha_bar, // K-1 x 1
      const arma::vec & z, // K-1 x 1
      const arma::vec & theta, // K x 1
      const arma::mat & U, // M x K
      const arma::rowvec & y, // 1 x M
      unsigned int i);

extern
  double calc_potential_energy(
      double sigma,
      double alpha,
      const arma::vec & alpha_bar, // K-1 x 1
      const arma::vec & z, // K-1 x 1
      const arma::vec & theta, // K x 1
      const arma::mat & U, // M x K
      const arma::rowvec & y);

extern
  double calc_potential_energy(
      const arma::vec & sigma, // diagonal sigma
      double alpha,
      const arma::vec & alpha_bar, // K-1 x 1
      const arma::vec & z, // K-1 x 1
      const arma::vec & theta, // K x 1
      const arma::mat & U, // M x K
      const arma::rowvec & y);

extern
  arma::vec map_betas_to_dirichlet(arma::vec z_vec);


