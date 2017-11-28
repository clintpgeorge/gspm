#include "utils.h"
#include "gspm.h"

//' Hamiltonian Monte Carlo within Augmented Gibbs sampler for Gaussian
//' Shrinkage Partial Membership models
//'
//' A GSPM model is indexed by scalar parameters K, alpha, lambda, and sigma
//'
//' @param K Number of components or subtypes
//' @param alpha Hyperparameter for the Dirichlet prior
//' @param lambda Variance smoothing parameter for the normal prior
//' @param sigma Variance parameter for the data generating normal
//' @param max_iter Maximum number of iterations for the HMC within AGS
//'                 algorithm
//' @param epsilon The step size of the HMC leep-frog iteration
//' @param Y The data matrix, an \code{n_samples x M} matrix, where
//'          \code{n_samples} is the number of observations and \code{M} is the
//'          number of features or genes
//' @param verbose 0 - no output; 1 - some output; 2 - more output
//'
//' @note
//' Author: Clint P. George
//'
//' License: GPL-3
//'
//' @family mcmc
//'
//' @export
// [[Rcpp::export]]
List spmm_ags_hmc(
    unsigned int K,
    double alpha,
    double lambda,
    double sigma,
    unsigned int max_iter,
    double epsilon,
    arma::mat Y,
    unsigned int verbose) {

  // Declare and initialize variables
  //
  unsigned int k, j, i, iter, kk, l;
  unsigned int L_iter;
  unsigned int M = Y.n_cols; // Number of features or genes
  unsigned int n_samples = Y.n_rows; // Number of samples
  double C_jk;
  double B_jk;
  double bij;
  double mu_jk;
  double sigma_jk;
  double yut;
  double ke_prop, ke;
  double pe_prop, pe;
  double H, H_prop;
  double accept_ratio;
  double epsilon_iter;
  arma::vec z_prop;
  arma::vec u_z;
  arma::vec grad_z;
  arma::vec grad_z_prop;
  arma::vec theta_prop;
  arma::uvec gt_1_idx;
  arma::uvec lt_0_idx;
  arma::vec A;
  arma::mat U_Theta_sum;
  cube Psi_samples = cube(M, K, max_iter);
  cube U_samples = cube(M, K, max_iter);
  cube Z_samples = cube(K - 1, n_samples, max_iter);
  cube Theta_samples = cube(K, n_samples, max_iter);

  vec alpha_bar = zeros<vec>(K - 1);
  vec log_post_Psi_U = zeros<vec>(max_iter);
  vec log_likelihood = zeros<vec>(max_iter); // Log likelihood for the GSPM model
  mat H_values = zeros<mat>(max_iter, n_samples);
  mat PE_values = zeros<mat>(max_iter, n_samples);
  mat Z_WallHMC_count = zeros<mat>(K - 1, n_samples);
  vec num_accept = zeros<vec>(n_samples);
  mat Psi = zeros<mat>(M, K);
  mat U = zeros<mat>(M, K);
  mat Z = zeros<mat>(K - 1, n_samples);
  mat Theta = zeros<mat>(K, n_samples);

  for (k = 0; k < (K - 1); k++) {
    alpha_bar(k) = (K - k - 1) * alpha;
  }

  for (i = 0; i < n_samples; i++) {
    for (k = 0; k < (K - 1); k++) {
      Z(k, i) = Rf_rbeta(alpha_bar(k), alpha);
    }
    Theta.col(i) = map_betas_to_dirichlet(Z.col(i));
  }

  for (j = 0; j < M; j++) {
    U.row(j) = Rcpp::as<arma::rowvec>(Rcpp::rnorm(K, 0, .1)); // normal with 0 mean and .1 s.d.
  }

  if (verbose > 0) {
    cout << endl << "Hamiltonian Monte Carlo within Gibbs sampling..." << endl;
  }

  for (iter = 0; iter < max_iter; iter++) { // for each HMC within AGS iteration

    // Save each sample
    U_samples.slice(iter) = U;
    Psi_samples.slice(iter) = Psi;
    Theta_samples.slice(iter) = Theta;
    Z_samples.slice(iter) = Z;

    // ==================== AGS ====================
    //
    if (verbose > 1) {
      cout << "Augmented Gibbs sampling - Interation " << (iter + 1);
    }
    A = arma::sum(arma::pow(Theta, 2), 1); // row sums
    for (j = 0; j < M; j++) { // for each gene or feature

      for (k = 0; k < K; k++) { // for each subtype
        // Sampling \psi_jk from generalized inverse Gaussian (GiG)
        Psi(j, k) = arma_rgig(1, .5, pow((U(j, k) / lambda), 2.), 1.)(0);

        // Sampling x_jk from a univariate Gaussian
        C_jk = Psi(j, k) * lambda * lambda;
        B_jk = 0.;
        for (i = 0; i < n_samples; i++) {
          bij = Y(i, j);
          for (kk = 0; kk < K; kk++) {
            if (k != kk) {
              bij -=  U(j, kk) * Theta(kk, i);
            }
          }
          B_jk += bij * Theta(k, i);
        }
        mu_jk = B_jk / (((sigma * sigma) / C_jk) + A(k)); // mean
        sigma_jk = 1. / sqrt((1. / C_jk) + (A(k) / (sigma * sigma))); // s.d.
        U(j, k) = R::rnorm(mu_jk, sigma_jk); // returns one normal variate
      } // end of loop for each subtype

    } // end of loop for each gene or feature

    // Computes the log posterior of Psi and U given Theta and Y (upto a
    // constant)
    U_Theta_sum = U * Theta; // M x K * K x n_samples -> M x n_samples
    yut = 0.;
    for (i = 0; i < n_samples; i++) { // for each observation
      yut += arma::accu(
        arma::pow((Y.row(i) - trans(U_Theta_sum.col(i))) / sigma, 2)
      );
    } // for each observation
    log_post_Psi_U(iter) = -.5 * (
      arma::accu(arma::pow(U, 2) / (Psi * lambda * lambda)) + yut
      );
    if (verbose > 1) {
      cout << ": log posterior (Psi, U) = " << log_post_Psi_U(iter) << endl;
    }
    // ==================== AGS ====================

    // ==================== HMC ====================
    //
    // Resets HMC tuning parameters: step size and leep-frog iterations
    epsilon_iter = (sample_uniform_int(5) + 1.) * epsilon;
    L_iter = (sample_uniform_int(5) + 1) * 2;

    if (verbose > 1) {
      cout << "Hamiltonian Monte Carlo - Iteration " << (iter + 1)
           << ": epsilon = " << epsilon_iter
           << " L = " << L_iter << endl;
    }
    for (i = 0; i < n_samples; i++) { // HMC for each observation

      if (verbose > 1) {
        cout << "HMC Iteration " <<  (iter + 1) << " sidx " << i << ":";
      }
      // generate u based on independent standard normal variates
      // with mu = 0 and sd = 1.
      // alt. arma::randn<arma::vec>(K - 1); //
      u_z = Rcpp::as<arma::vec>(Rcpp::rnorm(K - 1));
      grad_z = calc_grad_z(
        sigma,
        alpha,
        alpha_bar, // K-1 x 1
        Z.col(i), // K-1 x 1
        Theta.col(i), // K x 1
        U, // M x K
        Y.row(i), // 1 x M
        i
      );

      pe = calc_potential_energy(
        sigma,
        alpha,
        alpha_bar, // K-1 x 1
        Z.col(i), // K-1 x 1
        Theta.col(i), // K x 1
        U, // M x K
        Y.row(i)
      );
      ke = arma::accu(arma::pow(u_z, 2)); // Kinetic Energy K(u)
      H = pe + ke; // the Hamiltonian H

      // initialize variables for leapfrog steps
      z_prop = Z.col(i);
      for (l = 0; l < L_iter; l++) { // for each leapfrog step

        // cout << "LF Iteration #" << l << endl;

        // Make a half step in u
        u_z -= .5 * epsilon_iter * grad_z;

        // Make a full step in Z
        z_prop += epsilon_iter * u_z;

        // ============= Wall HMC ==================================
        // Handling the constraint 0 <= z.i.k <= 1
        // Reference: Neal (2011), Figure 5.8

        gt_1_idx = arma::find(z_prop > 1.);
        lt_0_idx = arma::find(z_prop < 0.);

        // TODO: need to find a better way to do the following
        arma::vec z_WHMC_count = Z_WallHMC_count.col(i);
        z_WHMC_count.elem(gt_1_idx) += 1;
        z_WHMC_count.elem(lt_0_idx) += 1;
        Z_WallHMC_count.col(i) = z_WHMC_count;

        while ((gt_1_idx.n_elem + lt_0_idx.n_elem) > 0) {
          z_prop.elem(gt_1_idx) = 1. - (z_prop.elem(gt_1_idx) - 1.);
          u_z.elem(gt_1_idx) = -u_z.elem(gt_1_idx);

          z_prop.elem(lt_0_idx) = 0. + (0. - z_prop.elem(lt_0_idx));
          u_z.elem(lt_0_idx) = -u_z.elem(lt_0_idx);

          gt_1_idx = arma::find(z_prop > 1.);
          lt_0_idx = arma::find(z_prop < 0.);
        }

        // ============= Wall HMC ==================================

        // Compute the gradient w.r.t. the new z and theta
        theta_prop = map_betas_to_dirichlet(z_prop); // K-1 -> K
        grad_z_prop = calc_grad_z(
          sigma,
          alpha,
          alpha_bar, // K-1 x 1
          z_prop, // K-1 x 1
          theta_prop, // K x 1
          U, // M x K
          Y.row(i), // 1 x M
          i
        );

        // Make a half step in u
        u_z -= .5 * epsilon_iter * grad_z_prop;

      } // for each leapfrog step


      ke_prop = arma::accu(arma::pow(u_z, 2)); // Kinetic Engergy K(u)
      if (!is_finite(ke_prop)) {
        stop("Proposed K.E. is infinite!");
      }
      pe_prop = calc_potential_energy(
        sigma,
        alpha,
        alpha_bar, // K-1 x 1
        z_prop, // K-1 x 1
        theta_prop, // K x 1
        U, // M x K
        Y.row(i)
      ); // Potential Energy
      if (!is_finite(pe_prop)) {
        stop("Proposed P.E. is infinite!");
      }
      H_prop = pe_prop + ke_prop; // the Hamiltonian H

      // accepts or rejects the HMC random walk
      accept_ratio = exp(-(H_prop - H));
      if(Rcpp::runif(1)(0) < accept_ratio) {
        num_accept(i) += 1;
        Z.col(i) = z_prop;
        Theta.col(i) = theta_prop;
        H_values(iter, i) = H_prop;
        PE_values(iter, i) = pe_prop;
        if (verbose > 1) {
          cout << " p.e. = " << pe_prop
               << " k.e. = " << ke_prop
               << " Accept." << endl;
        } else if (verbose > 0) {
          cout << "+";
        }
      } else {
        H_values(iter, i) = H;
        PE_values(iter, i) = pe;
        if (verbose > 1) {
          cout << " p.e. = " << pe
               << " k.e. = " << ke
               << " Reject." << endl;
        } else if (verbose > 0) {
          cout << "-";
        }
      }

      for (j = 0; j < M; j++) { // for each gene or feature
        double m_0 = arma::accu(arma::trans(U.row(j)) % Theta.col(i));
        log_likelihood(iter) -= std::pow(Y(i,j) - m_0, 2) / (2 * sigma * sigma);
      }


    } // end of HMC for each observation

    // ==================== HMC ====================

  } // end of an HMC within AGS iteration

  if (verbose > 0) {
    cout << endl << "End of Hamiltonian Monte Carlo within Gibbs sampling..."
         << endl;
  }

  return List::create(
    Named("num_accept") = wrap(num_accept),
    Named("log_post_Psi_U") = wrap(log_post_Psi_U),
    Named("H_values") = wrap(H_values),
    Named("PE_values") = wrap(PE_values),
    Named("Theta_samples") = wrap(Theta_samples),
    Named("Z_samples") = wrap(Z_samples),
    Named("Z_WallHMC_count") = wrap(Z_WallHMC_count),
    Named("U_samples") = wrap(U_samples),
    Named("Psi_samples") = wrap(Psi_samples),
    Named("log_likelihood") = wrap(log_likelihood)
  );

}
