// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

// gspm_ags_hmc
List gspm_ags_hmc(unsigned int K, double alpha, double lambda, double sigma, unsigned int sample_sigma, unsigned int sigma_update_iter, double gamma_shape, double gamma_rate, unsigned int max_iter, double epsilon, arma::mat Y, unsigned int verbose);
RcppExport SEXP gspm_gspm_ags_hmc(SEXP KSEXP, SEXP alphaSEXP, SEXP lambdaSEXP, SEXP sigmaSEXP, SEXP sample_sigmaSEXP, SEXP sigma_update_iterSEXP, SEXP gamma_shapeSEXP, SEXP gamma_rateSEXP, SEXP max_iterSEXP, SEXP epsilonSEXP, SEXP YSEXP, SEXP verboseSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< unsigned int >::type K(KSEXP);
    Rcpp::traits::input_parameter< double >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< double >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< double >::type sigma(sigmaSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type sample_sigma(sample_sigmaSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type sigma_update_iter(sigma_update_iterSEXP);
    Rcpp::traits::input_parameter< double >::type gamma_shape(gamma_shapeSEXP);
    Rcpp::traits::input_parameter< double >::type gamma_rate(gamma_rateSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type max_iter(max_iterSEXP);
    Rcpp::traits::input_parameter< double >::type epsilon(epsilonSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type Y(YSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type verbose(verboseSEXP);
    rcpp_result_gen = Rcpp::wrap(gspm_ags_hmc(K, alpha, lambda, sigma, sample_sigma, sigma_update_iter, gamma_shape, gamma_rate, max_iter, epsilon, Y, verbose));
    return rcpp_result_gen;
END_RCPP
}
// spmm_ags_hmc
List spmm_ags_hmc(unsigned int K, double alpha, double lambda, double sigma, unsigned int max_iter, double epsilon, arma::mat Y, unsigned int verbose);
RcppExport SEXP gspm_spmm_ags_hmc(SEXP KSEXP, SEXP alphaSEXP, SEXP lambdaSEXP, SEXP sigmaSEXP, SEXP max_iterSEXP, SEXP epsilonSEXP, SEXP YSEXP, SEXP verboseSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< unsigned int >::type K(KSEXP);
    Rcpp::traits::input_parameter< double >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< double >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< double >::type sigma(sigmaSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type max_iter(max_iterSEXP);
    Rcpp::traits::input_parameter< double >::type epsilon(epsilonSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type Y(YSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type verbose(verboseSEXP);
    rcpp_result_gen = Rcpp::wrap(spmm_ags_hmc(K, alpha, lambda, sigma, max_iter, epsilon, Y, verbose));
    return rcpp_result_gen;
END_RCPP
}
// arma_rgig
arma::vec arma_rgig(int n, double lambda, double chi, double psi);
RcppExport SEXP gspm_arma_rgig(SEXP nSEXP, SEXP lambdaSEXP, SEXP chiSEXP, SEXP psiSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type n(nSEXP);
    Rcpp::traits::input_parameter< double >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< double >::type chi(chiSEXP);
    Rcpp::traits::input_parameter< double >::type psi(psiSEXP);
    rcpp_result_gen = Rcpp::wrap(arma_rgig(n, lambda, chi, psi));
    return rcpp_result_gen;
END_RCPP
}
// sample_antoniak
double sample_antoniak(unsigned int N, double alpha);
RcppExport SEXP gspm_sample_antoniak(SEXP NSEXP, SEXP alphaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< unsigned int >::type N(NSEXP);
    Rcpp::traits::input_parameter< double >::type alpha(alphaSEXP);
    rcpp_result_gen = Rcpp::wrap(sample_antoniak(N, alpha));
    return rcpp_result_gen;
END_RCPP
}
// sample_uniform_int
unsigned int sample_uniform_int(unsigned int K);
RcppExport SEXP gspm_sample_uniform_int(SEXP KSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< unsigned int >::type K(KSEXP);
    rcpp_result_gen = Rcpp::wrap(sample_uniform_int(K));
    return rcpp_result_gen;
END_RCPP
}
// sample_multinomial
unsigned int sample_multinomial(arma::vec theta);
RcppExport SEXP gspm_sample_multinomial(SEXP thetaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type theta(thetaSEXP);
    rcpp_result_gen = Rcpp::wrap(sample_multinomial(theta));
    return rcpp_result_gen;
END_RCPP
}
// sample_dirichlet
arma::vec sample_dirichlet(unsigned int num_elements, arma::vec alpha);
RcppExport SEXP gspm_sample_dirichlet(SEXP num_elementsSEXP, SEXP alphaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< unsigned int >::type num_elements(num_elementsSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type alpha(alphaSEXP);
    rcpp_result_gen = Rcpp::wrap(sample_dirichlet(num_elements, alpha));
    return rcpp_result_gen;
END_RCPP
}
// arma_mvrnorm
arma::mat arma_mvrnorm(int n, arma::vec mu, arma::mat sigma);
RcppExport SEXP gspm_arma_mvrnorm(SEXP nSEXP, SEXP muSEXP, SEXP sigmaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type n(nSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type mu(muSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type sigma(sigmaSEXP);
    rcpp_result_gen = Rcpp::wrap(arma_mvrnorm(n, mu, sigma));
    return rcpp_result_gen;
END_RCPP
}
