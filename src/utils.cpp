///////////////////////////////////////////////////////////////////////////////
// Utility functions
///////////////////////////////////////////////////////////////////////////////

# include "utils.h"

/*****************************************************************************/
/* API
 *
 *
 * The function arma_rgig() and related function are copied from the source of
 * the R package GIGrvg. I did necessary modifications to modify the variable
 * type to arma::vec
 *
 *
 * Functions from GIGrvg: Added on March 12, 2017
 *  1. _gig_mode
 *  2. _rgig_ROU_noshift
 *  3. _rgig_newapproach1
 *  4. _rgig_ROU_shift_alt
 *  5. arma_rgig
 *
 */
/*****************************************************************************/

/*---------------------------------------------------------------------------*/
/* Prototypes of private functions                                           */
/*---------------------------------------------------------------------------*/

static double _gig_mode (double lambda, double omega);

/* Type 1 */
static void _rgig_ROU_noshift (arma::vec &res, int n, double lambda, double lambda_old, double omega, double alpha);

/* Type 4 */
static void _rgig_newapproach1 (arma::vec &res, int n, double lambda, double lambda_old, double omega, double alpha);

/* Type 8 */
static void _rgig_ROU_shift_alt (arma::vec &res, int n, double lambda, double lambda_old,  double omega, double alpha);

#define ZTOL (DOUBLE_EPS*10.0)

/*---------------------------------------------------------------------------*/

//' Draw sample from the Generalized Inverse Gaussian (GIG) distribution.
//'
//'
//'
//' @param n sample size (positive integer)
//' @param lambda parameter for distribution
//' @param chi parameter for distribution
//' @param psi parameter for distribution
//'
//' @return random sample of size 'n'
//'
//' @export
//'
//' @family utils
//'
//' @note Adapted from the R package GIGrvg.
//'
//' All rights reserved with the R packge authors:
//' Josef Leydold and Wolfgang Hörmann.
//'
//'
//'
// [[Rcpp::export]]
arma::vec arma_rgig (int n, double lambda, double chi, double psi)
{
  double omega, alpha;     /* parameters of standard distribution */
  vec res = zeros<vec>(n);
  // int i;

  /* check sample size */
  if (n<=0) {
    Rf_error("sample size 'n' must be positive integer.");
  }

  /* check GIG parameters: */
  if ( !(R_FINITE(lambda) && R_FINITE(chi) && R_FINITE(psi)) ||
  (chi <  0. || psi < 0)      ||
  (chi == 0. && lambda <= 0.) ||
  (psi == 0. && lambda >= 0.) ) {
    Rf_error("invalid parameters for GIG distribution: lambda=%g, chi=%g, psi=%g",
             lambda, chi, psi);
  }

  if (chi < ZTOL) {
    /* special cases which are basically Gamma and Inverse Gamma distribution */
    if (lambda > 0.0) {
      // for (i=0; i<n; i++) res(i) = rgamma(1, lambda, 2.0/psi)(0);
      res = rgamma(n, lambda, 2.0/psi);
    }
    else {
      // for (i=0; i<n; i++) res(i) = 1.0/rgamma(1, -lambda, 2.0/psi)(0);
      res = 1. / rgamma(n, -lambda, 2.0/psi);
    }
  }
  else if (psi < ZTOL) {
    /* special cases which are basically Gamma and Inverse Gamma distribution */
    if (lambda > 0.0) {
      // for (i=0; i<n; i++) res(i) = 1.0/rgamma(1, lambda, 2.0/chi)(0);
      res = 1. / rgamma(n, lambda, 2.0/chi);
    }
    else {
      // for (i=0; i<n; i++) res(i) = rgamma(1, -lambda, 2.0/chi)(0);
      res = rgamma(n, -lambda, 2.0/chi);
    }
  }
  else {
    double lambda_old = lambda;
    if (lambda < 0.) lambda = -lambda;
    alpha = sqrt(chi/psi);
    omega = sqrt(psi*chi);

    /* run generator */
    do {
      if (lambda > 2. || omega > 3.) {
        /* Ratio-of-uniforms with shift by 'mode', alternative implementation */
        _rgig_ROU_shift_alt(res, n, lambda, lambda_old, omega, alpha);
        break;
      }

      if (lambda >= 1.-2.25*omega*omega || omega > 0.2) {
        /* Ratio-of-uniforms without shift */
        _rgig_ROU_noshift(res, n, lambda, lambda_old, omega, alpha);
        break;
      }

      if (lambda >= 0. && omega > 0.) {
        /* New approach, constant hat in log-concave part. */
        _rgig_newapproach1(res, n, lambda, lambda_old, omega, alpha);
        break;
      }

      /* else */
      Rf_error("parameters must satisfy lambda>=0 and omega>0.");

    } while (0);
  }

  /* return result */
  return res;

} /* end of do_rgig() */

/*****************************************************************************/
/* Privat Functions                                                          */
/*****************************************************************************/

double _gig_mode (double lambda, double omega)
/*---------------------------------------------------------------------------*/
/* Compute mode of GIG distribution.                                         */
/*                                                                           */
/* Parameters:                                                               */
/*   lambda .. parameter for distribution                                    */
/*   omega ... parameter for distribution                                    */
/*                                                                           */
/* Return:                                                                   */
/*   mode                                                                    */
/*---------------------------------------------------------------------------*/
{
  if (lambda >= 1.)
    /* mode of fgig(x) */
    return (sqrt((lambda-1.)*(lambda-1.) + omega*omega)+(lambda-1.))/omega;
  else
    /* 0 <= lambda < 1: use mode of f(1/x) */
    return omega / (sqrt((1.-lambda)*(1.-lambda) + omega*omega)+(1.-lambda));
} /* end of _gig_mode() */

/*---------------------------------------------------------------------------*/

void _rgig_ROU_noshift (arma::vec &res, int n, double lambda, double lambda_old, double omega, double alpha)
/*---------------------------------------------------------------------------*/
/* Tpye 1:                                                                   */
/* Ratio-of-uniforms without shift.                                          */
/*   Dagpunar (1988), Sect.~4.6.2                                            */
/*   Lehner (1989)                                                           */
/*---------------------------------------------------------------------------*/
{
  double xm, nc;     /* location of mode; c=log(f(xm)) normalization constant */
  double ym, um;     /* location of maximum of x*sqrt(f(x)); umax of MBR */
  double s, t;       /* auxiliary variables */
  double U, V, X;    /* random variables */

  int i;             /* loop variable (number of generated random variables) */
  int count = 0;     /* counter for total number of iterations */

  /* -- Setup -------------------------------------------------------------- */

  /* shortcuts */
  t = 0.5 * (lambda-1.);
  s = 0.25 * omega;

  /* mode = location of maximum of sqrt(f(x)) */
  xm = _gig_mode(lambda, omega);

  /* normalization constant: c = log(sqrt(f(xm))) */
  nc = t*log(xm) - s*(xm + 1./xm);

  /* location of maximum of x*sqrt(f(x)):           */
  /* we need the positive root of                   */
  /*    omega/2*y^2 - (lambda+1)*y - omega/2 = 0    */
  ym = ((lambda+1.) + sqrt((lambda+1.)*(lambda+1.) + omega*omega))/omega;

  /* boundaries of minmal bounding rectangle:                   */
  /* we us the "normalized" density f(x) / f(xm). hence         */
  /* upper boundary: vmax = 1.                                  */
  /* left hand boundary: umin = 0.                              */
  /* right hand boundary: umax = ym * sqrt(f(ym)) / sqrt(f(xm)) */
  um = exp(0.5*(lambda+1.)*log(ym) - s*(ym + 1./ym) - nc);

  /* -- Generate sample ---------------------------------------------------- */

  for (i=0; i<n; i++) {
    do {
      ++count;
      U = um * unif_rand();        /* U(0,umax) */
  V = unif_rand();             /* U(0,vmax) */
  X = U/V;
    }                              /* Acceptance/Rejection */
  while (((log(V)) > (t*log(X) - s*(X + 1./X) - nc)));

    /* store random point */
    res(i) = (lambda_old < 0.) ? (alpha / X) : (alpha * X);
  }

  /* -- End ---------------------------------------------------------------- */

  return;
} /* end of _rgig_ROU_noshift() */


/*---------------------------------------------------------------------------*/

void _rgig_newapproach1 (arma::vec &res, int n, double lambda, double lambda_old, double omega, double alpha)
/*---------------------------------------------------------------------------*/
/* Type 4:                                                                   */
/* New approach, constant hat in log-concave part.                           */
/* Draw sample from GIG distribution.                                        */
/*                                                                           */
/* Case: 0 < lambda < 1, 0 < omega < 1                                       */
/*                                                                           */
/* Parameters:                                                               */
/*   n ....... sample size (positive integer)                                */
/*   lambda .. parameter for distribution                                    */
/*   omega ... parameter for distribution                                    */
/*                                                                           */
/* Return:                                                                   */
/*   random sample of size 'n'                                               */
/*---------------------------------------------------------------------------*/
{
	/* parameters for hat function */
	double A[3], Atot;  /* area below hat */
	double k0;          /* maximum of PDF */
	double k1, k2;      /* multiplicative constant */

	double xm;          /* location of mode */
	double x0;          /* splitting point T-concave / T-convex */
	double a;           /* auxiliary variable */

	double U, V, X;     /* random numbers */
	double hx;          /* hat at X */

	int i;              /* loop variable (number of generated random variables) */
	int count = 0;      /* counter for total number of iterations */

	/* -- Check arguments ---------------------------------------------------- */

	if (lambda >= 1. || omega >1.)
	  Rf_error ("invalid parameters");

	/* -- Setup -------------------------------------------------------------- */

	/* mode = location of maximum of sqrt(f(x)) */
	xm = _gig_mode(lambda, omega);

	/* splitting point */
	x0 = omega/(1.-lambda);

	/* domain [0, x_0] */
	k0 = exp((lambda-1.)*log(xm) - 0.5*omega*(xm + 1./xm));     /* = f(xm) */
	A[0] = k0 * x0;

	/* domain [x_0, Infinity] */
	if (x0 >= 2./omega) {
	  k1 = 0.;
	  A[1] = 0.;
	  k2 = pow(x0, lambda-1.);
	  A[2] = k2 * 2. * exp(-omega*x0/2.)/omega;
	}

	else {
	  /* domain [x_0, 2/omega] */
	  k1 = exp(-omega);
	  A[1] = (lambda == 0.)
	    ? k1 * log(2./(omega*omega))
	      : k1 / lambda * ( pow(2./omega, lambda) - pow(x0, lambda) );

	  /* domain [2/omega, Infinity] */
	  k2 = pow(2/omega, lambda-1.);
	  A[2] = k2 * 2 * exp(-1.)/omega;
	}

	/* total area */
	Atot = A[0] + A[1] + A[2];

	/* -- Generate sample ---------------------------------------------------- */

	for (i=0; i<n; i++) {
	  do {
	    ++count;

	    /* get uniform random number */
	    V = Atot * unif_rand();

	    do {

	      /* domain [0, x_0] */
	      if (V <= A[0]) {
	        X = x0 * V / A[0];
	        hx = k0;
	        break;
	      }

	      /* domain [x_0, 2/omega] */
	      V -= A[0];
	      if (V <= A[1]) {
	        if (lambda == 0.) {
	          X = omega * exp(exp(omega)*V);
	          hx = k1 / X;
	        }
	        else {
	          X = pow(pow(x0, lambda) + (lambda / k1 * V), 1./lambda);
	          hx = k1 * pow(X, lambda-1.);
	        }
	        break;
	      }

	      /* domain [max(x0,2/omega), Infinity] */
	      V -= A[1];
	      a = (x0 > 2./omega) ? x0 : 2./omega;
	      X = -2./omega * log(exp(-omega/2. * a) - omega/(2.*k2) * V);
	      hx = k2 * exp(-omega/2. * X);
	      break;

	    } while(0);

	    /* accept or reject */
	    U = unif_rand() * hx;

	    if (log(U) <= (lambda-1.) * log(X) - omega/2. * (X+1./X)) {
	      /* store random point */
	      res[i] = (lambda_old < 0.) ? (alpha / X) : (alpha * X);
	      break;
	    }
	  } while(1);

	}

    /* -- End ---------------------------------------------------------------- */

    return;
} /* end of _rgig_newapproach1() */

/*---------------------------------------------------------------------------*/

void _rgig_ROU_shift_alt (arma::vec &res, int n, double lambda, double lambda_old, double omega, double alpha)
/*---------------------------------------------------------------------------*/
/* Type 8:                                                                   */
/* Ratio-of-uniforms with shift by 'mode', alternative implementation.       */
/*   Dagpunar (1989)                                                         */
/*   Lehner (1989)                                                           */
/*---------------------------------------------------------------------------*/
{
  double xm, nc;     /* location of mode; c=log(f(xm)) normalization constant */
  double s, t;       /* auxiliary variables */
  double U, V, X;    /* random variables */

  int i;             /* loop variable (number of generated random variables) */
  int count = 0;     /* counter for total number of iterations */

  double a, b, c;    /* coefficent of cubic */
  double p, q;       /* coefficents of depressed cubic */
  double fi, fak;    /* auxiliary results for Cardano's rule */

  double y1, y2;     /* roots of (1/x)*sqrt(f((1/x)+m)) */

  double uplus, uminus;  /* maximum and minimum of x*sqrt(f(x+m)) */

  /* -- Setup -------------------------------------------------------------- */

  /* shortcuts */
  t = 0.5 * (lambda-1.);
  s = 0.25 * omega;

  /* mode = location of maximum of sqrt(f(x)) */
  xm = _gig_mode(lambda, omega);

  /* normalization constant: c = log(sqrt(f(xm))) */
  nc = t*log(xm) - s*(xm + 1./xm);

  /* location of minimum and maximum of (1/x)*sqrt(f(1/x+m)):  */

  /* compute coeffients of cubic equation y^3+a*y^2+b*y+c=0 */
  a = -(2.*(lambda+1.)/omega + xm);       /* < 0 */
  b = (2.*(lambda-1.)*xm/omega - 1.);
  c = xm;

  /* we need the roots in (0,xm) and (xm,inf) */

  /* substitute y=z-a/3 for depressed cubic equation z^3+p*z+q=0 */
  p = b - a*a/3.;
  q = (2.*a*a*a)/27. - (a*b)/3. + c;

  /* use Cardano's rule */
  fi = acos(-q/(2.*sqrt(-(p*p*p)/27.)));
  fak = 2.*sqrt(-p/3.);
  y1 = fak * cos(fi/3.) - a/3.;
  y2 = fak * cos(fi/3. + 4./3.*M_PI) - a/3.;

  /* boundaries of minmal bounding rectangle:                  */
  /* we us the "normalized" density f(x) / f(xm). hence        */
  /* upper boundary: vmax = 1.                                 */
  /* left hand boundary: uminus = (y2-xm) * sqrt(f(y2)) / sqrt(f(xm)) */
  /* right hand boundary: uplus = (y1-xm) * sqrt(f(y1)) / sqrt(f(xm)) */
  uplus  = (y1-xm) * exp(t*log(y1) - s*(y1 + 1./y1) - nc);
  uminus = (y2-xm) * exp(t*log(y2) - s*(y2 + 1./y2) - nc);

  /* -- Generate sample ---------------------------------------------------- */

  for (i=0; i<n; i++) {
    do {
      	++count;
      	U = uminus + unif_rand() * (uplus - uminus);    /* U(u-,u+)  */
  		V = unif_rand();                                /* U(0,vmax) */
  		X = U/V + xm;
    }                                         /* Acceptance/Rejection */
  	while ((X <= 0.) || ((log(V)) > (t*log(X) - s*(X + 1./X) - nc)));

    /* store random point */
    res(i) = (lambda_old < 0.) ? (alpha / X) : (alpha * X);
  }

  /* -- End ---------------------------------------------------------------- */

  return;
} /* end of _rgig_ROU_shift_alt() */

/*---------------------------end of GIGrvg functions ------------------------*/



//' Samples from the Antoniak distribution
//'
//' It's done by sampling \eqn{N} Bernoulli variables
//'
//' References:
//'
//'   http://www.jmlr.org/papers/volume10/newman09a/newman09a.pdf
//'
//' @param N Number of samples
//' @param alpha strength parameter
//'
//' @export
//'
//' @family utils
//'
//' @note
//'
//' Created on: May 19, 2016
//'
//' Created by: Clint P. George
//'
// [[Rcpp::export]]
double sample_antoniak (unsigned int N, double alpha){
  vec bs = zeros<vec>(N);
  for (unsigned int l = 0; l < N; l++){
    bs(l) = rbinom(1, 1, (alpha / (alpha + l)))(0);
  }
  return sum(bs);
}

/**
* Samples an integer from [0, K) uniformly at random
*
* Arguments:
* 		K - the upper interval
* Returns:
* 		the sampled integer
*/
// [[Rcpp::export]]
unsigned int sample_uniform_int (unsigned int K){
  return (unsigned int) (runif(1)(0) * (double)K); // To speedup
}

//' A speedy sampling from a multimomial distribution
//'
//' @param theta a multinomial probability vector (K x 1 vector)
//'
//' @return returns a class index from [0, K)
//'
//' @note
//' Author: Clint P. George
//'
//' Created on: February 11, 2016
//'
//' @family utils
//'
//' @export
// [[Rcpp::export]]
unsigned int sample_multinomial (arma::vec theta) {

  unsigned int t = 0;
  double total_prob = accu(theta);
  double u = runif(1)(0) * total_prob;
  double cumulative_prob = theta(0);

  while(u > cumulative_prob){
    t++;
    cumulative_prob += theta(t);
  }

  return t;

}


//' Samples from a Dirichlet distribution given a hyperparameter
//'
//' @param num_elements the dimention of the Dirichlet distribution
//' @param alpha the hyperparameter vector (a column vector)
//'
//' @return returns a Dirichlet sample (a column vector)
//'
//' @note
//' Author: Clint P. George
//'
//' Created on: 2014
//'
//' @family utils
//'
//' @export
// [[Rcpp::export]]
arma::vec sample_dirichlet (unsigned int num_elements, arma::vec alpha){

  arma::vec dirichlet_sample = arma::zeros<arma::vec>(num_elements);

  for ( register unsigned int i = 0; i < num_elements; i++ )
    dirichlet_sample(i) = rgamma(1, alpha(i), 1.0)(0); // R::rgamma(1, alpha(i));

  dirichlet_sample /= accu(dirichlet_sample);

  return dirichlet_sample;

}

/**
* Samples from a Dirichlet distribution given a hyperparameter
*
* Aruguments:
* 		num_elements - the dimention of the Dirichlet distribution
* 		alpha - the hyperparameter vector (a column vector)
* Returns:
* 		the Dirichlet sample (a column vector)
*/
arma::rowvec sample_dirichlet_row_vec (unsigned int num_elements, arma::rowvec alpha){

  arma::rowvec dirichlet_sample = arma::zeros<arma::rowvec>(num_elements);

  for ( register unsigned int i = 0; i < num_elements; i++ )
    dirichlet_sample(i) = rgamma(1, alpha(i), 1.0)(0); // R::rgamma(1, alpha(i));

  dirichlet_sample /= accu(dirichlet_sample);

  return dirichlet_sample;

}

/**
* Samples random permutations for a given count
*
* Arguments:
* 		n - the number of samples
* Return:
* 		order - a vector of indices that represents
* 				the permutations of numbers in [1, n]
**/
arma::uvec randperm (unsigned int n) {
  arma::uvec order = arma::zeros<arma::uvec>(n);
  unsigned int k, nn, takeanumber, temp;
  for (k=0; k<n; k++) order(k) = k;
  nn = n;
  for (k=0; k<n; k++) {
    takeanumber = sample_uniform_int(nn); // take a number between 0 and nn-1
    temp = order(nn-1);
    order(nn-1) = order(takeanumber);
    order(takeanumber) = temp;
    nn--;
  }
  return order;
}


arma::vec log_gamma_vec (arma::vec x_vec) {
  arma::vec lgamma_vec = arma::zeros<arma::vec>(x_vec.n_elem);
  for (unsigned int i = 0; i < x_vec.n_elem; i++)
    lgamma_vec(i) = lgamma(x_vec(i));
  return lgamma_vec;
}

arma::rowvec log_gamma_rowvec (arma::rowvec x_vec) {
  arma::rowvec lgamma_rowvec = arma::zeros<arma::rowvec>(x_vec.n_elem);
  for (unsigned int i = 0; i < x_vec.n_elem; i++)
    lgamma_rowvec(i) = lgamma(x_vec(i));
  return lgamma_rowvec;
}

arma::vec digamma_vec (arma::vec x_vec) {
  // digamma(wrap()) will do, with comparable performance
  arma::vec ret = arma::zeros<arma::vec>(x_vec.n_elem);
  for (unsigned int i = 0; i < x_vec.n_elem; i++)
    ret(i) = Rf_digamma(x_vec(i));
  return ret;
}

arma::rowvec digamma_rowvec (arma::rowvec x_vec) {
  arma::rowvec ret = arma::zeros<arma::rowvec>(x_vec.n_elem);
  for (unsigned int i = 0; i < x_vec.n_elem; i++)
    ret(i) = Rf_digamma(x_vec(i));
  return ret;
}

arma::vec trigamma_vec (arma::vec x_vec) {
  arma::vec ret = arma::zeros<arma::vec>(x_vec.n_elem);
  for (unsigned int i = 0; i < x_vec.n_elem; i++)
    ret(i) = Rf_trigamma(x_vec(i));
  return ret;
}

arma::vec tetragamma_vec (arma::vec x_vec) {
  arma::vec ret = arma::zeros<arma::vec>(x_vec.n_elem);
  for (unsigned int i = 0; i < x_vec.n_elem; i++)
    ret(i) = Rf_tetragamma(x_vec(i));
  return ret;
}

arma::vec gamma_col_vec (arma::vec x_vec){
  // It took 2hrs of my time in the April 19, 2014 morning to make this function
  // work. The main issue was with accessing the R gamma function from the
  // RcppArmadillo namespace. See
  // http://dirk.eddelbuettel.com/code/rcpp/html/Rmath_8h_source.html
  // gamma(as<NumericVector>(wrap(x_vec))) is another option, but it seems to be
  // slow. See
  // http://stackoverflow.com/questions/14253069/convert-rcpparmadillo-vector-to-rcpp-vector

  arma::vec gamma_vec = arma::zeros<arma::vec>(x_vec.n_elem);

  for (unsigned int i = 0; i < x_vec.n_elem; i++)
    gamma_vec(i) = Rf_gammafn(x_vec(i));

  return gamma_vec;
}

//' Generating a multivariate gaussian distribution using RcppArmadillo
//'
//' There are many ways to simulate a multivariate gaussian distribution
//' assuming that you can simulate from independent univariate normal
//' distributions. One of the most popular method is based on the Cholesky
//' decomposition.
//'
//' @param n - the number of samples
//' @param mu - the mean vector
//' @param sigma - the covariace matrix
//'
//' @note
//' Author: Ahmadou Dicko — written Mar 12, 2013
//' License: GPL
//'
//' Reference:
//' http://gallery.rcpp.org/articles/simulate-multivariate-normal/
//'
//' @export
// [[Rcpp::export]]
arma::mat arma_mvrnorm(int n, arma::vec mu, arma::mat sigma) {
  int ncols = sigma.n_cols;
  arma::mat Y = arma::randn(n, ncols);
  return arma::repmat(mu, 1, n).t() + Y * arma::chol(sigma);
}
