#ifndef COSMOLIKE_GAUSSIAN_COV_H
#define COSMOLIKE_GAUSSIAN_COV_H

#ifdef __cplusplus
extern "C" {
#endif

// Real-space estimators. Source ellipticity dispersions are per component.
enum probe_cov {
  XI_PLUS_COV = 0,  // <gamma_t gamma_t + gamma_x gamma_x>
  XI_MINUS_COV = 1, // <gamma_t gamma_t - gamma_x gamma_x>
  GAMMA_T_COV = 2,  // <delta_g gamma_t>, lens field first
  W_THETA_COV = 3   // <delta_g delta_g>
};

// These are stateless numerical building blocks, not a survey driver.
// See the definitions in gaussian_cov.c for equations and full contracts.
// Arrays use row pointers, compatible with the padded house malloc2d.
// The caller owns every array and must supply finite spectra and kernels.

void gaussian_wick_cov(
    const int ell_min,                  // first integer multipole, >= 0
    const int nell,                     // number of consecutive multipoles
    const double fsky,                  // survey area / (4 pi), in (0, 1]
    const double* const* cross_spectra, // [4][nell]: AC, BD, AD, BC signal
    const double* cross_noise,          // [4]: AC, BD, AD, BC white noise
    const int include_noise_noise,      // 0: real-space split; 1: full harmonic
    double* gaussian                    // output [nell], covariance at each ell
  );

void gaussian_project_cov(
    const int nleft,                    // number of rows of the left operator
    const int nright,                   // number of rows of the right operator
    const int nell,                     // shared number of multipoles
    const double* const* kernel_left,   // [nleft][nell], includes normalization
    const double* const* kernel_right,  // [nright][nell], normalized operator
    const double* gaussian,             // [nell], harmonic covariance
    double* const* weighted_left,       // scratch [nleft][nell], caller owned
    double* const* covariance           // output [nleft][nright], overwritten
  );

double annulus_pair_area_cov(
    const double area_sr,               // survey solid angle, in steradians
    const double theta_low_rad,         // lower separation, in radians
    const double theta_high_rad         // upper separation, in radians
  );

double gaussian_noise_pair_cov(
    const enum probe_cov probe_left,    // estimator of the left block
    const enum probe_cov probe_right,   // estimator of the right block
    const int* fields,                  // [4]: A, B, C, D, global field IDs
    const double* noise_ab,             // [2]: N_A and N_B, nonnegative
    const double pair_area_sr2          // ordered-pair angular area, in sr^2
  );

#ifdef __cplusplus
}
#endif
#endif
