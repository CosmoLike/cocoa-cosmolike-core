#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <gsl/gsl_math.h>
#include "utils_cfastpt.h"

static inline double factorial(int n) {	
	static long FACTORIAL_LIST[] = {1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880,\
									 								3628800, 39916800, 479001600, 6227020800, 87178291200};
  if (n<0) {
  	printf("factorial(n): n=%d \n",n);
  	exit(1);
  }
	if (n > 14) {
		return tgamma(n + 1.0);
	}
	return FACTORIAL_LIST[n];
}

static inline double wigner_3j_jjj_000(int j1, int j2, int j3) {
  int J = j1 + j2 + j3;
  if (J % 2 != 0) return 0.0;

  int ab_c = j1 + j2 - j3;
  int ac_b = j1 + j3 - j2;
  int bc_a = j2 + j3 - j1;
  if (ab_c < 0 || ac_b < 0 || bc_a < 0) return 0.0;

  int halfJ = J / 2;
  int sign = (halfJ % 2 ? -1 : 1);

  double log_result =
    0.5 * (lgamma(ab_c + 1) + lgamma(ac_b + 1) + lgamma(bc_a + 1) - lgamma(J + 2))
    + lgamma(halfJ + 1) - lgamma(halfJ - j1 + 1) - lgamma(halfJ - j2 + 1) - lgamma(halfJ - j3 + 1);

  return sign * exp(log_result);
}

static inline double wigner_6j(int j1, int j2, int j3, int j4, int j5, int j6) {
  int a = j1, b = j2, c = j5, d = j4, e = j3, f = j6;

  if (a+b-e < 0 || a+e-b < 0 || b+e-a < 0) return 0.0;
  if (c+d-e < 0 || c+e-d < 0 || d+e-c < 0) return 0.0;
  if (a+c-f < 0 || a+f-c < 0 || c+f-a < 0) return 0.0;
  if (b+d-f < 0 || b+f-d < 0 || d+f-b < 0) return 0.0;

  double log_pf =
    0.5 * (lgamma(a+b-e+1) + lgamma(a+e-b+1) + lgamma(b+e-a+1) - lgamma(a+b+e+2)
         + lgamma(c+d-e+1) + lgamma(c+e-d+1) + lgamma(d+e-c+1) - lgamma(c+d+e+2)
         + lgamma(a+c-f+1) + lgamma(a+f-c+1) + lgamma(c+f-a+1) - lgamma(a+c+f+2)
         + lgamma(b+d-f+1) + lgamma(b+f-d+1) + lgamma(d+f-b+1) - lgamma(b+d+f+2));

  int imin = (a+b+e > c+d+e) ? a+b+e : c+d+e;
  if (a+c+f > imin) imin = a+c+f;
  if (b+d+f > imin) imin = b+d+f;

  int imax = (a+b+c+d < a+d+e+f) ? a+b+c+d : a+d+e+f;
  if (b+c+e+f < imax) imax = b+c+e+f;

  double sum = 0.0;
  for (int i = imin; i <= imax; i++) {
    double log_term = lgamma(i+2)
      - lgamma(i-a-b-e+1) - lgamma(i-c-d-e+1)
      - lgamma(i-a-c-f+1) - lgamma(i-b-d-f+1)
      - lgamma(a+b+c+d-i+1) - lgamma(a+d+e+f-i+1)
      - lgamma(b+c+e+f-i+1);
    int sign = (i % 2 ? -1 : 1);
    sum += sign * exp(log_pf + log_term);
  }
  return sum;
}

int J_table(int Ncols, int Nterms, int (*terms)[Ncols], double *coeff_A,
                int (*out)[Ncols], double *coeff_out) {
  int row = 0;
  for (int i = 0; i < Nterms; i++) {
    const int alpha = terms[i][0];
    const int beta  = terms[i][1];
    const int l1    = terms[i][2];
    const int l2    = terms[i][3];
    const int l     = terms[i][4];
    const double cA = coeff_A[i];
 
    // coeff_B requires four parity conditions:
    //   (1) (J1+l2+l) even  (2) (l1+J2+l) even
    //   (3) (l1+l2+Jk) even (4) (J1+J2+Jk) even
    //
    // Conditions 1-3 each constrain a single loop variable.
    // |l-l2| has the same parity as l+l2, so J1=|l-l2| satisfies (1),
    // and stepping by 2 preserves it. Same logic for J2 and Jk.
    // Condition (4) is always satisfied when (1)-(3) hold.
 
    for (int J1 = abs(l-l2); J1 <= l+l2; J1 += 2) {       // satisfies (1)
      double w1 = wigner_3j_jjj_000(J1, l2, l);
      if (w1 == 0.0) continue;
 
      for (int J2 = abs(l1-l); J2 <= l1+l; J2 += 2) {     // satisfies (2)
        double w2 = wigner_3j_jjj_000(l1, J2, l);
        if (w2 == 0.0) continue;
 
        // tighten Jk bounds with triangle inequality from w3j(J1,J2,Jk)
        int Jk_min = abs(l1-l2);
        if (abs(J1-J2) > Jk_min) Jk_min = abs(J1-J2);
        if ((Jk_min + l1 + l2) % 2 != 0) Jk_min++;        // fix parity
        int Jk_max = l1+l2;
        if (J1+J2 < Jk_max) Jk_max = J1+J2;
 
        for (int Jk = Jk_min; Jk <= Jk_max; Jk += 2) {    // satisfies (3)
          double w3 = wigner_3j_jjj_000(l1, l2, Jk);
          if (w3 == 0.0) continue;
          double w4 = wigner_3j_jjj_000(J1, J2, Jk);
          if (w4 == 0.0) continue;
          double w6 = wigner_6j(J1, J2, Jk, l1, l2, l);
          if (w6 == 0.0) continue;
 
          int sign = ((l + (J1+J2+Jk) / 2) % 2 ? -1 : 1);
          double pf = sign * (2*J1+1) * (2*J2+1) * (2*Jk+1) / (M_PI * M_PI * M_PI);
 
          // copy all input columns, then overwrite l1,l2,l with J1,J2,Jk
          for (int c = 0; c < Ncols; c++) out[row][c] = terms[i][c];
          out[row][2] = J1;
          out[row][3] = J2;
          out[row][4] = Jk;
          coeff_out[row] = cA * pf * w1 * w2 * w3 * w4 * w6;
          row++;
        }
      }
    }
  }
  if (row == 0) {
    printf("J_table empty! Check input coefficients!\n");
    exit(1);
  }
  return row;
}
