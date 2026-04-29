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

// ---------------------------------------------------------------------------
// wigner_3j_jjj_000: Special case of the Wigner 3-j symbol with all m = 0:
//
//   ( j1  j2  j3 )
//   (  0   0   0 )
//
// Selection rules (return 0 if violated):
//   1. J = j1 + j2 + j3 must be even (parity conservation)
//   2. Triangle inequality: each pair sum must be >= the third
//      i.e., |j1-j2| <= j3 <= j1+j2 (and cyclic permutations)
//
// When the selection rules are satisfied, the closed-form expression is:
//   (-1)^(J/2) * Delta(j1,j2,j3) * (J/2)! / ((J/2-j1)! * (J/2-j2)! * (J/2-j3)!)
//
// where Delta(a,b,c) = sqrt((a+b-c)! * (a+c-b)! * (b+c-a)! / (a+b+c+1)!)
// is the triangle coefficient.
//
// OPTIMIZATION: The Delta coefficient is inlined (not a separate function
// call) and the entire computation is done in log-space using lgamma to
// avoid factorial overflow. A single exp() at the end converts back.
// This allows arbitrarily large angular momenta without a lookup table.
//
// Parameters:
//   j1, j2, j3:  integer angular momentum quantum numbers (>= 0)
// Returns:
//   The Wigner 3-j symbol value, or 0.0 if selection rules are violated
// ---------------------------------------------------------------------------
static inline double wigner_3j_jjj_000(int j1, int j2, int j3) {
  // Selection rule 1: J = j1 + j2 + j3 must be even
  int J = j1 + j2 + j3;
  if (J % 2 != 0) return 0.0;

  // Selection rule 2: triangle inequality
  // ab_c, ac_b, bc_a must all be >= 0 for a valid angular momentum coupling
  int ab_c = j1 + j2 - j3; // = J - 2*j3
  int ac_b = j1 + j3 - j2; // = J - 2*j2
  int bc_a = j2 + j3 - j1; // = J - 2*j1
  if (ab_c < 0 || ac_b < 0 || bc_a < 0) return 0.0;

  int halfJ = J / 2;

  int sign = (halfJ % 2 ? -1 : 1);  // Overall sign: (-1)^(J/2)

  // Log-space computation to avoid factorial overflow:
  // log(result) = log(Delta) + log((J/2)!) - log((J/2-j1)!) 
  //               - log((J/2-j2)!) - log((J/2-j3)!)
  // where log(Delta) = 0.5 * [lgamma(ab_c+1) + lgamma(ac_b+1) + lgamma(bc_a+1) - lgamma(J+2)]
  // using lgamma(n+1) = log(n!)
   double log_delta = 0.5 * (lgamma(ab_c + 1) + 
                             lgamma(ac_b + 1) + 
                             lgamma(bc_a + 1) - 
                             lgamma(J + 2));

  double log_result = log_delta + 
                      lgamma(halfJ + 1) - 
                      lgamma(halfJ - j1 + 1) - 
                      lgamma(halfJ - j2 + 1) - 
                      lgamma(halfJ - j3 + 1);

  return sign * exp(log_result);
}

// ---------------------------------------------------------------------------
// wigner_6j: Compute the Wigner 6-j symbol:
//
//   { j1  j2  j3 }
//   { j4  j5  j6 }
//
// The 6-j symbol arises in the recoupling of three angular momenta and
// appears in the coeff_B function when decomposing the FAST-PT tensor
// integrals into products of Wigner 3-j symbols.
//
// Internally this computes the Racah W-coefficient via:
//   {j1 j2 j3}
//   {j4 j5 j6} = (-1)^(j1+j2+j4+j5) * W(j1, j2, j5, j4; j3, j6)
//
// but the two sign factors (from wigner_6j and Racah) cancel:
//   (-1)^(j1+j2+j4+j5) * (-1)^(a+b+c+d) = (-1)^(2*(j1+j2+j4+j5)) = 1
// so the result is simply the Racah sum without any overall sign.
//
// The Racah formula uses remapped variables for compactness:
//   a=j1, b=j2, c=j5, d=j4, e=j3, f=j6
//
// SELECTION RULES (return 0 if any violated):
//   Four triangle inequalities must be satisfied:
//     triangle(a, b, e)  i.e. triangle(j1, j2, j3)
//     triangle(c, d, e)  i.e. triangle(j5, j4, j3)
//     triangle(a, c, f)  i.e. triangle(j1, j5, j6)
//     triangle(b, d, f)  i.e. triangle(j2, j4, j6)
//
// COMPUTATION:
//   The prefactor is the product of four Delta (triangle) coefficients,
//   computed in log-space to avoid overflow. The Racah sum runs over
//   integer i from imin to imax, where each term is a ratio of factorials
//   computed in log-space and combined with the log-prefactor before
//   exponentiating. This avoids overflow for large angular momenta.
//
// Parameters:
//   j1..j6:  integer angular momentum quantum numbers (>= 0)
// Returns:
//   The Wigner 6-j symbol value, or 0.0 if selection rules are violated
// ---------------------------------------------------------------------------
static inline double wigner_6j(int j1, int j2, int j3, int j4, int j5, int j6) {
  // Remap to Racah variables: W(a, b, c, d; e, f) = W(j1, j2, j5, j4; j3, j6)
  int a = j1, b = j2, c = j5, d = j4, e = j3, f = j6;

  // Triangle inequality checks for the four triads (a,b,e), (c,d,e), (a,c,f), (b,d,f)
  if (a+b-e < 0 || a+e-b < 0 || b+e-a < 0) return 0.0;
  if (c+d-e < 0 || c+e-d < 0 || d+e-c < 0) return 0.0;
  if (a+c-f < 0 || a+f-c < 0 || c+f-a < 0) return 0.0;
  if (b+d-f < 0 || b+f-d < 0 || d+f-b < 0) return 0.0;

  // Log of the product of four Delta (triangle) coefficients:
  //   log_pf = log(Delta(a,b,e) * Delta(c,d,e) * Delta(a,c,f) * Delta(b,d,f))
  // where Delta(x,y,z) = sqrt((x+y-z)! * (x+z-y)! * (y+z-x)! / (x+y+z+1)!)
  // The 0.5 handles the square roots; lgamma(n+1) = log(n!).
  double log_pf =
    0.5 * (lgamma(a+b-e+1) + lgamma(a+e-b+1) + lgamma(b+e-a+1) - lgamma(a+b+e+2)
         + lgamma(c+d-e+1) + lgamma(c+e-d+1) + lgamma(d+e-c+1) - lgamma(c+d+e+2)
         + lgamma(a+c-f+1) + lgamma(a+f-c+1) + lgamma(c+f-a+1) - lgamma(a+c+f+2)
         + lgamma(b+d-f+1) + lgamma(b+f-d+1) + lgamma(d+f-b+1) - lgamma(b+d+f+2));

  // Racah sum bounds:
  //   imin = max(a+b+e, c+d+e, a+c+f, b+d+f)
  //   imax = min(a+b+c+d, a+d+e+f, b+c+e+f)
  int imin = (a+b+e > c+d+e) ? a+b+e : c+d+e;
  if (a+c+f > imin) imin = a+c+f;
  if (b+d+f > imin) imin = b+d+f;

  int imax = (a+b+c+d < a+d+e+f) ? a+b+c+d : a+d+e+f;
  if (b+c+e+f < imax) imax = b+c+e+f;

  // Racah sum: each term is (-1)^i * (i+1)! / (product of 7 factorials),
  // computed in log-space and combined with log_pf before exponentiating.
  // Folding log_pf into each term avoids separate overflow/underflow of
  // exp(log_pf) when the prefactor alone is very large or very small.
  double sum = 0.0;
  for (int i = imin; i <= imax; i++) {
    // log of the Racah sum term (without sign or prefactor):
    //   log((i+1)!) - log((i-a-b-e)!) - log((i-c-d-e)!) - log((i-a-c-f)!)
    //              - log((i-b-d-f)!) - log((a+b+c+d-i)!) - log((a+d+e+f-i)!)
    //              - log((b+c+e+f-i)!)
    double log_term = lgamma(i+2)
      - lgamma(i-a-b-e+1) - lgamma(i-c-d-e+1)
      - lgamma(i-a-c-f+1) - lgamma(i-b-d-f+1)
      - lgamma(a+b+c+d-i+1) - lgamma(a+d+e+f-i+1)
      - lgamma(b+c+e+f-i+1);
    int sign = (i % 2 ? -1 : 1);
    sum += sign * exp(log_pf + log_term);
  }
  // log of the Racah sum term (without sign or prefactor):
  //   log((i+1)!) - log((i-a-b-e)!) - log((i-c-d-e)!) - log((i-a-c-f)!)
  //              - log((i-b-d-f)!) - log((a+b+c+d-i)!) - log((a+d+e+f-i)!)
  //              - log((b+c+e+f-i)!)
  return sum;
}

// ---------------------------------------------------------------------------
// J_table: Expand input terms {alpha, beta, l1, l2, l} into output terms
// {alpha, beta, J1, J2, Jk} by computing the coeff_B coupling coefficients.
//
// For each input term i with angular momenta (l1, l2, l) and coefficient
// coeff_A[i], this function loops over all valid (J1, J2, Jk) combinations
// and computes the coupling coefficient:
//
//   coeff_B = (-1)^(l + (J1+J2+Jk)/2) * (2*J1+1)*(2*J2+1)*(2*Jk+1) / pi^3
//             * w3j(J1,l2,l) * w3j(l1,J2,l) * w3j(l1,l2,Jk)
//             * w3j(J1,J2,Jk) * w6j(J1,J2,Jk,l1,l2,l)
//
// where w3j are Wigner 3-j symbols (all m=0) and w6j is the Wigner 6-j
// symbol. Only terms with nonzero coeff_B are stored in the output.
//
// The combined coefficient for each output row is: coeff_A[i] * coeff_B.
//
// LOOP OPTIMIZATIONS:
//   - J1, J2, Jk loops step by 2 (not 1), exploiting the parity selection
//     rules from coeff_B. The four parity conditions are:
//       (1) (J1+l2+l) even → J1 loop starts at |l-l2|, steps by 2
//       (2) (l1+J2+l) even → J2 loop starts at |l1-l|, steps by 2
//       (3) (l1+l2+Jk) even → Jk loop starts at |l1-l2|, steps by 2
//       (4) (J1+J2+Jk) even → automatically satisfied when (1)-(3) hold
//     This cuts the iteration count by ~8x vs stepping by 1.
//
//   - Jk bounds are tightened using the triangle inequality from
//     w3j(J1,J2,Jk): Jk must be in [|J1-J2|, J1+J2] as well as
//     [|l1-l2|, l1+l2]. The intersection gives fewer iterations.
//
//   - Wigner symbols are evaluated in order of increasing cost, with
//     early exits when any symbol is zero. w1 (depends only on J1) is
//     hoisted to the J1 loop, w2 (depends on J1, J2) to the J2 loop.
//
// INPUT LAYOUT: terms[i] = {alpha, beta, l1, l2, l} (Ncols columns)
// OUTPUT LAYOUT: out[row] = {alpha, beta, J1, J2, Jk} (same Ncols columns)
//   Columns 0-1 (alpha, beta) are copied through unchanged.
//   Columns 2-4 are overwritten: l1 → J1, l2 → J2, l → Jk.
//
// Parameters:
//   Ncols:     number of columns per row (always NCOLS = 5)
//   Nterms:    number of input terms (rows in the terms array)
//   terms:     input array of shape [Nterms][Ncols]
//   coeff_A:   input coefficients, length Nterms
//   out:       output array of shape [Nmax][Ncols] (caller-allocated)
//   coeff_out: output coefficients, length Nmax (caller-allocated)
// Returns:
//   Number of output rows written (always > 0, exits on error if 0)
// ---------------------------------------------------------------------------
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
  
    // J1 loop: couples (l, l2). Starts at |l-l2|, steps by 2 to satisfy
    // parity condition (1): (J1+l2+l) must be even.
    // w1 = w3j(J1, l2, l) is evaluated here and reused in inner loops.
    for (int J1 = abs(l-l2); J1 <= l+l2; J1 += 2) {
      double w1 = wigner_3j_jjj_000(J1, l2, l);
      if (w1 == 0.0) continue;
 
      // J2 loop: couples (l1, l). Starts at |l1-l|, steps by 2 to satisfy
      // parity condition (2): (l1+J2+l) must be even.
      // w2 = w3j(l1, J2, l) is evaluated here and reused in the Jk loop.
      for (int J2 = abs(l1-l); J2 <= l1+l; J2 += 2) 
      {     // satisfies (2)
        double w2 = wigner_3j_jjj_000(l1, J2, l);
        if (w2 == 0.0) continue;
 
        // Jk loop: couples (l1, l2). Bounds tightened by intersecting
        // the triangle inequality from w3j(l1, l2, Jk) with that from
        // w3j(J1, J2, Jk): Jk must satisfy both |l1-l2| <= Jk <= l1+l2
        // and |J1-J2| <= Jk <= J1+J2.
        int Jk_min = abs(l1-l2);
        if (abs(J1-J2) > Jk_min) Jk_min = abs(J1-J2);
        // Fix parity if the tightened lower bound broke condition (3): (l1+l2+Jk) must be even.
        if ((Jk_min + l1 + l2) % 2 != 0) Jk_min++;
        int Jk_max = l1+l2;
        if (J1+J2 < Jk_max) Jk_max = J1+J2;
 
        for (int Jk = Jk_min; Jk <= Jk_max; Jk += 2) {
          // Evaluate remaining Wigner symbols with early exit on zero
          double w3 = wigner_3j_jjj_000(l1, l2, Jk);
          if (w3 == 0.0) continue;
          double w4 = wigner_3j_jjj_000(J1, J2, Jk);
          if (w4 == 0.0) continue;
          double w6 = wigner_6j(J1, J2, Jk, l1, l2, l);
          if (w6 == 0.0) continue;
 
          // Sign factor: (-1)^(l + (J1+J2+Jk)/2)
          // Note: (J1+J2+Jk) is guaranteed even by conditions (1)-(3)
          int sign = ((l + (J1+J2+Jk) / 2) % 2 ? -1 : 1);
          
          // Prefactor: sign * (2*J1+1)*(2*J2+1)*(2*Jk+1) / pi^3
          double pf = sign * (2*J1+1) * (2*J2+1) * (2*Jk+1) / (M_PI * M_PI * M_PI);
 
          // Copy input columns (alpha, beta preserved), then overwrite
          // columns 2-4 with the expanded (J1, J2, Jk) values
          for (int c = 0; c < Ncols; c++) {
            out[row][c] = terms[i][c];
          }
          out[row][2] = J1;
          out[row][3] = J2;
          out[row][4] = Jk;

          // Combined coefficient: input coeff_A * coupling coeff_B
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
