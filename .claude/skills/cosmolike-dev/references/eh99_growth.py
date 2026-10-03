"""Eisenstein & Hu 1999 (ApJ 511, 5), Sec. 2.2: scale-dependent growth with
massive neutrinos, D_cb(k,z) (eq. 12) and D_cbnu(k,z) (eq. 13), compared
with CAMB's linear P(k,z) ratios. LCDM (w = -1), one massive species."""
import sys, numpy as np
H, OM, OB, TCMB = 0.6732, 0.3, 0.04, 2.7255
th = TCMB/2.7
def D1(z, OL):
  g2 = OM*(1+z)**3 + (1 - OM - OL)*(1+z)**2 + OL
  Oz, OLz = OM*(1+z)**3/g2, OL/g2
  zeq = 2.50e4*OM*H**2*th**-4
  return (1+zeq)/(1+z)*2.5*Oz/(Oz**(4/7) - OLz + (1 + Oz/2)*(1 + OLz/70))   # eq. 4
def eh(k, z, mnu, Nnu=1, which="cbnu"):
  OL = 1 - OM
  fnu = mnu/93.14/H**2/OM; fcb = 1 - fnu
  pcb = (5 - np.sqrt(1 + 24*fcb))/4                                         # eq. 11
  q = k*th**2/(OM*H**2)                                                     # eq. 5, k in 1/Mpc
  yfs = 17.2*fnu*(1 + 0.488*fnu**(-7/6))*(Nnu*q/fnu)**2                     # eq. 14
  d1 = D1(z, OL)
  if which == "cb":
    return (1 + (d1/(1 + yfs))**0.7)**(pcb/0.7)*d1**(1 - pcb)              # eq. 12
  return (fcb**(0.7/pcb) + (d1/(1 + yfs))**0.7)**(pcb/0.7)*d1**(1 - pcb)    # eq. 13
ks = np.array([5e-4, 1e-3, 3e-3, 0.01, 0.03, 0.05, 0.1, 0.2, 1.0])
which = sys.argv[1] if len(sys.argv) > 1 else "cbnu"
print("EH99 %s: D(k,z)/D(k0,z) - 1 [%%] (each normalized to z = 0), k = %s /Mpc" % (which, " ".join("%g" % k for k in ks)))
for mnu in (0.06, 0.15, 0.3, 0.6):
  for z in (1.0, 2.0):
    D = eh(ks, z, mnu, which=which)/eh(ks, 0.0, mnu, which=which)
    print("mnu = %.2f eV, z = %.0f: %s" % (mnu, z, " ".join("%+.2f" % x for x in 100*(D/D[0] - 1))))
