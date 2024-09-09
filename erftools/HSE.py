import numpy as np

from .constants import R_d, Cp_d, CONST_GRAV
from .EOS import getRhogivenThetaPress

def Newton_Raphson_hse(p,F,dz,C,T,qv,
                       qt=None,
                       g=CONST_GRAV,
                       RdOCp=R_d/Cp_d,
                       eps=1e-6,
                       tol=1e-12,
                       maxiter=25,
                       verbose=False):
    if qt is None:
        qt = qv
    p_init = p
    for it in range(maxiter):
        hi = getRhogivenThetaPress(T, p + eps, RdOCp, qv);
        lo = getRhogivenThetaPress(T, p - eps, RdOCp, qv);
        dFdp = 1.0 + (hi - lo)/(4*eps)*g*dz;
        p -= F/dFdp;
        assert p > 0.

        # Diagnose density and residual
        rho_d = getRhogivenThetaPress(T, p, RdOCp, qv);
        assert rho_d > 0.0
        rho_t = rho_d * (1. + qt);
        F = p + 0.5*rho_t*g*dz + C;
        #if verbose: print(f'F = {p} + 0.5*{rho_t}*9.81*{dz} + {C} = {F}')

        if np.abs(F) <= tol:
            break
    if verbose:
        print('Newton-Raphson converged after',it+1,'iterations',F)
    if it==maxiter:
        print('WARNING: HSE Newton iterations did not converge to tol =',tol)

    return p, rho_d

