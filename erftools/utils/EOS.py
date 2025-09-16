import numpy as np
from ..constants import Gamma, p_0, R_d, R_v, Cp_d, Cp_v

def getThgivenPandT(T, P, rdOcp=R_d/Cp_d):
    return T * (p_0/P)**rdOcp

def getdPdRgivenConstantTheta(rho, theta):
    return Gamma * p_0 * (R_d * theta / p_0)**Gamma * rho**(Gamma-1.0)

#def getPgivenRTh(rhotheta, qv=0, precip=True):
#    """Function to return pressure given density times theta
#
#    Parameters
#    ----------
#    rhotheta: float
#        Density times potential temperature
#    qv: float, optional
#        Water vapor mixing ratio
#    precip: bool
#        Precipitation is modeled by a single-moment microphysics model,
#        otherwise a warm moisture model is used.
#    """
#    if qv==0:
#        return p_0 * (R_d * rhotheta / p_0)**Gamma
#    elif not precip:
#        rhotheta_t = rhotheta*(1.0+qv);
#        return p_0 * (R_d * rhotheta_t / p_0)**Gamma
#    else:
#        R_t        =  R_d + qv*R_v;
#        Cp_t       = Cp_d + qv*Cp_v;
#        Gamma_t    = Cp_t / (Cp_t-R_t);
#        rhotheta_t = rhotheta*(1.0+qv);
#        return p_0 * (R_t * rhotheta_t / p_0)**Gamma_t
def getPgivenRTh(rhotheta, qv=None):
    # rhotheta is the dry (partial) density times dry potential temperature
    if qv is None:
        qv = np.zeros_like(rhotheta)
    return p_0 * (R_d * rhotheta * (1.0+(R_v/R_d)*qv) / p_0)**Gamma;

def getRhoThetagivenP(p):
    return (p * p_0**(Gamma-1))**(1.0/Gamma) / R_d

def getRhogivenThetaPress(theta, p, rdOcp=R_d/Cp_d, qv=0.0):
    return p_0**rdOcp * p**(1.0/Gamma) / (R_d * theta * (1.0 + R_v/R_d*qv))

def getTgivenRandRTh(rho, rhotheta, qv=0.0):
    # rho and rhotheta are dry values. We should be using moist value of
    # theta when using moisture, theta_m = theta * (1 + R_v/R_d*qv)
    p_loc = p_0 * (R_d * rhotheta * (1.0 + R_v/R_d*qv) / p_0)**Gamma
    # p = rho_d * R_d * T_v
    # where T_v = T * (1 + R_v/R_d*qv)
    return p_loc / (R_d * rho * (1.0 + R_v/R_d*qv) )

def getThgivenRandT(rho, T, rdOcp=R_d/Cp_d, qv=0.0):
    # p = rho_d * R_d * T_moist
    p_loc = rho * R_d * T * (1.0 + R_v/R_d*qv);
    # theta_d = T * (p0/p)^(R_d/C_p)
    return T * (p_0/p_loc)**rdOcp;
