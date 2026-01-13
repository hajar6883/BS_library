import numpy as np 

def hagan_SABR(K,F,T, alpha, beta, rho, nu, eps=1e-07):
    """ the hagan approx. for the SABR implied Black vol for european option"""

    one_minus_beta = 1.0 - beta
    FK_beta = (F * K) ** (one_minus_beta / 2.0)

    # Time correction common terms (order-T expansion