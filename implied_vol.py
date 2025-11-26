#Root-finding methods to invert the pricing formula.
# Newton–Raphson 
from core import EuropeanCall,BlackScholesModel
from greeks import CallGreeks

def implied_vol_newton(call, market_price, r, sigma_init=0.2, 
                       max_iter=100, tol=1e-8):

    sigma = sigma_init

    for i in range(max_iter):

        model = BlackScholesModel(r, sigma)
        
        price = call.price(model)
        vega  = CallGreeks(call, model).vega()

        # f(sigma) = model_price - market_price
        diff = price - market_price

        # convergence check
        if abs(diff) < tol:
            return sigma

        if vega < 1e-8:
            # Newton is unstable here
            return None   # fallback to Brent ideally

        sigma = sigma - diff / vega

        # Vol must remain positive
        if sigma < 0:
            sigma = 1e-8

  
    return None  # Did not converge




# Brent’s method