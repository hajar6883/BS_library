
from scipy.stats import norm as Normal  # Normal distributions helpers 
import numpy as np
class BlackScholesModel:
    def __init__(self, r, sigma, q=0.0):
        self.r = r
        self.q = q
        self.sigma = sigma

    def _d1_d2(self, S, K, T):
        if T <= 0 or self.sigma <= 0:
            raise ValueError("T and sigma must be positive")

        vol_sqrt = self.sigma * np.sqrt(T)
        d1 = (np.log(S / K) + (self.r - self.q + 0.5 * self.sigma**2) * T) / vol_sqrt
        d2 = d1 - vol_sqrt
        return d1, d2

    def call_price(self, S, K, T):
        if T <= 0:
            return max(S - K, 0.0)

        d1, d2 = self._d1_d2(S, K, T)
        return (
            np.exp(-self.q * T) * S * Normal.cdf(d1)
            - np.exp(-self.r * T) * K * Normal.cdf(d2)
        )

    def put_price(self, S, K, T):
        if T <= 0:
            return max(K - S, 0.0)

        d1, d2 = self._d1_d2(S, K, T)
        return (
            np.exp(-self.r * T) * K * Normal.cdf(-d2)
            - np.exp(-self.q * T) * S * Normal.cdf(-d1)
        )

   

    
    
        
        


    

