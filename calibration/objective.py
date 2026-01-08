
import time

import numpy as np
from surfaces.model_iv_surface import heston_iv_surface_on_m_grid

global CALL_COUNT
CALL_COUNT = 0

def heston_calib_loss_iv(
    params,  # (v0, kappa, theta, sigma, rho)
    spot, m_grid, maturities, r, q,
    IV_mkt, W=None,                 # weights matrix same shape as IV_mkt
    u_max=100.0, n_u=2001,
    cp="C",
    penalty_feller=0.0, ):


    CALL_COUNT += 1

    t0 = time.time()
    v0, kappa, theta, sigma, rho = params

    # Hard bounds guard (fast reject)
    if (v0 <= 0) or (kappa <= 0) or (theta <= 0) or (sigma <= 0) or (abs(rho) >= 0.999):
        return 1e9

    IV_model = heston_iv_surface_on_m_grid(
        spot=spot,
        m_grid=m_grid,
        maturities=maturities,
        r=r, q=q,
        heston_params=params,
        u_max=u_max, n_u=n_u,
        cp=cp,
        debug=False,
    )

    # Mask out bad points (NaNs from inversion or missing market vols)
    mask = np.isfinite(IV_model) & np.isfinite(IV_mkt)
    if not np.any(mask):
        return 1e9

    diff = IV_model[mask] - IV_mkt[mask] #residuals

    if W is None:
        loss = np.mean(diff**2)
    else:
        w = W[mask]
        # normalize weights to avoid scale issues
        w = w / np.mean(w)
        loss = np.mean(w * diff**2)
    
    

    # Optional soft penalty for Feller violation
    if penalty_feller > 0.0:
        feller_violation = max(0.0, sigma*sigma - 2.0*kappa*theta)
        loss += penalty_feller * (feller_violation**2)

    dt = time.time() - t0
    print(f"[call {CALL_COUNT:03d}] loss={loss:.6e}  time={dt:.2f}s")

    return float(loss)

