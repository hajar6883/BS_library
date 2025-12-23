This is a pricing library with modules for valuing European vanilla options using the Black–Scholes–Merton closed-form formula. We then extend the framework to price exotic derivatives using Monte Carlo or automatic differentiation.

This module is essential because it provides a benchmark price for Monte Carlo convergence, analytical Greeks to compare with numerical Greeks, and a control variate for Monte Carlo variance reduction.


## Quick overview 
pricing_engine/
│
├── core/
│   └──Black–Scholes Model
│   └──Closed-form pricing for european options
│   └──No-arbitrage bounds checks
│
├── greeks/
│   └──analytical_greeks   # Delta, Gamma, Vega....
├── implied_vol/
│   └──root finding algorithms (Bisection, Newton and Brent methods)
│
├── monte_carlo_pricer/
│   └──GBM Path Simulator # configurable
│   └──Variance Reduction # optional hooks
│          ├── Antithetic
│          ├── ControlVariate
│   └── Monte Carlo Pricer # unified
│   └── Diagnostics & Convergence
│   
├── notebooks/
│   └──demos
