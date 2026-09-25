import numpy as np
from scipy.optimize import brentq


def ces_quantity(x,y,weight,curvature):
    '''CES aggregate, including its Cobb–Douglas limit.'''
    if abs(curvature) < 1e-10:
        return np.exp((1 - weight) * np.log(x) + weight * np.log(y))
    log_sum = np.logaddexp(np.log1p(-weight) + curvature * np.log(x),
                          np.log(weight) + curvature * np.log(y))
    return np.exp(log_sum / curvature)


def production(S,H,K,ModelPar):
    '''
    Evaluate nested-CES output and marginal products.

    Parameters
    ----------
    S, H, K : float or ndarray
        Low-skill task composite, high-skill labour and capital.
    ModelPar : TypeModelParameters
        Production weights and curvatures.

    Returns
    -------
    dict
        Output, HK composite, factor prices and endogenous cost shares.
    '''
    X  = ces_quantity(H,K,ModelPar.α,ModelPar.χ)
    Y  = ces_quantity(S,X,ModelPar.γ,ModelPar.ψ)
    pL = (1 - ModelPar.γ) * (Y / S)**(1 - ModelPar.ψ)
    pX = ModelPar.γ * (Y / X)**(1 - ModelPar.ψ)
    s  = pX * (1 - ModelPar.α) * (X / H)**(1 - ModelPar.χ)
    r  = pX * ModelPar.α * (X / K)**(1 - ModelPar.χ)

    return dict(Y=Y,X=X,pL=pL,pX=pX,s=s,r=r,
                omega_L=pL * S / Y,omega_X=pX * X / Y,
                eta_H=s * H / (pX * X),eta_K=r * K / (pX * X))


def firm_at_interest(I,r,H,L,ModelPar):
    '''
    Solve capital demand at fixed interest rate and marginal task.

    Parameters
    ----------
    I, r, H, L : float
        Offshore share, rental rate and domestic labour supplies.
    ModelPar : TypeModelParameters
        Nested-CES production parameters.

    Returns
    -------
    dict
        Production quantities, prices and shares, including capital demand.
    '''
    S = L / (1 - I)

    def residual(log_K):
        return np.log(production(S,H,np.exp(log_K),ModelPar)['r'] / r)

    log_K = brentq(residual,np.log(H) - 40,np.log(H) + 40,xtol=1e-12)
    K     = np.exp(log_K)
    sol   = production(S,H,K,ModelPar)

    sol['K'] = K
    return sol


def factor_demands(w,s,r,I,Y,ModelPar):
    '''
    Compute CES conditional demands at unit output price.

    Parameters
    ----------
    w, s, r, I, Y : float
        Domestic factor prices, marginal task and output.
    ModelPar : TypeModelParameters
        Nested-CES production and offshoring parameters.

    Returns
    -------
    tuple
        Task-composite, high-skill labour and capital demands.
    '''
    omega = 1 - I - np.expm1(-ModelPar.θ * I) / ModelPar.θ
    chi   = ModelPar.χ
    alpha = ModelPar.α
    if abs(chi) < 1e-10:
        pX = (s / (1 - alpha))**(1 - alpha) * (r / alpha)**alpha
    else:
        total = ((1 - alpha)**(1 / (1 - chi)) * s**(-chi / (1 - chi))
                 + alpha**(1 / (1 - chi)) * r**(-chi / (1 - chi)))
        pX = total**((chi - 1) / chi)
    X = Y * (ModelPar.γ / pX)**(1 / (1 - ModelPar.ψ))
    S = Y * ((1 - ModelPar.γ) / (w * omega))**(1 / (1 - ModelPar.ψ))
    H = X * (pX * (1 - alpha) / s)**(1 / (1 - chi))
    K = X * (pX * alpha / r)**(1 / (1 - chi))
    return S,H,K
