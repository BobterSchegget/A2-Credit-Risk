### Question 3
#### Imports
from numpy.ma.core import sqrt, ones, zeros, mean, std, sort, floor, round, prod
from numpy import fill_diagonal
from scipy import stats
import matplotlib.pyplot as plt
import profile
#### One-factor model template

# One factor model
@profile
def simOneFactorInfectious(n, PD, EAD, LGD, rho, q):
    normDist = stats.norm(0, 1)
    z = normDist.rvs(1)  # systematic factor
    y = normDist.rvs(n)  # idiosyncratic factor
    rtilde = sqrt(rho)*z + sqrt(1-rho)*y # Asset values for obligors
    c = normDist.ppf(PD) # Critical thresholds for obligors
    default_1 = (rtilde < c)

    # Infectious defaults
    bernoulliDist = stats.bernoulli(q)
    Q = bernoulliDist.rvs((n, n)) # Generate infection matrix
    fill_diagonal(Q, 0) # Ensures no self-infections
    # for i in range(n): # Ensures no self-infections
    #     Q[i, i] = 0
    default_2 = 1 - prod(1 - default_1 * Q, axis = 1) # Infection in step 2

    # Combining both steps
    default = default_1 + default_2 - default_1 * default_2

    losses = default * EAD * LGD
    return sum(losses), sum(default)
#### Simulation
# The bank portfolios, numbers are obtained from assignment description
portfolios = {
    "ANB AMOR": {
        "n": 8000,
        "EAD": ones(8000),
        "LGD": 0.25 * ones(8000),
        "PD": 0.01 * ones(8000),
        "rho": 0.25,
    },
    "NSN": {
        "n": 1017,
        "EAD": ones(1000).tolist() + [10]*10 + [50]*4 + [100]*2 + [500],
        "LGD": 0.5 * ones(1017),
        "PD": 0.02 * ones(1017),
        "rho": 0.25,
    },
    "ROBA": {
        "n": 1017,
        "EAD": ones(1000).tolist() + [10]*10 + [50]*4 + [100]*2 + [500],
        "LGD": 0.5 * ones(1017),
        "PD": ([0.0275]*1000 + [0.02]*10 + [0.0175]*4 + [0.015]*2 + [0.008]),
        "rho": 0.25,
    },
}
# Parameters
runs = 20 # Number of simulation runs.
alpha = 0.99 # alpha-value
q = 0.001 # Infectious default parameter
# Subplots
fig1, axes1 = plt.subplots(1, 3, figsize=(16, 5))
fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))

# The simulation for each portfolio
for idx, p in enumerate(portfolios):
    n = portfolios[p]["n"]
    EAD = portfolios[p]["EAD"]
    LGD = portfolios[p]["LGD"]
    PD = portfolios[p]["PD"]
    rho = portfolios[p]["rho"]
    
    losses = zeros(runs)
    defaults = zeros(runs)
    for i in range(runs):
        loss, default = simOneFactorInfectious(n, PD, EAD, LGD, rho, q)
        losses[i] = loss
        defaults[i] = default

    # Plot for total loss
    axes1[idx].hist(losses, bins=int(round(sqrt(runs))), color='blue', alpha=0.7, edgecolor='black')
    #axes1[idx].set_yscale('log')
    axes1[idx].set_title(f"Total loss at {p} Bank")
    axes1[idx].set_xlabel("Total loss")
    axes1[idx].set_ylabel("Frequency")

    # Plot for number of defaults
    axes2[idx].hist(defaults, bins=int(round(sqrt(runs))), color='blue', alpha=0.7, edgecolor='black')
    #axes2[idx].set_yscale('log')
    axes2[idx].set_title(f"Number of defaults at {p} Bank")
    axes2[idx].set_xlabel("Number of defaults")
    axes2[idx].set_ylabel("Frequency")

    # Calculate metrics
    EL = mean(losses)
    UL = std(losses)
  
    sortLosses = sort(losses)
    idx = int(floor(alpha * runs))
    VaR = sortLosses[idx] # Value-at-risk

    EC = VaR - EL # Economical capital
    TCE = mean(losses[losses > VaR]) # Tail conditional expectation

    # Confidence intervals (95%)
    z = 1.96
    EL_CI = [round(EL - z * (UL / sqrt(runs)), 2), round(EL + z * (UL / sqrt(runs)), 2)]
    UL_CI = [round(UL - z * (UL / sqrt(runs)), 2), round(UL + z * (UL / sqrt(runs)), 2)]
    VaR_CI = [round(VaR - z * (UL / sqrt(runs)), 2), round(VaR + z * (UL / sqrt(runs)), 2)]
    EC_CI = [round(EC - z * (UL / sqrt(runs)), 2), round(EC + z * (UL / sqrt(runs)), 2)]
    TCE_CI = [round(TCE - z * (UL / sqrt(runs)), 2), round(TCE + z * (UL / sqrt(runs)), 2)]

    # Print results with confidence intervals
    print(f"Portfolio of {p} bank")
    print(f"Expected loss (EL): {round(EL, 2)} ({EL_CI[0]}, {EL_CI[1]})")
    print(f"Unexpected loss (UL): {round(UL, 2)} ({UL_CI[0]}, {UL_CI[1]})")
    print(f"Value-at-Risk (VaR): {round(VaR, 2)} ({VaR_CI[0]}, {VaR_CI[1]})")
    print(f"Economic capital (EC): {round(EC, 2)} ({EC_CI[0]}, {EC_CI[1]})")
    print(f"Tail conditional expectation (TCE): {round(TCE, 2)} ({TCE_CI[0]}, {TCE_CI[1]})")
    print("\n")

plt.tight_layout()
plt.show()