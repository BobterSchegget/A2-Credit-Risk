from numpy.ma.core import sqrt, ones, zeros, mean, std, sort, floor
from scipy import stats
import matplotlib.pyplot as plt

# One factor model
def simOneFactor(n, PD, EAD, LGD, rho):
    normDist = stats.norm(0, 1)
    z = normDist.rvs(1)  # systematic factor
    y = normDist.rvs(n)  # idiosyncratic factor
    rtilde = sqrt(rho)*z + sqrt(1-rho)*y # Asset values for obligors
    c = normDist.ppf(PD) # Critical thresholds for obligors
    default = (rtilde < c)
    losses = default * EAD * LGD
    return sum(losses)

n = 100     # The number of obligors in the portfolio.
runs = 5000 # Number of realizations.

EAD = stats.uniform.rvs(0, 2, size=n)
LGD = ones(n)
PD = 0.25 * ones(n);
rho = 0.2
losses = zeros(runs)
for i in range(runs):
    losses[i] = simOneFactor(n, PD, EAD, LGD, rho)

# Histogram total loss distribution
plt.figure()   # create a new plot window
plt.hist(losses, bins=100, density=False)
plt.show()

print(mean(losses))  # Expected loss
print(std(losses))   # Unexpected loss

alpha = 0.95
sortLosses = sort(losses)
idx = int(floor(alpha * runs))
VaR = sortLosses[idx]
print(VaR) # Value-at-risk

TCE = mean(losses[losses > VaR])
print(TCE)  # Tail conditional expectation


# Bernoulli mixture model
def simBernoulliMixture(n, EAD, LGD, Pdist):
    P = Pdist.rvs(1)
    binDist = stats.bernoulli(P)
    default = binDist.rvs(n)
    losses = default * EAD * LGD
    return sum(losses)

Pdist = stats.beta(1, 3)
print(Pdist.mean())   # PD = 0.25
losses = zeros(runs)
for i in range(runs):
    losses[i] = simBernoulliMixture(n, EAD, LGD, Pdist)

# Histogram total loss distribution
plt.figure()   # create a new plot window
plt.hist(losses, bins=100, density=False)
plt.show()

print(mean(losses))  # Expected loss
print(std(losses))   # Unexpected loss

alpha = 0.95
sortLosses = sort(losses)
idx = int(floor(alpha * runs))
VaR = sortLosses[idx]
print(VaR) # Value-at-risk

TCE = mean(losses[losses > VaR])
print(TCE)  # Tail conditional expectation


