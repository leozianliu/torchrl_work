import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parameters of the Normal distribution
mu = 0
sigma = 0.1

# Generate normal samples
x = np.random.normal(mu, sigma, size=10000)
y = np.tanh(x)

# Plot original Normal distribution
xx = np.linspace(-4, 4, 1000)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(xx, norm.pdf(xx, mu, sigma), label="Normal PDF", color='orange')
plt.title("Normal Distribution (μ=0, σ=1)")
plt.xlabel("x")
plt.ylabel("Probability Density")
plt.grid(True)
plt.legend()

# Plot the TanhNormal distribution (transformed samples)
plt.subplot(1, 2, 2)
plt.hist(y, bins=100, density=True, alpha=0.7, color='skyblue', edgecolor='gray')
plt.title("Histogram of tanh(Normal(μ=0, σ=1))")
plt.xlabel("Value after tanh")
plt.ylabel("Density")
plt.grid(True)
plt.xlim(-1.05, 1.05)

plt.tight_layout()
plt.show()
