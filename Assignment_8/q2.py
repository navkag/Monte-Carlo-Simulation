import random
import math

alpha=[2082, 1999, 2008, 2047, 2199, 2153, 1999, 2136, 2053, 2121, 1974, 2110, 2110, 2168, 2035, 2019, 2044, 2191, 2284, 1912, 2196, 2099, 2041, 2192, 2188, 1984, 2158, 2019, 2032, 2051, 2192, 2133, 2142, 2113, 2150, 2221, 2046, 2127]

def generate_gamma(shape):
	y=0
	for i in (0,shape):
		u=random.random()
		x=-math.log(u)
		y+=x
	return y


def probability_generator(n):
    count=0
    for i in range(0,n):
        values=[0]
        x=0
        for j in alpha:
            y=0
            
            if j==2284:
                x=generate_gamma(j)
                values.append(x)
            else: 
                values.append(generate_gamma(j))

        if max(values)==x: count+=1	
    probability= count/n
    print(f"Value of μ = P( X19 = max i (Xi)) using conditional monte carlo technique is {probability}")

probability_generator(100000)


import numpy as np
from scipy.stats import dirichlet

# Parameters
alpha = np.array([0.5] * 38)  # Dirichlet distribution parameters
num_samples = 1000000  # Number of Monte Carlo samples
target_index = 18  # Index of X₁₉

# Generate random samples from the Dirichlet distribution
samples = dirichlet.rvs(alpha, size=num_samples)

# Calculate conditional probability that X₁₉ is the maximum value
conditional_probability = np.mean(samples[:, target_index] == samples.max(axis=1))

# The conditional expectation μ is the calculated probability
mu = conditional_probability

# Output the result
print(f"Conditional Expectation using direct dirichlett distribution from scipy library is μ: {mu}")
