import numpy as np
from scipy.stats import norm
from numpy import random
import math

np.random.seed(14)

#parameters
lambda_poisson = 2.9
shapeK = 0.8
scale_sigma = 3
n_values = [100,10000]

# Confidence Intervals
alpha = 0.01  # 1% significance level
delta = norm.ppf(1 - alpha / 2)

def generate_poisson():
	L = np.exp(-lambda_poisson)
	k = 0
	p = 1
	while p >= L:
		k += 1
		p *= np.random.uniform()
	return k

def generate_weib():
    u = np.random.uniform()
    x = ((-np.log(1 - u) / lambda_poisson) ** (1 / shapeK))*scale_sigma
    return x	

def rainfall():
    storms = generate_poisson()
    total_rainfall = 0

    for i in range(storms):
        total_rainfall += generate_weib() 

    return total_rainfall

#simple monte carlo 
def simple_monte_carlo(n):
    below_threshold = 0
    for i in range(n):
        total_rainfall = rainfall()
        if total_rainfall < 5:
            below_threshold += 1
    
    return below_threshold / n

def sn_calc(n, mu_hat):
    sn_square = 0
    for i in range(n):
        u = np.random.uniform()
        y = 0
        total_rainfall = rainfall()
        if total_rainfall < 5:
            y = 1
        sn_square += math.pow(y-mu_hat, 2)
    sn_square/= (n-1)
    sn = np.sqrt(sn_square)
    return sn

w = [0] * 7
def calculate_w():
	summation = 0
	for s in range(6):
		x = (np.exp(-lambda_poisson) * (lambda_poisson ** s) ) / math.factorial(s)
		w[s] = x
		summation += x
	w[6] = 1 - summation

def stratified_monte_carlo(n):
	calculate_w()
	num = [0] * 7
	mu = [0] * 7
	den = [0] * 7
	s = [0] * 7
	for i in range(n):
		S = generate_poisson()
		total_rain = 0
		for j in range(S):
			d = generate_weib()
			total_rain += d
		if S > 6:
			S = 6
		den[S] += 1
		if total_rain < 5:
			num[S] += 1
	estimate = 0
	for i in range(7):
		if den[i] != 0: mu[i] = num[i] / den[i] 
	var = strat_confidence_interval(n, mu)
	for i in range(7):
		if den[i] != 0: estimate += (w[i] / den[i]) * num[i]
	l = estimate - 2.58 * (var/ np.sqrt(n))
	u = estimate + 2.58 * (var/ np.sqrt(n))
	return estimate, l, u
	
def strat_confidence_interval(n, mu):
	s2 = [0] * 7
	den = [0] * 7
	for i in range(n):
		y = 0
		S = generate_poisson()
		total_rain = 0
		for j in range(S):
			d = generate_weib()
			total_rain += d
		if S > 6:
			S = 6
		den[S] += 1
		if total_rain < 5:
			y = 1
		s2[S] += (y - mu[S]) ** 2
	var = 0
	for i in range(7):
		if den[i] != 0: var += (w[i] ** 2) * (s2[i] / (den[i] - 1))
	return np.sqrt(var)
	

for n in n_values:
    print("For n = {}".format(n))

    print("Simple Monte Carlo")
    In = simple_monte_carlo(n)
    sn = sn_calc(n, In)
    L = (In) - (delta*sn/math.sqrt(n))
    U = (In) + (delta*sn/math.sqrt(n))

    print("Probability \t\t= {}".format(In))
    print("Confidence Interval \t= [{}, {}]".format(L, U))
    print("Variance \t\t= {}".format(sn*sn))
    print("Interval Length \t= {}".format(U-L))

    print()

    print("Stratification Method")
    In, L, U = stratified_monte_carlo(n)

    print("Probability \t\t= {}".format(In))
    print("Confidence Interval \t= [{}, {}]".format(L, U))
    print("Variance \t\t= {}".format(sn*sn))
    print("Interval Length \t= {}".format(U-L))

    print("-------------------------------------------------------------------------------------------------------")