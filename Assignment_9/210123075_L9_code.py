import numpy as np
import matplotlib.pyplot as plt


def gen_exp(theta):
    u = np.random.uniform()
    return (-theta * np.log(u))


theta = [4, 4, 2, 5, 2, 3, 2, 3, 2, 2]
values = list()
cnt = 0
n = 10_000
for _ in range(n):
    T_1 = gen_exp(theta[0])
    T_2 = gen_exp(theta[1])
    T_3 = gen_exp(theta[2])
    T_4 = gen_exp(theta[3])
    T_5 = gen_exp(theta[4])
    T_6 = gen_exp(theta[5])
    T_7 = gen_exp(theta[6])
    T_8 = gen_exp(theta[7])
    T_9 = gen_exp(theta[8])
    T_10 = gen_exp(theta[9])

    E_10 = T_10 + T_1 + max(T_4 + T_2, T_9 + max(T_5 + T_2, T_6 + T_3, T_7 + T_3), T_3 + T_8)
    values.append(E_10)
    if (E_10 > 70):
        cnt += 1
    
print(f"Probability of missing the deadlines: {round(cnt / n, 10)}")
# print(cnt)
print(f"Mean: {np.mean(values)}")
print(f"Standard deviation: {np.std(values)}", end='\n\n')

mc_variance = np.var(values)

plt.hist(values, bins=100)
plt.xlabel("Samples")
plt.ylabel("Frequencies")
plt.show()

# part (e) using importance sampling.

def p(x, theta):
    lambda_ = 1 / theta
    return lambda_ * np.exp(-lambda_ * x)


values_1 = list()
cnt = 0
n = 10_000
for _ in range(n):
    T_1 = gen_exp(theta[0] * 4) 
    T_1 *= (p(T_1, theta[0]) / p(T_10, theta[0] * 4))
    T_2 = gen_exp(theta[1] * 4) 
    T_2 *= (p(T_2, theta[1]) / p(T_2, theta[1] * 4))
    T_3 = gen_exp(theta[2] * 4) 
    T_3 *= (p(T_3, theta[2]) / p(T_3, theta[2] * 4))
    T_4 = gen_exp(theta[3] * 4) 
    T_4 *= (p(T_4, theta[3]) / p(T_4, theta[3] * 4))
    T_5 = gen_exp(theta[4] * 4) 
    T_5 *= (p(T_5, theta[4]) / p(T_5, theta[4] * 4))
    T_6 = gen_exp(theta[5] * 4) 
    T_6 *= (p(T_6, theta[5]) / p(T_6, theta[5] * 4))
    T_7 = gen_exp(theta[6] * 4) 
    T_7 *= (p(T_7, theta[6]) / p(T_7, theta[6] * 4))
    T_8 = gen_exp(theta[7] * 4) 
    T_8 *= (p(T_8, theta[7]) / p(T_8, theta[7] * 4))
    T_9 = gen_exp(theta[8] * 4) 
    T_9 *= (p(T_9, theta[8]) / p(T_9, theta[8] * 4))
    T_10 = gen_exp(theta[9] * 4) 
    T_10 *= (p(T_10, theta[9]) / p(T_10, theta[9] * 4))

    E_10 = T_10 + T_1 + max(T_4 + T_2, T_9 + max(T_5 + T_2, T_6 + T_3, T_7 + T_3), T_3 + T_8)
    values_1.append(E_10)
    if (E_10 > 70):
        cnt += 1
    
print(f"Probability of missing the deadlines: {round(cnt / n, 10)}")
# print(cnt)
print(f"Mean: {np.mean(values_1)}")
print(f"Standard deviation: {np.std(values_1)}")
print(f"Effective sample size {mc_variance / np.var(values_1)}", end='\n\n')

plt.hist(values_1, bins=100)
plt.xlabel("Samples")
plt.ylabel("Frequencies")
plt.show()

# part (f) using importance sampling.
means, variances, eff_ss = list(), list(), list()
for k in [3, 4, 5]:
    print(f"Considering k = {k}.")
    values_2 = list()
    cnt = 0
    n = 10_000
    for _ in range(n):
        T_1 = gen_exp(theta[0] * k) 
        T_1 *= (p(T_1, theta[0]) / p(T_10, theta[0] * k))
        T_2 = gen_exp(theta[1] * k) 
        T_2 *= (p(T_2, theta[1]) / p(T_2, theta[1] * k))
        T_3 = gen_exp(theta[2]) 
        T_4 = gen_exp(theta[3] * k) 
        T_4 *= (p(T_4, theta[3]) / p(T_4, theta[3] * k))
        T_5 = gen_exp(theta[4]) 
        T_6 = gen_exp(theta[5]) 
        T_7 = gen_exp(theta[6]) 
        T_8 = gen_exp(theta[7]) 
        T_9 = gen_exp(theta[8]) 
        T_10 = gen_exp(theta[9] * k) 
        T_10 *= (p(T_10, theta[9]) / p(T_10, theta[9] * k))

        E_10 = T_10 + T_1 + max(T_4 + T_2, T_9 + max(T_5 + T_2, T_6 + T_3, T_7 + T_3), T_3 + T_8)
        values_2.append(E_10)
        if (E_10 > 70):
            cnt += 1
        
    print(f"Probability of missing the deadlines: {round(cnt / n, 10)}")
    # print(cnt)
    print(f"Mean: {np.mean(values_2)}")
    means.append(np.mean(values_2))
    print(f"Standard deviation: {np.std(values_2)}")
    variances.append(np.var(values_2))
    print(f"Effective sample size: {mc_variance / np.var(values_2)}", end='\n\n')
    eff_ss.append(mc_variance / np.var(values_2))

    plt.hist(values_2, bins=100)
    plt.xlabel("Samples")
    plt.ylabel("Frequencies")
    plt.show()
    
# Part (h)
geld = np.argmin(eff_ss)
print(f"k = {geld + 3} has the minimum effective sample size.")
lower_bound = means[geld] - 2.58 * np.sqrt(variances[geld])
upper_bound = means[geld] + 2.58 * np.sqrt(variances[geld])
print(f"99% Confidence interval for k = {geld + 3}: {(lower_bound, upper_bound)}")