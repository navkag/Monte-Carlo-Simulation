import numpy as np
import pandas
import matplotlib.pyplot as plt
import time
import random
import math


def box_muller():
    # u1, u2 = np.random.uniform(size=2)
    u1 = random.uniform(0, 1)
    u2 = random.uniform(0, 1)
    
    R = math.sqrt(-2 * math.log(u1))    
    theta = 2 * math.pi * u2
    z1 = R * math.cos(theta)
    z2 = R * math.sin(theta)
    
    return (z1, z2)


def marsaglia_bray(n_samples):
    rejected, cnt = 0, 0
    accepted = 0
    samples = []
    while (accepted < n_samples):
        # np.random.seed(accepted)
        # u1, u2 = np.random.uniform(size=2)
        
        u1 = random.uniform(0, 1)
        u2 = random.uniform(0, 1)
        
        cnt += 1
        u1 = 2 * u1 - 1
        u2 = 2 * u2 - 1
        
        if (u1 ** 2 + u2 ** 2 > 1):
            rejected += 1
            continue
        
        common_term = -2 * math.log(u1 ** 2 + u2 ** 2) / (u1 ** 2 + u2 ** 2)
        common_term = math.sqrt(common_term)
        
        z1 = u1 * common_term
        z2 = u2 * common_term
        accepted += 1
        samples.append([z1, z2])
        
    return samples, rejected, cnt
         

def normal_density(mean, std, samples):
	y = (1 / (np.sqrt(2 * math.pi) * std) ) * np.exp(-0.5 * ((samples - mean) / std) ** 2)
	return y


def all_in_one(samples, mean, std, plot_name, time_req, mode):
    intervals = np.arange(mean - 8, mean + 8.1, 0.1)
    samples_updated = list()
    
    # Get X_i = mean + sigma * U_i
    for sample in samples:
        samples_updated.append(sample * std + mean)    
    samples_updated_np = np.array(samples_updated)
    
    print(f"For sample size: {len(samples)}:")
    print(f"Sample mean: {np.mean(samples_updated_np)}")
    print(f"Sample variance: {np.var(samples_updated_np)}")
    print(f"Execution time for {len(samples)}: {time_req}s.")
    if (mode == 1):
        print()
    
    plt.figure(figsize=(12, 8))
    plt.hist(samples_updated_np, bins=80, density=True)
    actual_samples = np.linspace(mean - 10, mean + 10, 2500, endpoint=True)
    y = normal_density(mean, std, actual_samples)
    plt.plot(actual_samples, y, color='r')
    if (mode == 1):
        plt.title(f"Box-Muller Method Sampling from N({mean}, {round(std ** 2, 2)}) for {len(samples)} samples:")
    else:
        plt.title(f"Marsaglia-Bray Method Sampling from N({mean}, {round(std ** 2, 2)}) for {len(samples)} samples:")
    plt.savefig(plot_name)
    plt.show()

# Box muller - 100 samples
samples = list()

checkpoint1 = time.time()

for i in range(100):
    # np.random.seed(i)
    [z1,z2] = box_muller()
    samples.append(z1)
    # samples.append(z2)
 
checkpoint2 = time.time()
time_taken = checkpoint2 - checkpoint1

all_in_one(samples, mean=0, std=1, plot_name="BM-100-1", time_req=time_taken, mode=1)
all_in_one(samples, mean=0, std=np.sqrt(5), plot_name="BM-100-2", time_req=time_taken, mode=1)
all_in_one(samples, mean=5, std=np.sqrt(5), plot_name="BM-100-3", time_req=time_taken, mode=1)

# Box muller - 10000 samples
samples = list()

checkpoint1 = time.time()

for i in range(10000):
    # np.random.seed(i)
    [z1,z2] = box_muller()
    samples.append(z1)
    # samples.append(z2)
 
checkpoint2 = time.time()
time_taken = checkpoint2 - checkpoint1

all_in_one(samples, mean=0, std=1, plot_name="BM-10000-1", time_req=time_taken, mode=1)
all_in_one(samples, mean=0, std=np.sqrt(5), plot_name="BM-10000-2", time_req=time_taken, mode=1)
all_in_one(samples, mean=5, std=np.sqrt(5), plot_name="BM-10000-3", time_req=time_taken, mode=1)

# Box muller - 100000 samples
samples = list()

checkpoint1 = time.time()

for i in range(100000):
    # np.random.seed(i)    
    [z1,z2] = box_muller()
    samples.append(z1)
    # samples.append(z2)
 
checkpoint2 = time.time()
time_taken = checkpoint2 - checkpoint1

all_in_one(samples, mean=0, std=1, plot_name="BM-100000-1", time_req=time_taken, mode=1)
all_in_one(samples, mean=0, std=np.sqrt(5), plot_name="BM-100000-2", time_req=time_taken, mode=1)
all_in_one(samples, mean=5, std=np.sqrt(5), plot_name="BM-100000-3", time_req=time_taken, mode=1)


# Marsaglia and Bray - 100 samples.
checkpoint1 = time.time()

samples, rejected_samples, total = marsaglia_bray(n_samples=100)
 
checkpoint2 = time.time()
time_taken = checkpoint2 - checkpoint1

samples = np.array(samples)[:, 0]
samples = np.ravel(samples)

all_in_one(samples, mean=0, std=1, plot_name="MB-100-1", time_req=time_taken, mode=2)
print(f"Proportion of values rejected: {rejected_samples / total}\n")
all_in_one(samples, mean=0, std=np.sqrt(5), plot_name="MB-100-2", time_req=time_taken, mode=2)
print(f"Proportion of values rejected: {rejected_samples / total}\n")
all_in_one(samples, mean=5, std=np.sqrt(5), plot_name="MB-100-3", time_req=time_taken, mode=2)
print(f"Proportion of values rejected: {rejected_samples / total}\n")

# Marsaglia and Bray - 10000 samples.
checkpoint1 = time.time()

samples, rejected_samples, total = marsaglia_bray(n_samples=10000)
 
checkpoint2 = time.time()
time_taken = checkpoint2 - checkpoint1

samples = np.array(samples)[:, 0]
samples = np.ravel(samples)

all_in_one(samples, mean=0, std=1, plot_name="MB-10000-1", time_req=time_taken, mode=2)
print(f"Proportion of values rejected: {rejected_samples / total}\n")
all_in_one(samples, mean=0, std=np.sqrt(5), plot_name="MB-10000-2", time_req=time_taken, mode=2)
print(f"Proportion of values rejected: {rejected_samples / total}\n")
all_in_one(samples, mean=5, std=np.sqrt(5), plot_name="MB-10000-3", time_req=time_taken, mode=2)
print(f"Proportion of values rejected: {rejected_samples / total}\n")

# Marsaglia and Bray - 100000 samples.
checkpoint1 = time.time()

samples, rejected_samples, total = marsaglia_bray(n_samples=100000)
 
checkpoint2 = time.time()
time_taken = checkpoint2 - checkpoint1

samples = np.array(samples)[:, 0]
samples = np.ravel(samples)

all_in_one(samples, mean=0, std=1, plot_name="MB-100000-1", time_req=time_taken, mode=2)
print(f"Proportion of values rejected: {rejected_samples / total}\n")
all_in_one(samples, mean=0, std=np.sqrt(5), plot_name="MB-100000-2", time_req=time_taken, mode=2)
print(f"Proportion of values rejected: {rejected_samples / total}\n")
all_in_one(samples, mean=5, std=np.sqrt(5), plot_name="MB-100000-3", time_req=time_taken, mode=2)
print(f"Proportion of values rejected: {rejected_samples / total}\n")