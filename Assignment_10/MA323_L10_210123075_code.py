import matplotlib.pyplot as plt

def VDC(n, b):
    cnt = list()
    
    for i in range(n):
        sum, num, k = 0, i, -1
        while(num):
            residue = num % b
            sum += residue * (b ** k)
            num = int(num / b)
            k -= 1
        cnt.append(sum)
        
    return cnt


def LCG(n):
    x0, a, b, m, values = 100, 1597, 0, 244944, list()
    
    for i in range(n):
        x1 = (a * x0 + b) % m
        u1 = x1 / m
        values.append(u1)
        x0 = x1
        
    return values
    
#1) print first 25 values of the Van der Corput sequence using the radical inverse function.
print(VDC(25, 2))
new_seq = VDC(1000, 2)

plt.scatter(new_seq[:-1], new_seq[1:])
plt.show()

#2) Compare 100 and 100000 values generated by the VDC and LCG.
vdc_seq1 = VDC(100, 2)
vdc_seq2 = VDC(100000, 2)
lcg_seq1 = LCG(100)
lcg_seq2 = LCG(100000)

plt.figure(figsize=(12, 10))
plt.subplot(2, 2, 1)
plt.hist(vdc_seq1, bins=100)
plt.title("VDC with n=100")

plt.subplot(2, 2, 2)
plt.hist(vdc_seq2, bins=100)
plt.title("VDC with n=100000")

plt.subplot(2, 2, 3)
plt.hist(lcg_seq1, bins=100)
plt.title("LCG with n=100, seed=100, a=1597, b=0, m=244944")

plt.subplot(2, 2, 4)
plt.hist(lcg_seq2, bins=100)
plt.title("LCG with n=100000, seed=100, a=1597, b=0, m=244944")

plt.show()

#3) Halton sequence for 2 dimensions.
halton_seq1_1 = VDC(100, 2)
halton_seq1_2 = VDC(100, 3)

halton_seq2_1 = VDC(100000, 2)
halton_seq2_2 = VDC(100000, 3)

plt.figure(figsize=(10, 10))
plt.subplot(2, 1, 1)
plt.scatter(halton_seq1_1, halton_seq1_2)
plt.xlabel("fi2(i)")
plt.ylabel("fi3(i)")
plt.title("Halton with n=100")

plt.subplot(2, 1, 2)
plt.scatter(halton_seq2_1, halton_seq2_2)
plt.title("Halton with n=100000")
plt.xlabel("fi2(i)")
plt.ylabel("fi3(i)")

plt.show()