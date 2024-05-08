import numpy as np
from scipy.stats import poisson, weibull_min, norm
import math

print("***** Q3 *****")

# doing this for 100000 values



mu_values = [1.0, 1.2, 0.8, 1.5, 1.3] # sum of mu_values is 5.8
sigma2_values = [0.1, 0.2, 0.15, 0.3, 0.25] # sum of sigma2_values is 1

total = 100000



print(f"So mu_values that I took are {mu_values}\n")
print(f"So sigma2_values that I took are {sigma2_values}\n")
print(f"Total number of iteration to estimate actual expectation is {total}\n")


def BOX_MULLER_exponential(N,mu,sigma2):
    values=[]
    sigma=math.sqrt(sigma2)
    
    U1=np.random.uniform(size=N//2)
    U2=np.random.uniform(size=N//2)
    R= np.sqrt(-2* np.log(U1))
    theta= 2*math.pi*U2
    for i in range(N//2):
        values.append(math.pow(math.e,mu+ sigma * R[i]*math.cos(theta[i])))
        values.append(math.pow(math.e,mu+ sigma* R[i]*math.sin(theta[i])))
        
   
    return values
    
    
list1= BOX_MULLER_exponential(total,mu_values[0],sigma2_values[0])
list2= BOX_MULLER_exponential(total,mu_values[1],sigma2_values[1])
list3= BOX_MULLER_exponential(total,mu_values[2],sigma2_values[2])
list4= BOX_MULLER_exponential(total,mu_values[3],sigma2_values[3])
list5= BOX_MULLER_exponential(total,mu_values[4],sigma2_values[4])


def fx():
    
    main_list=[]
    u=0
    
    for i in range(total):
        main_list.append(max(0,1/5 * (list1[i]+list2[i]+list3[i]+list4[i]+list5[i])))
        u=u+max(0,1/5 * (list1[i]+list2[i]+list3[i]+list4[i]+list5[i]))
    
    u= u/total
    
    return main_list,u
    

def hx():
    
    main_list=[]
    u=0
    
    for i in range(total):
        main_list.append(max(0,1/5 * (list1[i]*list2[i]*list3[i]*list4[i]*list5[i])))
        u=u+max(0,1/5 * (list1[i]*list2[i]*list3[i]*list4[i]*list5[i]))
    
    u= u/total
    
    return main_list,u


fx_list, u = fx()

hx_list, theta_dash= hx()

actual_theta= 1/5*math.pow(math.e,(5.8 + 1/2 * (1)))


numerator_sum=0
denominator_sum=0 

for i in range(total):
    numerator_sum+= (fx_list[i]-u)*(hx_list[i]-theta_dash)
    denominator_sum+= (hx_list[i]-theta_dash)*(hx_list[i]-theta_dash)
    

beta = numerator_sum/denominator_sum


estimated_mu= u + beta*(actual_theta-theta_dash)  



print(f"Estimated mean using covariate method comes out to be {estimated_mu}\n")