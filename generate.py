import math
import random as rand
import numpy as np
# generates data based on a given mean and variance for both
# x and y
# writes it out to whatever file is opened and read
# in the format:
# x y
# x y
# etc

# cd ../../../Users/ryancarter/Documents/AAAUNI/400/COSC480/LTE_RACH_positioning-master/



x_mean = -1.08802089
y_mean = -2.79061636

x_variance = 1.7958253154354453

y_variance = 1.7953863308602886

# z_variance = 0.0000001

file = open("100_150.txt", "w")
file.write("")
file.close()
file = open("100_150.txt", "a")

x = [rand.normalvariate(x_mean, math.sqrt(x_variance)) for i in range(0,7189)]
y = [rand.normalvariate(y_mean, math.sqrt(y_variance)) for i in range(0,7189)]
# z = [rand.normalvariate(z_mean, math.sqrt(z_variance)) for i in range(0,597)]

print("Generated mean of X: "+str(np.mean(np.array(x))))
print("Generated mean of Y: "+str(np.mean(np.array(y))))
# print(np.mean(np.array(z)))

for i in range(0, 7189):
    file.write(str(x[i])+" "+
    str(y[i])+"\n")#+
    # str(z[i])+"\n")

# 12.002494371649512
# 6.009519908989065
# 2.0099634413775416
