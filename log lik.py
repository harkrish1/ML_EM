import math
def log_likelihood(data, mu, sigma,pi, num_mix):
	L = []
	for i in range(len(data)):
		sum = 0
		for j in range(num_mix):
			sum = sum + (data[j]-mu[i])**2/(2*sigma[i]**2)
		term = len(data)*(math.log(math.sqrt(2*math.pi))+math.log(sigma[i]))
		L.append(-term - sum)
	return L


data = [1,2,3,4]
mu = [1.5,3.5]
sigma = [0.5,0.5]
num_mix=2
print log_likelihood(data, mu, sigma, num_mix)