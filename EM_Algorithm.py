import easygui
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from random import uniform

data = []
with open("C:\Users\harsh\Downloads\data3.txt") as file:
	for line in file:
		line = line.strip() 
		data.append(float(line))

def prob_data_cluster(data, num_mix, mu, sigma):
	p = []
	for i in range(num_mix):
		p.append([])
		for j in range(len(data)):
			p[i].append([])
			power = -1*((data[j]-mu[i])**2)/(2*sigma[i]**2)
			p[i][j]=(math.exp(power)/(math.sqrt(2 *math.pi)*sigma[i]))
	return p

def prob_cluster_data(p_data_cluster, pi):
	p=[]
	n=[]
	den = []
	for i in range(len(p_data_cluster[0])):
		den_1 = 0
		for j in range(len(p_data_cluster)):
			den_1 = float(den_1) + (float(pi[j])*float(p_data_cluster[j][i]))
		den.append(den_1)
	for i in range(len(p_data_cluster)):
		p.append([])
		n1 = 0
		for j in range(len(p_data_cluster[i])):
			p[i].append([])
			p[i][j] = (float(p_data_cluster[i][j])*float(pi[i]))/float(den[j])
			n1 = n1 + p[i][j]
		n.append(n1)
	return p,n


def maximization(p_cluster_data, data, n_cluster):
	# mean calculation
	mu = []
	for i in range(len(p_cluster_data)):
		sum = 0
		for j in range(len(p_cluster_data[i])):
			sum = sum + (float(p_cluster_data[i][j])*float(data[j]))
		mu.append(float(sum)/float(n_cluster[i]))
	# SD Calculation
	sigma = []
	for i in range(len(p_cluster_data)):
		sum = 0
		for j in range(len(p_cluster_data[i])):
			sum = float(sum) + (float(p_cluster_data[i][j])*float(data[j]**2))
		sigma.append(math.sqrt((float(sum)/float(n_cluster[i]))-float(mu[i]**2)))
	pi = []
	for i in range(len(p_cluster_data)):
		pi.append(n_cluster[i]/len(p_cluster_data[1]))
		
	return mu, sigma, pi

def log_likelihood(data, mu, sigma, num_mix, pi):
	p_data_cluster = prob_data_cluster(data, num_mix, mu, sigma)
	L=0
	for i in range(len(data)):
		for j in range(num_mix):
			L = L + (pi[j]*p_data_cluster[j][i])
	return L

# Initialize Step
def initialize(num_mix):
	pi = []
	mu = []
	sigma = []
	for i in range(num_mix):
		pi.append(1.0/num_mix)
		mu.append(uniform(100, 1000))
		sigma.append(uniform(1000, 10000))
	return mu, sigma, pi


num_mix = 2

log_lik = []
max_lik = []
while num_mix <= 20:
	print "# Mixtures Iteration ", num_mix
	log_lik.append([])
	mu = initialize(num_mix)[0]
	sigma = initialize(num_mix)[1]
	pi = initialize(num_mix)[2]
	iters = 0
	while iters<1000:
		# Expectation Step
		p_data_cluster = prob_data_cluster(data, num_mix, mu, sigma)
		p_cluster_data = prob_cluster_data(p_data_cluster, pi)[0]
		n_cluster = prob_cluster_data(p_data_cluster, pi)[1]
		# Maximization Step
		mu = maximization(p_cluster_data, data, n_cluster)[0]
		sigma = maximization(p_cluster_data, data, n_cluster)[1]
		pi = maximization(p_cluster_data, data, n_cluster)[2]
		# Finding Likelihood
		log_lik[num_mix-2].append(log_likelihood(data, mu, sigma, num_mix, pi))
		iters = iters + 1
	max_lik.append(max( log_lik[num_mix-2] ) ) 
	num_mix = num_mix + 1
	
plt.plot(range(2,21), max_lik)
plt.ylabel('Log Likelihood for three clusters')
plt.xlabel('Number of Iterations')
plt.show()
# print "Means:"
# print mu
# print "Sigma:"
# print sigma
# print "Pi:"
# print pi

# 3 gaussians
# Means:
# [0.2900957302982042, 0.8005439540993955, 0.5916513870330805]
# Sigma:
# [0.08434155226215548, 0.02634409917128631, 0.06774578838274804]
# Pi:
# [0.19413456032914914, 0.322992286175951, 0.4828731534948998]

# 2 gaussians
# Means:
# [0.6666595246807696, 0.2738310337285384]
# Sigma:
# [0.12470142694847133, 0.07567647991226498]
# Pi:
# [0.8317845415932306, 0.16821545840676874]

# 4 gaussians
# Means:
# [0.5899899146973651, 0.26145366722840546, 0.8006749099961208, 0.37700825890821954]
# Sigma:
# [0.0697235845959165, 0.06913159252967965, 0.026229259029779094, 0.02622761920763644]
# Pi:
# [0.4918552418893182, 0.15186233804982632, 0.3218655218208801, 0.03441689823997553]

