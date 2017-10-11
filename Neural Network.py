######################################################################
# Importing Required Packages
######################################################################
import numpy
import itertools
import random
from random import shuffle
######################################################################
# Defining  Functions
######################################################################

def act_pot(w,x):
	if len(w) == len(x):
		sum = 0
		for i in range(len(w)): 
			sum = sum + (w[i]*x[i])
		return sum
	else:
		return "error"
# Logistic Function
def out(v):
	return 1.0/(1+numpy.exp(-v))
def dif_out(v):
	return (1-out(v))*(out(v))

	
def error(d,y):
	return d-y



######################################################################
# Initialize inputs
######################################################################

every_row_error = []
learning_rate_list = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]
num_epoch = 14000000
momentum = 0.9

#Initialize inputs
input = []
for seq in itertools.product("01", repeat=4):
	temp = [1]
	for i in seq:
		temp.append(int(i))
	input.append(temp)

# input = input[0:3]
print (input)
des_out= []
for i in input:
	if sum(i)%2 == 0:
		des_out.append(1)
	else: 
		des_out.append(0)
final_epochs = []



######################################################################
# Start Learning Process
######################################################################
for learning_rate in learning_rate_list:
	#Initialize neural network
	random.seed(80)
	layer1_weights = []
	layer1_prev_weights = []
	for i in range(4):
		temp = []
		prev_temp = []
		for j in range(5):
			temp.append(random.uniform(-1,1))
			# temp.append(0)
			prev_temp.append(0)
		layer1_weights.append(temp)
		layer1_prev_weights.append(prev_temp)
	
	layer2_weights = [random.uniform(-1,1),random.uniform(-1,1),random.uniform(-1,1),random.uniform(-1,1),random.uniform(-1,1)]
	# layer2_weights = [0,0,0,0,0]
	layer2_prev_weights = [0,0,0,0,0]
	
	for epoch in range(num_epoch): # Looping for each epoch
		print ("Epoch :" + str(epoch))
		for row in range(len(input)): # Looping for each row in dataset
			print ("Row ", row)
			hidden_outputs = [1]
			for j in layer1_weights:
				activation = act_pot(j,input[row])
				act_out = out(activation)
				hidden_outputs.append(act_out)
			
			# Calculating output perceptron outputs
			activation = act_pot(layer2_weights,hidden_outputs)
			act_out = out(activation)
			# Starting Back Propogation of error
			final_error = error(des_out[row], act_out)
			diff_act_out = dif_out(activation)
			delta_l2_weight = []
			for i in range(len(hidden_outputs)): # calculating change of weights with momentum for output layer
				delta_l2_weight.append((learning_rate * final_error * diff_act_out * hidden_outputs[i]) + (momentum * layer2_prev_weights[i])) 
			delta_l1_weight = []
			for j in range(len(layer1_weights)):
				activation = act_pot(layer1_weights[j],input[row])
				diff_hid_out = dif_out(activation)
				weight_to_output = layer2_weights[j+1] # Ignoring the bias term for multiplication for hidden layer
				temp = []
				for item in range(len(input[row])):# calculating change of weights with momentum for hidden layer
					temp.append((learning_rate * diff_hid_out * final_error * diff_act_out * weight_to_output * input[row][item]) + (momentum * layer1_prev_weights[j][item]))
				delta_l1_weight.append(temp)
			layer2_weights = [x + y for x, y in zip(layer2_weights, delta_l2_weight)] 
			temp = []
			for i in range(len(layer1_weights)):
				temp.append([x + y for x, y in zip(layer1_weights[i], delta_l1_weight[i])])
			layer1_weights = temp
			layer1_prev_weights = list(delta_l1_weight)	
			layer2_prev_weights = list(delta_l2_weight)	

			# Checking errors for all rows
			total_error = []
			for row in range(len(input)):
				hidden_outputs = [1]
				for j in layer1_weights:
					activation = act_pot(j,input[row])
					act_out = out(activation)
					hidden_outputs.append(act_out)
				# Calculating output perceptron outputs
				activation = act_pot(layer2_weights,hidden_outputs)
				act_out = out(activation)
				total_error.append(error(des_out[row], act_out))
			error_tracker = 0
			sum = 0
			for i in range(len(total_error)):
				sum = sum + (total_error[i]*total_error[i])
				if abs(total_error[i])>0.05:
					error_tracker = error_tracker + 1
			every_row_error.append(sum/len(total_error))
			if error_tracker == 0: 
				print ("Convergence Reached")
				final_epochs.append([learning_rate, epoch])
				break
		if error_tracker == 0: 
			break
	if error_tracker == 0: 
		break
print(final_epochs)
