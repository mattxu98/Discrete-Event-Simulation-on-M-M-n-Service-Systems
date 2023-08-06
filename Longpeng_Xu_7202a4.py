'''
Project: Discrete Event Simulation on M/M/n Service Systems
Author: Longpeng Xu (https://github.com/mattxu98)
Date: 20230609
'''
# Data pipeline
import pandas as pd
interarr = pd.read_csv('data.csv', header=0, usecols=['inter_arrival_time'])
service = pd.read_csv('data.csv', header=0, usecols=['service_time'])



# Create sampler with replacement
import numpy as np

class sampler:
    def __init__(self, population):
        self.population = population
        self.remaining = list(self.population)

    def sample(self, n):
		if n > len(self.remaining):
            raise ValueError("Cannot sample more elements than remaining")
        sample = np.random.choice(self.remaining, size=n, replace=True)
        #self.remaining = [el for el in self.remaining if el not in sample]
        #print(f"The number of remaining samples is {len(self.remaining)}")
        return sample

# Create the instances
sampler_interarr = sampler(interarr.values.reshape(1,-1)[0])
sampler_service = sampler(service.values.reshape(1,-1)[0])



# Simulation
def DES(T, n, num_ser, burnin):
    '''
    T:       stop time
    n:  the number of batches
    num_ser: the number of servers
    burnin:  the first burnin% of the samples to be discarded
    '''
    arr_times = [[0.0000, 0]]
    end_times = []
    wait_periods = []
    pct_batches = []
    mean_batches = []
    step = 0

	# Continue simulation until T is reached
    while arr_times[-1][0] < T:

        # Compute the arrival time of a new customer via a sample from ECDF
        if step == 0:
            arr_time = 0.0000
        else:
            arr_time = arr_times[-1][0] + float(sampler_interarr.sample(1))

        # Enter into the shortest queue
        # queue_len is # customers still in the queue at the new customer's arrival
        # np.random.choice is used in case of multiple queues have the shortest length
        queue_len = np.array([])
        for i in range(1, num_ser + 1):
            queue_len = np.append(queue_len, len([j for j in end_times if j[1] == i and arr_time - j[0] < 0]))
        shortest = list(np.argwhere(queue_len == min(queue_len)).reshape(1,-1)[0])
        which_ser = np.random.choice(shortest) + 1

        # Append the arrival time to arr_times
        if step == 0:
            arr_times[0][1] = which_ser
        else:
            arr_times += [[arr_time, which_ser]]

        # Compute the waiting period
        # the sum of the differences between arr_time (new customer) and end_time of each customers still in the shortest queue  
        wait_period = abs(sum([arr_time - k[0] for k in end_times if k[1] == which_ser and arr_time - k[0] < 0]))

        # Sample the service period
        ser_period = float(sampler_service.sample(1))

        # Compute the end time
        end_time = arr_time + wait_period + ser_period
        end_times += [[end_time, which_ser]]

        # Append wait_period of current customer to wait_periods
        wait_periods += [wait_period]
        #print(f"Customer who chose server {which_ser}, with arrival time {round(arr_time,4)}, wait period {round(wait_period,4)}, service period {round(ser_period,4)}, end time {round(end_time,4)}.")
        step += 1

    # Compute the border index in arr_times for each batch, after burn-in
    batch_cutoff_time = np.linspace(int(T * burnin), T, n+1)
    batch_cutoff_ind = [int(np.argwhere(np.array(arr_times)[:, 0] >= k)[0]) for k in batch_cutoff_time]

    # Compute the percentage of customers waiting < 8 min and the average time waiting for each batch
    for i in range(n):
        start, end = batch_cutoff_ind[i], batch_cutoff_ind[i+1]
        pct_batches += [round(len([m for m in wait_periods[start:end] if m<=8]) / len(wait_periods[start:end]), 6)]
        mean_batches += [round(np.mean(wait_periods[start:end]), 6)]

    #return pct_batches, np.mean(pct_batches), mean_batches, np.mean(mean_batches)
    return np.mean(pct_batches), np.mean(mean_batches)




# One group of simulations
for i in range(5,11):
	mean_pct_batches, mean_mean_batches = DES(T=3000, n=50, num_ser=i, burnin=0.3)
	print(f"With {i} desks, {mean_pct_batches*100}% customers wait < 8 min (averaged across the batches), a customer waits for {mean_mean_batches} mins (averaged across the batches).")
  
# Another 20 groups of simulations
for j in range(20):
  for i in range(7,10):
    mean_pct_batches, mean_mean_batches = DES(T=3000, n=50, num_ser=i, burnin=0.3)
    print(f"With {i} desks, {mean_pct_batches*100}% customers wait < 8 min (averaged across the batches), a customer waits for {mean_mean_batches} mins (averaged across the batches).")
  print("=====================================================================")
  


# Plot flow chart for project dynamics
from graphviz import Digraph

dot = Digraph()
dot.attr(rankdir='LR') # horizontal arrangement

# Create nodes
dot.node('1', 'Data Pipeline', shape='box')
dot.node('2', 'Define a sampler', shape='box')
dot.node('3', 'Arrival time', shape='box')
dot.node('4', 'Select the shortest\nqueue', shape='box')
dot.node('5', 'Waiting period', shape='box')
dot.node('6', 'Service period', shape='box')
dot.node('7', 'End time', shape='box')
dot.node('8', 'Drop burnin,\nsplit to batches w.r.t time', shape='box')
dot.node('9', 'Compute results\nw.r.t. batches', shape='box')

# Create edges
dot.edges(['12', '23', '34', '45', '56', '67', '78', '89'])

# Save the flow chart
dot.render('7202a4fig0', view=True)

# Figure 1
import matplotlib.pyplot as plt
plt.rcParams.update({'text.usetex': True, 'font.family': 'serif'})
plt.figure(figsize=(4,3), dpi=600, facecolor='white')

x = range(1,11)
y = (0,0,0,0,0,0,0, 0.990173, 0.996075, 0.998493)
plt.plot(x,y)
plt.xlabel('\# of service desks')
plt.ylabel("\% of customers waiting $\le$ 8 mins", fontsize=8)
plt.savefig('7202a4fig1.pdf')
plt.show()

# Figure 2
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(7, 3), tight_layout=True, dpi=600, facecolor='white')
axes[0].hist(pct_batches)
axes[0].set(xlabel="\% of customers waiting $\le$ 8 mins ($\cdot 10^2$)")
axes[1].hist(mean_batches)
axes[1].set(xlabel="minutes waiting")
plt.savefig('7202a4fig2.pdf')
plt.show()