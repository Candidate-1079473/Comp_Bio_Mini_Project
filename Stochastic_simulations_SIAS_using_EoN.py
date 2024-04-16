import EoN
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
import random
import pickle 
import numpy as np
#import seaborn as sns

# Apply the default theme
#sns.set_theme()
plt.style.use('default')
N = 4739
#G3 = nx.watts_strogatz_graph(N, 2, 0)


#NB: 0.00028909052*4739 = 1.37
G = nx.fast_gnp_random_graph(N, 0.00028909052)

parameters = (0.02, 0.04, 0.088, 0.13, 0.18, 0.04, 0.009, 0.07)
b_c, b_d, g_c, g_d, a_c, a_d, s_cd, s_dc = parameters

t_max = 30

H = nx.DiGraph()
H.add_node('S')
H.add_edge('S', 'I', rate = a_d)
H.add_edge('S', 'A', rate = a_c)
H.add_edge('I', 'S', rate = g_d)
H.add_edge('A', 'S', rate = g_c)
H.add_edge('I', 'A', rate = s_dc)
H.add_edge('A', 'I', rate = s_cd)


J = nx.DiGraph()
J.add_edge(('I', 'S'), ('I', 'I'), rate = b_d)
J.add_edge(('A', 'S'), ('A', 'A'), rate = b_c)


IC = defaultdict(lambda: 'S')


return_statuses = ('S', 'I', 'A')

t, S, I, A = EoN.Gillespie_simple_contagion(G, H, J, IC, return_statuses,
                                        tmax = t_max)#float('Inf'))

#now save the simulation data
with open("simulation_data.pkl", "wb") as pickle_file:
    pickle.dump([t, S, I, A], pickle_file)
    
    
#with open("simulation_data.pkl", "rb") as pickle_file:
#    Pick = pickle.load(pickle_file)

#t, S, I, A = Pick[0], Pick[1], Pick[2], Pick[3] 

plt.figure(figsize=(10, 6))
#Now plot the simulation
plt.plot(t, S)
plt.plot(t, I)
plt.plot(t, A)

#Now load data from Pairwise ODEs
# Reading data from the pickle file
with open("data.pkl", "rb") as pickle_file:
    Pairwise_y = pickle.load(pickle_file)

Pairwise_S, Pairwise_I, Pairwise_A = Pairwise_y[0], Pairwise_y[1], Pairwise_y[2]

print(len(Pairwise_S))
plt.plot(np.linspace(0,t_max, 1000), Pairwise_S, label = 'Neutral', color = 'cornflowerblue', linewidth=4)
plt.plot(np.linspace(0,t_max, 1000), Pairwise_I, label = 'Unhappy', color = 'gold', linewidth=4)
plt.plot(np.linspace(0,t_max, 1000), Pairwise_A, label = 'Happy', color = 'lightgreen', linewidth=4)

#Now overlay the simulation
plt.plot(t, S, color = 'blue', linewidth=3)
plt.plot(t, I, color = 'orange', linewidth=3)
plt.plot(t, A, color = 'darkgreen', linewidth=3)


plt.xlabel('Time', fontsize=24)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.legend(fontsize=20)


plt.savefig('SIAS_Stochastic_Simulation.png')

print(S[-1]/4739, I[-1]/4739, A[-1]/4739)