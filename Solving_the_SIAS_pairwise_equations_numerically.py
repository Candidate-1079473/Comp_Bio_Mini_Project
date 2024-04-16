#Solving the SIAS pairwise equations numerically;
import pickle
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


#the max time, make sure this is the same as in the stochastic_simulations_SIAS_using_EoN file
t_max = 30

# Function defining the system of 9 ODEs
def odesystem(t, y, b_d, g_d, a_d, s_dc, s_cd, b_c, g_c, a_c, kappa):
    [S, I, A, SS, SI, SA, II, IA, AA] = y
    
    phi_c = g_c + b_c + a_c + a_d + s_cd
    phi_d = g_d + b_d + a_d + a_c + s_dc
    
    dSdt = -b_d*SI - b_c*SA + g_d*I + g_c*A - (a_d + a_c)*S
    dIdt = b_d * SI - g_d * I + a_d * S - s_dc * I + s_cd * A
    dAdt = b_c * SA - g_c * A + a_c * S + s_dc * I - s_cd * A
    dSSdt = -2 * (a_c + a_d) * SS - 2 * b_c * kappa * (SS * SI)/S - 2 * b_d * kappa * (SS * SA)/S + g_d * SI + g_c * SA
    dSIdt = -phi_d * SI + g_d * II + g_c * IA + (b_d * kappa * SI / S) * (SS - SI) + a_d * SS + s_cd * SA
    dSAdt = -phi_c * SA + g_c * AA + g_d * IA + (b_c * kappa * SA / S) * (SS - SA) + a_c * SS + s_dc * SI
    dIIdt = 2 * b_d * SI * (1 + kappa * (SI / S)) - 2 * (g_d + s_dc) * II + 2 * s_cd * IA + 2 * a_d * SI
    dIAdt = kappa * (b_d + b_c) * (SI * SA)/S - (g_c + g_d) * IA - (s_cd + s_dc) * IA + a_c * SI + a_d * SA
    dAAdt = 2 * b_c * SA * (1 + kappa * (SA / S)) - 2 * (g_c + s_cd) * AA + 2 * s_dc * IA + 2 * a_c * SA
    
    return [dSdt, dIdt, dAdt, dSSdt, dSIdt, dSAdt, dIIdt, dIAdt, dAAdt]

# Initial conditions, remember S + I + A = N, the pairs sum to nN
#
initial_conditions = [4739, 0.0, 0.0, 6492.43, 0.0, 0.0, 0.0, 0.0, 0.0] # maybe need to start

# Time span
t_span = (0, t_max)


#Let's suppose that n, the mean number of contacts is say, 1.37 so kappa is 0.37/1.37 ~ 0.27
#b_c = 0.02, b_d = 0.04, g_c =0.088, g_d= 0.13, a_c = 0.18, a_d = 0.04, s_{cd} = 0.009,$ and $s_{dc} = 0.07
# N = 4739
#%We will thus find the steady state of our SIAS system (numerically) and compare this to the proportion listed above.
#We assume the total population, denoted by $N$, is constant and use the value of $N= 4739$, which is the approximate size of the FHS in both the first and second examinations. Therefore the initial condition, $$(S_0, I_0, A_0) = (3080.35, 426.51, 1232.14),$$
parameters = (0.02, 0.04, 0.088, 0.13, 0.18, 0.04, 0.009, 0.07, 0.27)  # Example parameters, replace with your own values
b_c, b_d, g_c, g_d, a_c, a_d, s_cd, s_dc, kappa = parameters

# Solve the system of ODEs
sol = solve_ivp(
    fun=lambda t, y: odesystem(t, y, b_d, g_d, a_d, s_dc, s_cd, b_c, g_c, a_c, kappa),
    t_span=t_span,
    y0=initial_conditions,
    t_eval=np.linspace(0, t_max, 1000)
)

# Plot the results
plt.figure(figsize=(10, 6))
#for i in range(3, 9):
#    plt.plot(sol.t, sol.y[i], label=f'[{["S", "I", "A", "SS", "SI", "SA", "II", "IA", "AA"][i]}]', linewidth=3)
#plt.ylabel('Variable Value', fontsize=24)

for i in range(3):
    plt.plot(sol.t, sol.y[i], label=f'[{["S", "I", "A"][i]}]', linewidth= 4)
    print(sol.y[i][-1]/4739)
plt.ylabel('Subpopulation Size', fontsize=24)



# Saving data to a pickle file
with open("data.pkl", "wb") as pickle_file:
    pickle.dump(sol.y, pickle_file)

plt.xlabel('Time', fontsize=24)

plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.legend(fontsize=20)
plt.show()

