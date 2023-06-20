import numpy as np
import plotly.graph_objects as go
import json

def next_pow_two(n):
    i = 1
    while i < n:
        i = i << 1
    return i


def autocorr_func_1d(x, norm=True):
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))

    
    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
    acf = acf / (len(x)*np.ones(len(x)) - np.arange(len(x)))
    #acf /= 4 * n

    # Optionally normalize
    if norm:
        acf /= acf[0]

    return acf

with open('test_config.json', 'r') as f:
    json_object = json.loads(f.read()) 

parameters_list = ["E_m", "E_d", "nu", "sigma"]
chain_data = np.loadtxt(json_object.get('MCMC').get('chain_name'), delimiter=',')
posterior = chain_data.reshape(chain_data.shape[0], chain_data.shape[1]// len(parameters_list), len(parameters_list))
# chain_state/step number, chain_index, parameter_index

gap = np.arange(json_object.get('MCMC').get('nsteps'))
fig = go.Figure()
chain_number = 5
for i in range(len(parameters_list)-1):
    fig.add_trace(go.Scatter(x=gap, y=autocorr_func_1d(posterior[:, chain_number, i]),  
                    mode='lines+markers',
                    name=parameters_list[i]))
#fig.update_layout(yaxis_type = "log")
fig.update_layout(title="Auto Correlation Function Vs. Gap",
    xaxis_title="Gap",
    yaxis_title="Auto Correlation Function")
fig.show() 
fig.write_html('probabilistic_identification/extraplot.html')

import emcee
emcee.autocorr.integrated_time(posterior)

# Compute the estimators for a few different chain lengths

N = np.exp(np.linspace(np.log(100), np.log(posterior.shape[0]), 10)).astype(int)

# Automated windowing procedure following Sokal (1989)
def auto_window(taus, c):
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1

def autocorr_new(y, c=5.0):
    total_chain_steps = y.shape[0]
    number_of_chains = y.shape[1]
    f = np.zeros(total_chain_steps)
    for i in range(number_of_chains): #for each chain
        f += autocorr_func_1d(y[:,i])
    f /= number_of_chains
    taus = 2.0 * np.cumsum(f) + 1.0
    window = auto_window(taus, c)
    return taus[window]

#def tau_calculation(y):
#    return 1+ 2*np.sum(autocorr_func_1d(y)[:500])

new = np.empty((len(N), len(parameters_list)))

for i, n in enumerate(N):
    for j in range(len(parameters_list)):
        new[i,j] = autocorr_new(posterior[:n, :, j])


import matplotlib.pyplot as plt
for i in range(len(parameters_list)):
    plt.plot(N, new[:,i], "o-", label=parameters_list[i])
ylim = plt.gca().get_ylim()
#plt.plot(N, N / 50.0, "--k", label=r"$\tau = N/50$")
plt.ylim(ylim)
plt.xlabel("number of samples, $N$")
plt.ylabel(r"$\tau$ estimates")
plt.legend(fontsize=14);
plt.show()
