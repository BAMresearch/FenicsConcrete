#import numpy as np
import plotly.graph_objects as go
import numpy as np
import json
import chain_analysis_functions as caf
import arviz as az
from probeye.postprocessing.sampling_plots import create_pair_plot
from probeye.postprocessing.sampling_plots import create_posterior_plot
from probeye.postprocessing.sampling_plots import create_trace_plot
import emcee


#########################################################################################################################################
#########################################################################################################################################
#1st Step - Reading data and rearranging it
#########################################################################################################################################
######################################################################################################################################### 

with open('probabilistic_identification/results_reformulatedortho/test_config.json', 'r') as f:
    json_object = json.loads(f.read()) 

#inference_data = az.from_json(json_object.get('MCMC').get('arviz_data_name'))

parameters_list = ["E_2", "E_d", "nu", "sigma"] #["E", "nu", "sigma"]#"E_d", 
chain_data = np.loadtxt(json_object.get('MCMC').get('chain_name'), delimiter=',')
posterior = chain_data.reshape(chain_data.shape[0], chain_data.shape[1]// len(parameters_list), len(parameters_list))
# chain_state/step number, chain_index, parameter_index
posterior =posterior[:,:,:]
#########################################################################################################################################
#########################################################################################################################################
#1st Step - Plotting Data
#########################################################################################################################################
#########################################################################################################################################

###################################################################################
# Auto Correlation Function Vs. Gap Plot
###################################################################################
chain_number = 5

gap = np.arange(json_object.get('MCMC').get('nsteps'))

fig1 = go.Figure()
for i in range(len(parameters_list)):
    fig1.add_trace(go.Scatter(x=gap, y = caf.autocorr_func_1d(posterior[:, chain_number, i]),  
                    mode='lines+markers',
                    name=parameters_list[i]))
#fig.update_layout(yaxis_type = "log")
fig1.update_layout(title="Auto Correlation Function Vs. Gap",
    xaxis_title="Gap",
    yaxis_title="Auto Correlation Function")
fig1.show() 
fig1.write_html(json_object.get('MCMC').get('acf_plot_name'))

###################################################################################
# Auto Correlation Time Vs. Samples or Steps Plot
###################################################################################
#emcee.autocorr.integrated_time(posterior)
act_new = np.empty(len(parameters_list))
for j in range(len(parameters_list)):
    act_new[j] = caf.autocorr_new(posterior[:, :, j])

ESS = 10000/act_new # Effective Sample Size
print('Effective Sample Size: ', ESS)

# Compute the estimators for a few different chain lengths
N = np.exp(np.linspace(np.log(100), np.log(posterior.shape[0]), 20)).astype(int)
act = np.empty((len(N), len(parameters_list)))

for i, n in enumerate(N):
    for j in range(len(parameters_list)):
        act[i,j] = caf.autocorr_new(posterior[:n, :, j])

fig2 = go.Figure()
for i in range(len(parameters_list)):
    fig2.add_trace(go.Scatter(x=N, y = act[:,i],  
                    mode='lines+markers',
                    name=parameters_list[i]))
#fig.update_layout(yaxis_type = "log")
fig2.update_layout(title="Auto Correlation Time Vs. Chain Length",
    xaxis_title="Chain Length",
    yaxis_title="Auto Correlation Time")
fig2.show() 
fig2.write_html(json_object.get('MCMC').get('act_plot_name'))


#####################################################################################
# Plot Mean
#####################################################################################

N = np.exp(np.linspace(np.log(100), np.log(posterior.shape[0]), 20)).astype(int)
mean = np.empty((len(N), len(parameters_list)))
for i, n in enumerate(N):
    for j in range(len(parameters_list)):
        mean[i,j] = np.sum(posterior[:n, :, j])/(posterior[:n, :, j].shape[0]*posterior[:n, :, j].shape[1])


fig3 = go.Figure()
for i in range(len(parameters_list)):
    fig3.add_trace(go.Scatter(x=N, y = mean[:,i],  
                    mode='lines+markers',
                    name=parameters_list[i]))
#fig.update_layout(yaxis_type = "log")
fig3.update_layout(title="Mean Vs. Chain Length",
    xaxis_title="Chain Length",
    yaxis_title="Mean")
fig3.show() 
fig3.write_html(json_object.get('MCMC').get('mean_plot_name'))

#####################################################################################
# Arviz Plots from the sliced inference data
#####################################################################################

chain_length = 10000
total_chains = 20
import xarray as xr

#posterior[:,chain_length,0]

#E = xr.DataArray(posterior[:chain_length,:,0].T, dims=("chain", "draw"), coords={"chain": np.arange(total_chains), "draw": np.arange(chain_length)})
#nu = xr.DataArray(posterior[:chain_length,:,1].T, dims=("chain", "draw"), coords={"chain": np.arange(total_chains), "draw": np.arange(chain_length)})
#sigma = xr.DataArray(posterior[:chain_length,:,2].T, dims=("chain", "draw"), coords={"chain": np.arange(total_chains), "draw": np.arange(chain_length)})
#sliced_posterior = xr.Dataset({"$E$": E, "$\\nu$": nu, "$\sigma_{model}$": sigma})

E_m = xr.DataArray(posterior[:chain_length,:,0].T, dims=("chain", "draw"), coords={"chain": np.arange(total_chains), "draw": np.arange(chain_length)})
E_d = xr.DataArray(posterior[:chain_length,:,1].T, dims=("chain", "draw"), coords={"chain": np.arange(total_chains), "draw": np.arange(chain_length)})
nu = xr.DataArray(posterior[:chain_length,:,2].T, dims=("chain", "draw"), coords={"chain": np.arange(total_chains), "draw": np.arange(chain_length)})
sigma = xr.DataArray(posterior[:chain_length,:,3].T, dims=("chain", "draw"), coords={"chain": np.arange(total_chains), "draw": np.arange(chain_length)})
sliced_posterior = xr.Dataset({"$E_m$": E_m, "$E_d$": E_d, "$\\nu$": nu, "$\sigma_{model}$": sigma})
axes1 = az.plot_pair(sliced_posterior, kind = 'kde', show=False)
axes2 = az.plot_posterior(sliced_posterior, kind = 'kde', show=True)

fig4 = axes1.ravel()[0].figure
fig4.savefig(json_object.get('MCMC').get('az_pair_plot_name'))

fig5 = axes2.ravel()[0].figure
fig5.savefig(json_object.get('MCMC').get('az_trace_plot_name'))

""" E_m = xr.DataArray(inference_data.posterior.data_vars['$E_d$'].values[:,:chain_length], dims=("chain", "draw"), coords={"chain": np.arange(total_chains), "draw": np.arange(chain_length)})
E_d = xr.DataArray(inference_data.posterior.data_vars['$E_m$'].values[:,:chain_length], dims=("chain", "draw"), coords={"chain": np.arange(total_chains), "draw": np.arange(chain_length)})
nu = xr.DataArray(inference_data.posterior.data_vars['$\\nu$'].values[:,:chain_length], dims=("chain", "draw"), coords={"chain": np.arange(total_chains), "draw": np.arange(chain_length)})
sigma = xr.DataArray(inference_data.posterior.data_vars['$\sigma_{model}$'].values[:,:chain_length], dims=("chain", "draw"), coords={"chain": np.arange(total_chains), "draw": np.arange(chain_length)})
sliced_posterior = xr.Dataset({"$E_m$": E_m, "$E_d$": E_d, "$\\nu$": nu, "$\sigma_{model}$": sigma})
az.plot_pair(sliced_posterior, kind = 'kde', show=True) """

#inference_data.posterior.data_vars['$E_d$'] 
#az.plot_autocorr(inference_data, var_names = '$E_m$')
#az.plot_autocorr(inference_data)

#ACT
""" import matplotlib.pyplot as plt
for i in range(len(parameters_list)):
    plt.plot(N, act[:,i], "o-", label=parameters_list[i])
ylim = plt.gca().get_ylim()
#plt.plot(N, N / 50.0, "--k", label=r"$\tau = N/50$")
plt.ylim(ylim)
plt.xlabel("number of samples, $N$")
plt.ylabel(r"$\tau$ estimates")
plt.legend(fontsize=14)
plt.show() """

#Mean
""" import matplotlib.pyplot as plt
for i in range(len(parameters_list)):
    plt.plot(N, mean[:,i], "o-", label=parameters_list[i])
#ylim = plt.gca().get_ylim()
#plt.ylim(ylim)
plt.xlabel("number of samples, $N$")
plt.ylabel(r"$\nu$ estimates")
plt.legend(fontsize=14)
plt.show() """