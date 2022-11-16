from mailbox import MH
from ossaudiodev import SNDCTL_SEQ_RESETSAMPLES
from pickletools import int4
import numpy as np
import matplotlib.pyplot as plt

#Original target density plot ---------------------------------------------------------------------------------------------------------------
x1=np.linspace(-1,1,60)
x2=np.linspace(-1,1,60)


X1, X2 = np.meshgrid(x1,x2)
log_target_density = -10*(X1**2-X2)**2 - (X2-0.25)**4

import plotly.express as plx
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Surface(z=log_target_density, x=x1, y=x2, contours={"z": {"show": True, "start": -0.5, "end": 0.5, "size": 0.1, "project":{"z":True}}},opacity=0.5))

""" from matplotlib import cm
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X1, X2, target_density, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

plt.show() """

#fig.update_traces(contours_z=dict(show=True, usecolormap=True,
#                                  highlightcolor="limegreen", project_z=True, start= -2, end= 0, size= 0.25),)

fig.update_layout(
    title="Metropolis-Hastings Chain",
    xaxis_title="x",
    yaxis_title="Probability density")

#Metropolis Hastings Implementation ----------------------------------------------------------------------------------------------------

import math
def log_target_density(x):
    return -10*(x[0]**2-x[1])**2 - (x[1]-0.25)**4


def metropolis_hastings(startVector, nsamples, ndim, sigma):
    MH_chain = np.zeros((nsamples, ndim))
    MH_chain[0,:] = startVector 
    counter = 0
    acception_count = 0
    target_density_at_x_n = log_target_density(startVector)

    for counter in range(nsamples-1):
        proposal = MH_chain[counter,:] + np.random.normal(0, sigma**2,2)
 
        log_alpha = log_target_density(proposal) - target_density_at_x_n

        if math.log(np.random.uniform(0,1)) < log_alpha:
            MH_chain[counter+1,:] =  proposal
            target_density_at_x_n = log_target_density(proposal)
            acception_count += 1
        else:
            MH_chain[counter+1,:] = MH_chain[counter,:]

    return MH_chain, acception_count

num_samples = 5000
sigma = 0.4
Metropolis_chain, accepted_proposals = metropolis_hastings(np.array([3,5]), num_samples, 2, sigma)
print(accepted_proposals)

#Plot Markov Chain ---------------------------------------------------------------------------------------------------------------------------
fig.add_trace(go.Scatter3d(
    x=Metropolis_chain[:,0], y=Metropolis_chain[:,1], z=np.zeros((num_samples,)),
    marker=dict(
        size=4,
        #color=z,
        colorscale='Viridis',
    ),
    line=dict(
        color='darkblue',
        width=2
    )
))

fig.show()

# Convergence Measures and their plots ------------------------------------------------------------------------------------------------------

def ergodic_mean(chain):
    mean1 = np.divide(np.cumsum(chain, axis=0)[:,0],np.arange(1,chain.shape[0]+1))
    mean2 = np.divide(np.cumsum(chain, axis=0)[:,1],np.arange(1,chain.shape[0]+1))
    return mean1, mean2

def autocovariance(chain, nsamples, ndim):
    gap = nsamples-1
    autocov = np.zeros((gap, ndim))
    mean = np.sum(chain,axis=0)/nsamples
    variance = np.sum(np.square(chain - mean),axis=0)/nsamples
    #print(variance)
    for j in range(1,gap+1):
        autocov[j-1,:] = np.divide(np.sum(np.multiply((chain[:nsamples-j] - mean),(chain[j::1] - mean)), axis=0),((nsamples-j)*variance))
        #autocov[j-1,1] = np.divide(np.sum(np.multiply((chain[:nsamples-gap] -mean),(chain[gap::1] - mean)), axis=0),((nsamples-j)*variance))
    return autocov

import matplotlib.pyplot as plt 

""" 
ergo_mean_x1 , ergo_mean_x2 = ergodic_mean(Metropolis_chain)
fig, (ax1,ax2) = plt.subplots(1,2)
ax1.plot(np.arange(1,num_samples+1, 1, dtype=None),ergo_mean_x1)
ax1.set_xlabel('Samples') # X-Label
ax1.set_ylabel('Ergodic Mean x1') # Y-Label


ax2.plot(np.arange(1,num_samples+1, 1, dtype=None),ergo_mean_x2)
ax2.set_xlabel('Samples') # X-Label
ax2.set_ylabel('Ergodic Mean x2') # Y-Label
plt.show() """


auto_covariance = autocovariance(Metropolis_chain, num_samples, 2)

fig, (ax1,ax2) = plt.subplots(1,2)
ax1.plot(np.arange(1,num_samples, 1, dtype=None),auto_covariance[:,0])
ax1.set_xlabel('gap') # X-Label
ax1.set_ylabel('Autocovariance x1') # Y-Label


ax2.plot(np.arange(1,num_samples, 1, dtype=None),auto_covariance[:,1])
ax2.set_xlabel('gap') # X-Label
ax2.set_ylabel('Autocovariance x2') # Y-Label
plt.show()
