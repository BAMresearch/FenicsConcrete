import numpy as np
sp_factor = [0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4] #
inferred_parameters = np.zeros((6, 6))
inferred_parameters[0,:] = [0.4199, 9.83e-6, 0.278, 0.326, 0.00238, 0.00226]
inferred_parameters[1,:] = [0.4199, 0.00234, 0.28, 0.326, 2e-6, 3e-7]
inferred_parameters[2,:] = [0.4199, 0.00234, 0.28, 0.326, 2e-6, 3e-7]
inferred_parameters[3,:] = [0.4199, 0.00234, 0.28, 0.326, 2e-6, 3e-7]
inferred_parameters[4,:] = [0.4199, 0.00234, 0.28, 0.326, 2e-6, 3e-7]
inferred_parameters[5,:] = [0.4199, 0.00234, 0.28, 0.326, 2e-6, 3e-7]

import plotly.graph_objects as go
fig1 = go.Figure()
inferred_parameters_name = ['E_m', 'E_d', 'nu', 'G_12', 'k_x', 'k_y']

for i in range(inferred_parameters.shape[1]):
        print(sp_factor, inferred_parameters[:,i])
        fig1.add_trace(go.Scatter(x=sp_factor, y=[x for x in inferred_parameters[:,i]],
                        mode='lines+markers',
                        name=inferred_parameters_name[i]))
fig1.add_hline(y=0.1, line_dash="dot")
fig1.update_xaxes(type="log")
fig1.update_yaxes(type="log")

#fig1.update_layout(yaxis_type = "log")

fig1.update_traces(marker=dict(size=11,
                              line=dict(width=2,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='lines+markers'))
fig1.show()
fig1.write_html('Inferred Parameters Vs. Sparsity Factor'+'.html')

np.savetxt("foo.csv", inferred_parameters, delimiter=",")

#arr = np.loadtxt("foo.csv",
#                 delimiter=",", dtype=float)