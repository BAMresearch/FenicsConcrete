import plotly.graph_objects as go
import numpy as np
inferred_parameters = np.loadtxt('Inferred Parameters Vs. Sparsity Factor (1% Noise, Sparse Data)_RelativeError_final.csv', dtype=float, delimiter=",")
fig1 = go.Figure()
inferred_parameters_name = ['E_m', 'E_d', 'nu', 'G_12', 'K_x', 'K_y']

t = [0.5, 0.4, 0.3, 0.2, 0.1, 0.]
for i in range(inferred_parameters.shape[1]):
        fig1.add_trace(go.Scatter(x=t, y=[x for x in inferred_parameters[:,i]],
                        mode='markers',
                        name=inferred_parameters_name[i]))
fig1.add_hline(y=0.1, line_dash="dot")
fig1.update_xaxes(autorange="reversed")
#fig1.update_xaxes(type="log")
fig1.update_yaxes(type="log")
fig1.update_layout(title="Inferred Parameters Vs. Sparsity Factor (1% Noise, Sparse Data)",
    xaxis_title="Sparsity Factor",
    yaxis_title="Inferred Parameters (Log Scale)",
    legend_title="Parameters",)

fig1.update_traces(marker=dict(size=11,
                              line=dict(width=2,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))

fig1.show()
fig1.write_html('Inferred Parameters Vs. Sparsity Factor (1% Noise, Sparse Data)'+'.html')
