import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def strain_and_displacement_calculator(_sigma_xx, _E, _nu, _length=1000, _breadth=200):
    strain_xx = _sigma_xx/_E
    strain_yy = -_nu*strain_xx

    disp_x = strain_xx*_length
    disp_y = strain_yy*_breadth

    data_strain = np.array([strain_xx, strain_yy])
    data_disp = np.array([disp_x, disp_y])

    return data_strain, data_disp

#measured_data = np.array([strain_xx+np.random.normal(0, 0.001), strain_yy+np.random.normal(0, 0.001)])

E = 10e3 #in MPa
nu = 0.2

length = 1000 #in mm
breadth = 1000
thickness = 1
force = 5e3 #in Newtons

sigma_xx = force/(breadth*thickness)
measured_data_strain, measured_data_disp = strain_and_displacement_calculator(sigma_xx, E, nu)

# Cost function plot
fig_strain, ax_strain = plt.subplots(subplot_kw={"projection": "3d"})
fig_disp, ax_disp = plt.subplots(subplot_kw={"projection": "3d"})

E_buildup = np.linspace(5e3,20e3,100)
nu_buildup = np.linspace(0.01,0.55,20)
E_buildup, nu_buildup= np.meshgrid(E_buildup, nu_buildup)
cost_func_val_strain = np.zeros((E_buildup.shape[0],E_buildup.shape[1]))
cost_func_val_disp = np.zeros((E_buildup.shape[0],E_buildup.shape[1]))

for i in range(E_buildup.shape[0]):
    for j in range(nu_buildup.shape[1]):
        model_output_strain, model_output_disp = strain_and_displacement_calculator(sigma_xx, E_buildup[i,j], nu_buildup[i,j])

        delta_strain = model_output_strain - measured_data_strain
        cost_func_val_strain[i,j] = np.linalg.norm(delta_strain)

        delta_disp = model_output_disp - measured_data_disp
        cost_func_val_disp[i,j] = np.linalg.norm(delta_disp)
        print(np.linalg.norm(delta_disp[1]), np.linalg.norm(delta_disp[0]), cost_func_val_disp[i,j] )
        
# Plot the surface
surf_strain = ax_strain.plot_surface(E_buildup, nu_buildup, cost_func_val_strain, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax_strain.set_xlabel('E')
ax_strain.set_ylabel('nu')
ax_strain.set_zlabel('Cost function value')
fig_strain.colorbar(surf_strain, shrink=0.5, aspect=5)
ax_strain.set_title('From strain data')

surf_disp = ax_disp.plot_surface(E_buildup, nu_buildup, cost_func_val_disp, cmap=cm.coolwarm,
                   linewidth=0, antialiased=False)
ax_disp.set_xlabel('E')
ax_disp.set_ylabel('nu')
ax_disp.set_zlabel('Cost function value')
fig_disp.colorbar(surf_disp, shrink=0.5, aspect=5)
ax_disp.set_title('From displacement data')

plt.show()

""" 
#N/m², kg, m, sec
rho = 7750 #kg/m^3
L=1 #m
W=0.2 #m
delta = L/W
E=210e9
max_deflection_m = (1.5*rho*9.81*delta**2*L**2)/E
print(max_deflection_m)

#N/mm², kg, mm, sec
rho = 7750e-9 #kg/mm^3
L=1000 #mm
W=200 #mm
delta = L/W
E=210e3
max_deflection_mm = (1e-3*1.5*rho*9.81e3*delta**2*L**2)/E
print(max_deflection_mm)

a=10**-9
print(a) """

