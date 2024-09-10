### Define params
import numpy as np

mu = 0.5 # radio-protection for non-proliferative cells
eta = 0.1376 # radio-protection for GSCs
n = 10 # number of cell divisions of PCs before they become TCs
k = 1 # carrying capacity
delta_s = 0.001 # death rate of GSCs, T_{1/2}=693 days
delta_v =  0.01*np.ones(n) # death rate of TCs, T_{1/2}=69.3 days
delta_v[n-1] = 0.1 # death rate of TCs, T_{1/2}=6.93 days
delta_m = 0.25 # death rate of AMSCs, T_{1/2}= days
delta_b = 0.5 # decay rate of BMP4, T_{1/2}=1.39 days
u_s = 0.5 # uptake rate of BMP4 by GSCs
C = 0.5 # release rate of BMP4 by AMSCs
lam = 1 # max probability rate in detection & death
Ps_max = 0.56 # Max probability of self renewal
Ps_min = 0 # Min probability of self renewal
psi = 2 # fix sensitivity
mv_rho_scale = 2/365 # TC proliferation rate mv = mv_rho_scale*rho
ms_mv_scale = 1 # GSC proliferation rate = ms_mv_scale*mv
detect_threshold = 0.2
death_threshold = 0.7
detection_sensitivity = 100
death_sensitivity = 20
# Radiotherapy parameters according to 10.1088/0031-9155/55/12/001 paper rho is linearly related to alpha by factor of 0.005
alpha_rho_scale = 0.005
# RT schedule params
resection_to_RT_delay = 30 # days from resection to RT start
t_RT_interval = 1 # time in days between RT doses
t_RT_cycle = 7 # time in days between cycles
n_RT_repeat = 5 # number of doses on RT (remainder is off)
n_RT_cycles = 6 # number of cycles to repeat
t_RT_wait = t_RT_cycle - t_RT_interval*n_RT_repeat # time to wait for next cycle
m_fixed = 10.1762 
m_pulse_init = 30.3315
m_init = 200 # m_init = 20 to get same levels as in assay
resect_fraction = 0.917 
sigma = 0.01 # sd of truncated normal for psi values
B_background = 7.86 # background BMP4 level