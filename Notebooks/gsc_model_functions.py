from numba import jit
import numpy as np
import scipy.stats as stats
from scipy.stats import lognorm

import matplotlib.pyplot as plt
import pandas as pd
from sksurv.compare import compare_survival
from sksurv.nonparametric import kaplan_meier_estimator

@jit
def gsc_model_dudt(u, n, Ps,k,ms,mv,delta_s,delta_v):
    
    dudt = np.zeros(n+1)
    N_ = np.sum(u)
    # u = [s, v1, v2, v3, ..., vn]
    dudt[0] = (2*Ps - 1) * ms * u[0] * (1 - (N_)/k) - delta_s*u[0]
    dudt[1] = 2*(1 - Ps) * ms * u[0] * (1 - (N_)/k) - mv[0]*u[1]*(1 - (N_)/k) - delta_v[0]*u[1]
    # NOTE: 2:n indexes elements 2, 3, 4, ... n-1
    dudt[2:n] = 2*mv[0:n-2]*u[1:n-1]*(1 - (N_)/k) - mv[1:n-1]*u[2:n]*(1 - (N_)/k) - delta_v[1:n-1]*u[2:n]
    dudt[n] = 2*mv[n-2]*u[n-1]*(1 - (N_)/k) - delta_v[n-1]*u[n]
    return dudt

@jit
# alpha_i relates v_i to v_{i-1}, alpha_i*v_i = v_{i-1}
# input: 
# N is a scalar, vector or matrix of values of total population size
# mv is a (nx1) vector of proliferation rates of v1,v2,...,vn
# k is a scalar carrying capacity
# delta_v is a (nx1) vector of decay rates of v1,v2,...,vn
# i is a positive integer generation number
#
# output: 
# alpha_i is the same shape as N, values of alpha_i for each value of N
def alpha_i(N,mv,k,delta_v,i):
    return 2*mv[i-1]*(1-N/k)/(delta_v[i]+mv[i]*(1-N/k))

@jit
# alpha_1 defines v1 in terms of v2,...,vn
# input: 
# N is a scalar, vector or matrix of values of total population size
# mv is a (nx1) vector of proliferation rates of v1,v2,...,vn
# k is a scalar carrying capacity
# delta_v is a (nx1) vector of decay rates of v1,v2,...,vn
# n is a positive integer number of generations
#
# output: 
# alpha_1 is the same shape as N, values of alpha_1 for each value of N
def alpha_1(N,mv,k,delta_v,n):
    denominator = np.ones_like(N)
    alpha_product = np.ones_like(N)
    # accumulate n-1 terms in denominator
    for i in range(1,n):
      alpha_product *= alpha_i(N,mv,k,delta_v,i)
      denominator += alpha_product
    return 1/denominator

@jit
# S_prolif defines total V proliferation rate 
# input: 
# N is a scalar, vector or matrix of values of total population size
# mv is a (nx1) vector of proliferation rates of v1,v2,...,vn
# k is a scalar carrying capacity
# delta_v is a (nx1) vector of decay rates of v1,v2,...,vn
# n is a positive integer number of generations
#
# output: 
# S_prolif is the same shape as N, values of S_prolif for each value of N
def S_prolif(N,mv,k,delta_v,n):
    # first term in numerator and denominator
    alpha_product = np.ones_like(N)
    numerator = mv[0]*np.ones_like(N)
    denominator = np.ones_like(N)
    # accumulate n-2 terms in numerator and denominator
    for i in range(1,n-1):
      alpha_product *= alpha_i(N,mv,k,delta_v,i)
      numerator += mv[i]*alpha_product
      denominator += alpha_product
    # add final term to denominator
    alpha_product *= alpha_i(N,mv,k,delta_v,n-1)
    denominator += alpha_product
    return numerator/denominator

@jit
# input: 
# S_death defines total V proliferation rate 
# N is a scalar, vector or matrix of values of total population size
# mv is a (nx1) vector of proliferation rates of v1,v2,...,vn
# k is a scalar carrying capacity
# delta_v is a (nx1) vector of decay rates of v1,v2,...,vn
# n is a positive integer number of generations
#
# output: 
# S_death is the same shape as N, values of S_death for each value of N
def S_death(N,mv,k,delta_v,n):
    # first term in numerator and denominator
    alpha_product = np.ones_like(N)
    numerator = delta_v[0]*np.ones_like(N)
    denominator = np.ones_like(N)
    # accumulate n-1 terms in numerator and denominator
    for i in range(1,n):
      alpha_product *= alpha_i(N,mv,k,delta_v,i)
      numerator += delta_v[i]*alpha_product
      denominator += alpha_product
    return numerator/denominator

@jit
def gsc_model_reduced_dudt(u, n, Ps,k,ms,mv,delta_s,delta_v):
    
    dudt = np.zeros(2)
    s = u[0]
    V = u[1]
    N_ = s+V
    dudt[0] = (2*Ps - 1) * ms * s * (1 - (N_)/k) - delta_s*s
    dudt[1] = 2*(1 - Ps) * ms * s * (1 - (N_)/k) + S_prolif(N_,mv,k,delta_v,n)*V*(1 - (N_)/k) - S_death(N_,mv,k,delta_v,n)*V
    return dudt

@jit
def gsc_model_reduced_dVdt(s,V,n,Ps,k,ms,mv,delta_s,delta_v):
    
    N_ = s+V
    dVdt = 2*(1 - Ps) * ms * s * (1 - (N_)/k) + S_prolif(N_,mv,k,delta_v,n)*V*(1 - (N_)/k) - S_death(N_,mv,k,delta_v,n)*V
    return dVdt


@jit
def radiation(u,alpha,beta,eta,mu,d):
    
    gamma = np.exp(-eta*(alpha*d + beta*d*d))
    u[0] =  u[0]*gamma
    gamma = np.exp(-(alpha*d + beta*d*d))
    u[1:-1] =  u[1:-1]*gamma
    gamma = np.exp(-mu*(alpha*d + beta*d*d))
    u[-1] =  u[-1]*gamma
    
    return u

@jit
def resection(u,resect_fraction):

    #For now assume that resection is equal among all cell compartments
    u[0] =  u[0]*(1 - resect_fraction)
    u[1:-1] =  u[1:-1]*(1 - resect_fraction)
    u[-1] =  u[-1]*(1 - resect_fraction)
    
    return u

@jit
def detection_death(threshold, N, m, lam ):
    
    #prob = N**m / (threshold**m + N**m)
    
    prob = lam / (1 + np.exp(-m*(N - threshold)))
    
    return prob

# according to this paper rho is linearly related to alpha by factor of 0.005
# 10.1088/0031-9155/55/12/001
@jit
def calc_alpha_from_rho(rhos,alpha_rho_scale=0.005):
    x = rhos * alpha_rho_scale
    return x

# assume that alpha/beta = 10 ratio is fixed
@jit
def calc_beta(alpha, ratio =10):
    beta = alpha / ratio
    return beta

@jit
def simulate_model(t_final,dt,u0,psi,Ps_max,Ps_min,n,k,ms,mv,delta_s,delta_v,delta_m,delta_b,C,u_s,n_RT_repeat,n_RT_cycles,rad_on,alpha,beta,eta,mu,t_RT_wait,t_RT_interval,detect_threshold,detection_sensitivity,death_sensitivity,lam,resection_to_RT_delay,BMP4_on,death_threshold,resect_on,resect_fraction,rng_seed,m_init) : 
    np.random.seed(rng_seed)
    t = np.arange(0, t_final+dt/2, dt) # arange(a,b,c) uses the open interval [a,b) with steps c
    nt = len(t)
    dt_out = 1

    u = np.zeros((nt,n+1))
    VS = np.zeros(nt)
    N = np.zeros(nt)
    m = np.zeros(nt)
    B = np.zeros(nt)

    t_rad = -25
    rad_counter = 0
    detect_size = 0
    detect_t = 0
    t2 = 0 # doubling time, approximated by time from 0.1*k to 0.2*k
    tp1 = 0 # time to cross 0.1*k
    tp2 = 0 # time to cross 0.2*k

    # define IC
    u[0,:] = u0
    VS[0] = np.sum(u[0,1:n+1]) # sum the 1th to nth entries
    N[0] = u[0,0] + VS[0]

    i = 0
    while death_threshold > 0 and i<nt:
            
            Ps = Ps_min + (Ps_max - Ps_min)*(1 / (1 + psi*B[i]))
                
            u[i+1,:] = u[i,:] + dt * gsc_model_dudt(u[i,:],n,Ps,k,ms,mv,delta_s,delta_v)

            
            # update ASMC mesenchymal cells delivering BMP
            m[i+1] = m[i] + dt * (- delta_m * m[i])
            B[i+1] = B[i] + dt * (C * m[i] - u_s*B[i]*u[i,0] - delta_b*B[i])

            # apply radiation
            if abs(t[i]-t_rad) < dt/2 and rad_counter < n_RT_repeat*n_RT_cycles and rad_on == 1 :
                u[i+1,:] = radiation(u[i+1,:],alpha,beta,eta,mu,d=2)
                rad_counter = rad_counter + 1
                if rad_counter % n_RT_repeat == 0 :
                    t_rad = t_rad + t_RT_wait # wait for next cycle
                else:
                    t_rad = t_rad + t_RT_interval
                    
            # test for tumor detection
            # generate a random uniform number between [0,1]
            rand = np.random.random_sample()
            # as the density of N increases the probability of detection and resection increases to 1
            if (rand < detection_death(detect_threshold, np.sum(u[i+1,:]),detection_sensitivity,lam) *dt and detect_threshold >0): 
                detect_threshold = -1
                detect_size = np.sum(u[i,:])
                detect_t = t[i]
                detect_i = i
                i_rad = detect_i + int(resection_to_RT_delay/dt)
                if resect_on ==1:
                    u[i+1,:] = resection(u[i+1,:],resect_fraction)
                t_rad = t[i+1] + resection_to_RT_delay
                # apply BMP4 at time of resection
                if  BMP4_on ==1 :
                    m[i+1] = m_init
        
            VS[i+1] = np.sum(u[i+1,1:n+1])
            N[i+1] = u[i+1,0] + VS[i+1]
        
            # estimate doubling time
            if (N[i]<0.1*k and N[i+1]>=0.1*k):
                ip1 = i+1
                tp1 = t[i+1]
            if (N[i]<0.2*k and N[i+1]>=0.2*k):
                ip2 = i+1
                tp2 = t[i+1]

            # test for death        
            # generate a random uniform number between [0,1]
            rand = np.random.random_sample()
            # as the density of N increases the probability of death increases to 1
            if rand < detection_death(death_threshold, N[i+1],death_sensitivity, lam) * dt:
                death_threshold = -1
        
            i = i + 1

    # cut all them off at final index
    it_store = np.unique(np.append(np.arange(0,i,int(np.floor(dt_out/dt))),[i-1,detect_i,detect_i+1,i_rad,ip1,ip2])) # indices for daily output and special times
    u = u[it_store,:]
    N = N[it_store]
    VS = VS[it_store]
    t = t[it_store]
    m = m[it_store]
    B = B[it_store]

    t2 = tp2-tp1

    return u,N,VS,t,m,B,detect_size,detect_t,tp1,tp2


@jit
def simulate_model_shifted(t_final,dt,u0,psi,Ps_max,Ps_min,n,k,ms,mv,delta_s,delta_v,delta_m,delta_b,C,u_s,n_RT_repeat,n_RT_cycles,rad_on,alpha,beta,eta,mu,t_RT_wait,t_RT_interval,detect_threshold,detection_sensitivity,death_sensitivity,lam,resection_to_RT_delay,BMP4_on,death_threshold,resect_on,resect_fraction,rng_seed,m_init) : 
    np.random.seed(rng_seed)
    t = np.arange(0, t_final+dt/2, dt) # arange(a,b,c) uses the open interval [a,b) with steps c
    nt = len(t)
    dt_out = 1

    u = np.zeros((nt,n+1))
    VS = np.zeros(nt)
    N = np.zeros(nt)
    m = np.zeros(nt)
    B = np.zeros(nt)

    RT_started = 0
    stop = 0
    t_rad = -25
    rad_counter = 0
    detect_size = 0
    detect_t = 0
    t2 = 0 # doubling time, approximated by time from 0.1*k to 0.2*k
    tp1 = 0 # time to cross 0.1*k
    tp2 = 0 # time to cross 0.2*k

    # define IC
    u[0,:] = u0
    VS[0] = np.sum(u[0,1:n+1]) # sum the 1th to nth entries
    N[0] = u[0,0] + VS[0]

    i = 0
    while death_threshold > 0 and i<nt:
            
            Ps = Ps_min + (Ps_max - Ps_min)*(1 / (1 + psi*B[i]))
                
            u[i+1,:] = u[i,:] + dt * gsc_model_dudt(u[i,:],n,Ps,k,ms,mv,delta_s,delta_v)

            
            # update ASMC mesenchymal cells delivering BMP
            m[i+1] = m[i] + dt * (- delta_m * m[i])
            B[i+1] = B[i] + dt * (C * m[i] - u_s*B[i]*u[i,0] - delta_b*B[i])
            if  BMP4_on ==1 and RT_started==1 and stop ==0 :
                stop =1
                m[i+1] = m_init

            # apply radiation
            if abs(t[i]-t_rad) < dt/2 and rad_counter < n_RT_repeat*n_RT_cycles and rad_on == 1 :
                u[i+1,:] = radiation(u[i+1,:],alpha,beta,eta,mu,d=2)
                rad_counter = rad_counter + 1
                RT_started = 1
                if rad_counter % n_RT_repeat == 0 :
                    t_rad = t_rad + t_RT_wait # wait for next cycle
                else:
                    t_rad = t_rad + t_RT_interval
                    
            # test for tumor detection
            # generate a random uniform number between [0,1]
            rand = np.random.random_sample()
            # as the density of N increases the probability of detection and resection increases to 1
            if (rand < detection_death(detect_threshold, np.sum(u[i+1,:]),detection_sensitivity,lam) *dt and detect_threshold >0): 
                detect_threshold = -1
                detect_size = np.sum(u[i,:])
                detect_t = t[i]
                detect_i = i
                i_rad = detect_i + int(resection_to_RT_delay/dt)
                if resect_on ==1:
                    u[i+1,:] = resection(u[i+1,:],resect_fraction)
                t_rad = t[i+1] + resection_to_RT_delay

        
            VS[i+1] = np.sum(u[i+1,1:n+1])
            N[i+1] = u[i+1,0] + VS[i+1]
        
            # estimate doubling time
            if (N[i]<0.1*k and N[i+1]>=0.1*k):
                ip1 = i+1
                tp1 = t[i+1]
            if (N[i]<0.2*k and N[i+1]>=0.2*k):
                ip2 = i+1
                tp2 = t[i+1]

            # test for death        
            # generate a random uniform number between [0,1]
            rand = np.random.random_sample()
            # as the density of N increases the probability of death increases to 1
            if rand < detection_death(death_threshold, N[i+1],death_sensitivity, lam) * dt:
                death_threshold = -1
        
            i = i + 1

    # cut all them off at final index
    it_store = np.unique(np.append(np.arange(0,i,int(np.floor(dt_out/dt))),[i-1,detect_i,detect_i+1,i_rad,ip1,ip2])) # indices for daily output and special times
    u = u[it_store,:]
    N = N[it_store]
    VS = VS[it_store]
    t = t[it_store]
    m = m[it_store]
    B = B[it_store]

    t2 = tp2-tp1

    return u,N,VS,t,m,B,detect_size,detect_t,tp1,tp2


@jit
def simulate_model_continuos(t_final,dt,u0,psi,Ps_max,Ps_min,n,k,ms,mv,delta_s,delta_v,delta_m,delta_b,C,u_s,n_RT_repeat,n_RT_cycles,rad_on,alpha,beta,eta,mu,t_RT_wait,t_RT_interval,detect_threshold,detection_sensitivity,death_sensitivity,lam,resection_to_RT_delay,BMP4_on,death_threshold,resect_on,resect_fraction,rng_seed,m_fixed) : 
    np.random.seed(rng_seed)
    t = np.arange(0, t_final+dt/2, dt) # arange(a,b,c) uses the open interval [a,b) with steps c
    nt = len(t)
    dt_out = 1

    u = np.zeros((nt,n+1))
    VS = np.zeros(nt)
    N = np.zeros(nt)
    m = np.zeros(nt)
    B = np.zeros(nt)

    t_rad = -25
    rad_counter = 0
    detect_size = 0
    detect_t = 0
    tp1 = 0 # time to cross 0.1*k
    tp2 = 0 # time to cross 0.2*k

    # define IC
    u[0,:] = u0
    VS[0] = np.sum(u[0,1:n+1]) # sum the 1th to nth entries
    N[0] = u[0,0] + VS[0]

    i = 0
    while death_threshold > 0 and i<nt:
            

            Ps = Ps_min + (Ps_max - Ps_min)*(1 / (1 + psi*B[i]))
                
            u[i+1,:] = u[i,:] + dt * gsc_model_dudt(u[i,:],n,Ps,k,ms,mv,delta_s,delta_v)

            # After detection and resection, MSCs are fixed 
            if detect_threshold ==-1 and rad_counter < n_RT_repeat*n_RT_cycles:
                m[i+1] = m_fixed
            else: # once RT has ended then constant delivery is stopped and MSCs start decaying
                m[i+1] = m[i] + dt * (- delta_m * m[i])
            
            B[i+1] = B[i] + dt * (C * m[i] - u_s*B[i]*u[i,0] - delta_b*B[i])

            # apply radiation
            if abs(t[i]-t_rad) < dt/2 and rad_counter < n_RT_repeat*n_RT_cycles and rad_on == 1 :
                u[i+1,:] = radiation(u[i+1,:],alpha,beta,eta,mu,d=2)
                rad_counter = rad_counter + 1
                if rad_counter % n_RT_repeat == 0 :
                    t_rad = t_rad + t_RT_wait # wait for next cycle
                else:
                    t_rad = t_rad + t_RT_interval
                    
            # test for tumor detection
            # generate a random uniform number between [0,1]
            rand = np.random.random_sample()
            # as the density of N increases the probability of detection and resection increases to 1
            if (rand < detection_death(detect_threshold, np.sum(u[i+1,:]),detection_sensitivity,lam) *dt and detect_threshold >0): 
                detect_threshold = -1
                detect_size = np.sum(u[i,:])
                detect_t = t[i]
                detect_i = i
                i_rad = detect_i + int(resection_to_RT_delay/dt)
                if resect_on ==1:
                    u[i+1,:] = resection(u[i+1,:],resect_fraction)
                t_rad = t[i+1] + resection_to_RT_delay

        
            VS[i+1] = np.sum(u[i+1,1:n+1])
            N[i+1] = u[i+1,0] + VS[i+1]
        
            # estimate doubling time
            if (N[i]<0.1*k and N[i+1]>=0.1*k):
                ip1 = i+1
                tp1 = t[i+1]
            if (N[i]<0.2*k and N[i+1]>=0.2*k):
                ip2 = i+1
                tp2 = t[i+1]

            # test for death        
            # generate a random uniform number between [0,1]
            rand = np.random.random_sample()
            # as the density of N increases the probability of death increases to 1
            if rand < detection_death(death_threshold, N[i+1],death_sensitivity, lam) * dt:
                death_threshold = -1
        
            i = i + 1

    # cut all them off at final index
    it_store = np.unique(np.append(np.arange(0,i,int(np.floor(dt_out/dt))),[i-1,detect_i,detect_i+1,i_rad,ip1,ip2])) # indices for daily output and special times
    u = u[it_store,:]
    N = N[it_store]
    VS = VS[it_store]
    t = t[it_store]
    m = m[it_store]
    B = B[it_store]

    return u,N,VS,t,m,B,detect_size,detect_t,tp1,tp2


@jit
def simulate_model_pulsatile(t_final,dt,u0,psi,Ps_max,Ps_min,n,k,ms,mv,delta_s,delta_v,delta_m,delta_b,C,u_s,n_RT_repeat,n_RT_cycles,rad_on,alpha,beta,eta,mu,t_RT_wait,t_RT_interval,detect_threshold,detection_sensitivity,death_sensitivity,lam,resection_to_RT_delay,BMP4_on,death_threshold,resect_on,resect_fraction,rng_seed,m_pulse_init) : 
    np.random.seed(rng_seed)
    t = np.arange(0, t_final+dt/2, dt) # arange(a,b,c) uses the open interval [a,b) with steps c
    nt = len(t)
    dt_out = 1

    u = np.zeros((nt,n+1))
    VS = np.zeros(nt)
    N = np.zeros(nt)
    m = np.zeros(nt)
    B = np.zeros(nt)

    t_rad = -25
    rad_counter = 0
    detect_size = 0
    detect_t = 0
    tp1 = 0 # time to cross 0.1*k
    tp2 = 0 # time to cross 0.2*k

    # define IC
    u[0,:] = u0
    VS[0] = np.sum(u[0,1:n+1]) # sum the 1th to nth entries
    N[0] = u[0,0] + VS[0]


    i = 0
    while death_threshold > 0 and i<nt:
            
            Ps = Ps_min + (Ps_max - Ps_min)*(1 / (1 + psi*B[i]))
                
            u[i+1,:] = u[i,:] + dt * gsc_model_dudt(u[i,:],n,Ps,k,ms,mv,delta_s,delta_v)


            m[i+1] = m[i] + dt * (- delta_m * m[i])
            B[i+1] = B[i] + dt * (C * m[i] - u_s*B[i]*u[i,0] - delta_b*B[i])

            # BMP4 is Pulsatile in combination with RT
            if abs(t[i]-t_rad) < dt/2 and rad_counter % n_RT_repeat == 0 and rad_counter < n_RT_repeat*n_RT_cycles:
                m[i+1] = m[i] + m_pulse_init



            # apply radiation
            if abs(t[i]-t_rad) < dt/2 and rad_counter < n_RT_repeat*n_RT_cycles and rad_on == 1 :
                u[i+1,:] = radiation(u[i+1,:],alpha,beta,eta,mu,d=2)
                rad_counter = rad_counter + 1
                if rad_counter % n_RT_repeat == 0 :
                    t_rad = t_rad + t_RT_wait # wait for next cycle
                else:
                    t_rad = t_rad + t_RT_interval
                    
            # test for tumor detection
            # generate a random uniform number between [0,1]
            rand = np.random.random_sample()
            # as the density of N increases the probability of detection and resection increases to 1
            if (rand < detection_death(detect_threshold, np.sum(u[i+1,:]),detection_sensitivity,lam) *dt and detect_threshold >0): 
                detect_threshold = -1
                detect_size = np.sum(u[i,:])
                detect_t = t[i]
                detect_i = i
                i_rad = detect_i + int(resection_to_RT_delay/dt)
                if resect_on ==1:
                    u[i+1,:] = resection(u[i+1,:],resect_fraction)
                t_rad = t[i+1] + resection_to_RT_delay

        
            VS[i+1] = np.sum(u[i+1,1:n+1])
            N[i+1] = u[i+1,0] + VS[i+1]
        
            # estimate doubling time
            if (N[i]<0.1*k and N[i+1]>=0.1*k):
                ip1 = i+1
                tp1 = t[i+1]
            if (N[i]<0.2*k and N[i+1]>=0.2*k):
                ip2 = i+1
                tp2 = t[i+1]

            # test for death        
            # generate a random uniform number between [0,1]
            rand = np.random.random_sample()
            # as the density of N increases the probability of death increases to 1
            if rand < detection_death(death_threshold, N[i+1],death_sensitivity, lam) * dt:
                death_threshold = -1
        
            i = i + 1

    # cut all them off at final index
    it_store = np.unique(np.append(np.arange(0,i,int(np.floor(dt_out/dt))),[i-1,detect_i,detect_i+1,i_rad,ip1,ip2])) # indices for daily output and special times
    u = u[it_store,:]
    N = N[it_store]
    VS = VS[it_store]
    t = t[it_store]
    m = m[it_store]
    B = B[it_store]

    return u,N,VS,t,m,B,detect_size,detect_t,tp1,tp2

def phase2_trial_fun(n_trials,n_patients,distinct_arms,rho_case,shape,loc,scale): 
    # n_trials (int), number of virtual clinical trials to calculate average from.
    # n_patients (int), number of patients per phase 2 virtual trial
    # distinct_arms (bool), whether the BMP4 and noBMP4 arms should be distinct sub-populations
    # rho_case (int), how to select patients from the rho distribution

    from gsc_model_params import mu,eta,n,k,delta_s,delta_v,delta_m,delta_b,u_s,C,lam,Ps_max,Ps_min,psi,mv_rho_scale,ms_mv_scale,mv_rho_scale,ms_mv_scale,detect_threshold,death_threshold,detection_sensitivity,death_sensitivity,alpha_rho_scale,resection_to_RT_delay,t_RT_interval,t_RT_cycle,n_RT_repeat,n_RT_cycles,t_RT_wait,resect_fraction

    s0 = 0.001 # Initial GSCs

    u0 = np.zeros(n+1)
    u0[0] = s0

    n0 = 0.001
    s0 = 0.01*n0 # fraction of initial tumour
    v_ratio = 1.95 # ratio between successive compartments
    v0 = ((n0-s0)*(v_ratio-1)/(v_ratio**n-1))*(v_ratio**np.arange(n))
    u0 = np.zeros(n+1)
    u0[0] = s0
    u0[1:] = v0

    psi_mean_values = [0, 0.5, 5, 50] # mean psi value to generate samples from turn normal
    frac_succ = np.zeros(len(psi_mean_values)) # save the number of trials that are successful

    # set up time grid
    t_final = 8000
    dt = 0.01
    t = np.arange(0, t_final+dt/2, dt)

    save_data = np.zeros([n_trials*n_patients, 5]) # store all the data we need
    # for each trial find out p value
    p_values = np.zeros([len(psi_mean_values),n_trials])
    BMP4_mean_survival = np.zeros([len(psi_mean_values),n_trials])
    noBMP4_mean_survival = np.zeros([len(psi_mean_values),n_trials])
    mean_psi = np.zeros([len(psi_mean_values),n_trials])
    mean_rho_BMP4 = np.zeros([len(psi_mean_values),n_trials])
    mean_rho_noBMP4 = np.zeros([len(psi_mean_values),n_trials])
    all_rhos = np.zeros([len(psi_mean_values),n_trials*n_patients])

    # we want each patient to have a unique random seed so that across all simulations they get the same series of random numbers
    random_seeds = np.arange(0,n_patients,1)

    w = 0
    figs = []

    for psi_mean in psi_mean_values:

        for trial in range(n_trials):

            # for each trial generate a unique set of heterogenous patients
            # set up distinct sets for BMP and noBMP first, then overwrite the latter with the former if distinct sets are not required 
            np.random.seed(trial) 
            if psi_mean==0 : 
                psi_samples_BMP4 = np.zeros(n_patients)
                psi_samples_noBMP4 = np.zeros(n_patients)
            else : 
                psi_samples_BMP4 = trunc_norm(psi_mean,1,n_patients)
                psi_samples_noBMP4 = trunc_norm(psi_mean,1,n_patients)
            np.random.seed(trial) 
            pro_rates_sampled_BMP4 = np.sort(lognorm.rvs(shape, loc, scale=scale, size=n_patients))
            pro_rates_sampled_noBMP4 = np.sort(lognorm.rvs(shape, loc, scale=scale, size=n_patients))
            if rho_case==0 : 
                ### consider four cases here beyond the base case:
                ### 0) sample required n_patients
                pro_rates_sampled_BMP4 = np.sort(lognorm.rvs(shape, loc, scale=scale, size=n_patients))
                pro_rates_sampled_noBMP4 = np.sort(lognorm.rvs(shape, loc, scale=scale, size=n_patients))
            elif rho_case==1 :
                ### 1) sample 5x required, take the top 20% (n_patients)
                pro_rates_sampled_BMP4 = np.sort(lognorm.rvs(shape, loc, scale=scale, size=5*n_patients))
                pro_rates_sampled_BMP4 = pro_rates_sampled_BMP4[-n_patients:]
                pro_rates_sampled_noBMP4 = np.sort(lognorm.rvs(shape, loc, scale=scale, size=5*n_patients))
                pro_rates_sampled_noBMP4 = pro_rates_sampled_noBMP4[-n_patients:]
            elif rho_case==2 :
                ### 2) sample 2x required, take the top 50% (n_patients)
                pro_rates_sampled_BMP4 = np.sort(lognorm.rvs(shape, loc, scale=scale, size=2*n_patients))
                pro_rates_sampled_BMP4 = pro_rates_sampled_BMP4[-n_patients:]
                pro_rates_sampled_noBMP4 = np.sort(lognorm.rvs(shape, loc, scale=scale, size=2*n_patients))
                pro_rates_sampled_noBMP4 = pro_rates_sampled_noBMP4[-n_patients:]
            elif rho_case==3 :
                ### 3) sample a distribution with 2x scale parameter (not enough to get much response)
                pro_rates_sampled_BMP4 = np.sort(lognorm.rvs(shape, loc, scale=2*scale, size=n_patients))
                pro_rates_sampled_noBMP4 = np.sort(lognorm.rvs(shape, loc, scale=2*scale, size=n_patients))
            elif rho_case==4 :
                ### 4) sample a distribution with 3x scale parameter (this is  enough to get a significant response)
                pro_rates_sampled_BMP4 = np.sort(lognorm.rvs(shape, loc, scale=3*scale, size=n_patients))
                pro_rates_sampled_noBMP4 = np.sort(lognorm.rvs(shape, loc, scale=3*scale, size=n_patients))
            elif rho_case==5 :
                ### 5) sample 4x required, take the top 50% (2*n_patients) and split in two between BMP4 and noBMP4
                pro_rates_sampled = np.sort(lognorm.rvs(shape, loc, scale=scale, size=4*n_patients))
                pro_rates_sampled_BMP4 = pro_rates_sampled[-2*n_patients::2]
                pro_rates_sampled_noBMP4 = pro_rates_sampled[-(2*n_patients-1)::2]

            if not(distinct_arms) :
                pro_rates_sampled_noBMP4 = pro_rates_sampled_BMP4
                psi_samples_noBMP4 = psi_samples_BMP4

            all_rhos[w,trial*n_patients:(trial*n_patients+n_patients)] = pro_rates_sampled_BMP4

            rad_on = 1        
            BMP4_on = 1
            resect_on = 1

            q = 0 # loop variable

            BMP4_survival = np.zeros(n_patients)
            noBMP4_survival = np.zeros(n_patients)

            for psi, pro_r in zip(psi_samples_BMP4,pro_rates_sampled_BMP4):
                
                    mv = mv_rho_scale*pro_r*np.ones(n)
                    ms = ms_mv_scale*mv[0]
                    mv[n-1] = 0 
            
                    # calc alpha as proportional to rho
                    alpha = calc_alpha_from_rho(pro_r)
                    beta = calc_beta(alpha)
            
                    # simulate the model
                    u,N,VS,t,m,B,detect_size,detect_t,_,_ = simulate_model(t_final,dt,u0,psi,Ps_max,Ps_min,n,k,ms,mv,delta_s,delta_v,delta_m,delta_b,C,u_s,n_RT_repeat,n_RT_cycles,rad_on,alpha,beta,eta,mu,t_RT_wait,t_RT_interval,detect_threshold,detection_sensitivity,death_sensitivity,lam,resection_to_RT_delay,BMP4_on,death_threshold,resect_on,resect_fraction,random_seeds[q])

                
                    # save the survival time of the BMP4 arm
                    save_data[(trial*n_patients)+q,2] = t[-1]-detect_t
                    # svae the trial number
                    save_data[(trial*n_patients)+q,0] = trial
                    # save the psi value
                    save_data[(trial*n_patients)+q,1] = psi
                    BMP4_survival[q] = t[-1]-detect_t
                    
                    q = q+1

            # for each of the patients run the same thing again but with no BMP4 to act as a virtual control
            rad_on = 1
            BMP4_on = 0
            resect_on = 1

            q = 0 # loop variable

            for psi, pro_r in zip(psi_samples_noBMP4, pro_rates_sampled_noBMP4):
                
                    mv = mv_rho_scale*pro_r*np.ones(n)
                    ms = ms_mv_scale*mv[0]
                    mv[n-1] = 0 
            
                    # calc alpha as proportional to rho 
                    alpha = calc_alpha_from_rho(pro_r)
                    beta = calc_beta(alpha)

                    u,N,VS,t,m,B,detect_size,detect_t,_,_ = simulate_model(t_final,dt,u0,psi,Ps_max,Ps_min,n,k,ms,mv,delta_s,delta_v,delta_m,delta_b,C,u_s,n_RT_repeat,n_RT_cycles,rad_on,alpha,beta,eta,mu,t_RT_wait,t_RT_interval,detect_threshold,detection_sensitivity,death_sensitivity,lam,resection_to_RT_delay,BMP4_on,death_threshold,resect_on,resect_fraction,random_seeds[q])
                
                    # save the survival time of the virtual control arm
                    save_data[(trial*n_patients)+q,3] = t[-1]-detect_t
                    # svae the trial number
                    save_data[(trial*n_patients)+q,0] = trial
                    noBMP4_survival[q] = t[-1]-detect_t
                    q = q+1
            BMP4_mean_survival[w,trial] = np.mean(BMP4_survival)
            noBMP4_mean_survival[w,trial] = np.mean(noBMP4_survival)
            mean_psi[w,trial] = np.mean(psi_samples_BMP4)
            mean_rho_BMP4[w,trial] = np.mean(pro_rates_sampled_BMP4)
            mean_rho_noBMP4[w,trial] = np.mean(pro_rates_sampled_noBMP4)


        survival_BMP4_df = pd.DataFrame(save_data, columns=['trial','psi','Survival_time_BMP4','Virtual_Control', 'Status'])

        # has to be set to boolean not just integer
        survival_BMP4_df['Status'] = True
        
        nrows = int(np.sqrt(n_trials))
        ncols = int(n_trials/nrows)

        fig = plt.figure(figsize=(15, 15))
        gs = fig.add_gridspec(nrows, ncols, hspace=0, wspace=0)
        axs = gs.subplots(sharex=True, sharey=True)

        for i in range(n_trials):

            df = survival_BMP4_df[survival_BMP4_df['trial']==i]
            # need to create a structed array:
            dtype = [('event_indicator', bool), ('time', float)]
            time = np.zeros([n_patients*2])
            time[0:n_patients,] = np.array(df["Virtual_Control"])
            time[n_patients:,] = np.array(df["Survival_time_BMP4"])
            event_indicators = np.ones([n_patients*2])
            structured_array = np.array( list(zip(event_indicators, time)) , dtype=dtype)

            group = np.zeros(n_patients*2)
            group[n_patients:] = 1
            chisq, p_value, stats1, covariance = compare_survival(y = structured_array, group_indicator= group, return_stats=True)
            p_values[w,i] = p_value

            time, survival_prob, conf_int = kaplan_meier_estimator(df["Status"], df["Virtual_Control"], conf_type="log-log")
            fig.axes[i].step(time, survival_prob, where="post")
            fig.axes[i].fill_between(time, conf_int[0], conf_int[1], alpha=0.25, step="post",  label='_nolegend_')
            time, survival_prob, conf_int = kaplan_meier_estimator(df["Status"], df["Survival_time_BMP4"], conf_type="log-log")
            fig.axes[i].step(time, survival_prob, where="post")
            fig.axes[i].fill_between(time, conf_int[0], conf_int[1], alpha=0.25, step="post",  label='_nolegend_')
            # fig.axes[i].ylim(0, 1)
            fig.axes[i].text(.9,.8,'p=' + str('%.*g' % (3, p_value)),horizontalalignment='right',transform=fig.axes[i].transAxes)
            if p_value < 0.05 :
                fig.axes[i].set_facecolor('0.9')

        fig.supylabel("Survival")
        fig.supxlabel("Time (day)")
        fig.suptitle(r"Simulated survival BMP4 + RT, " + str(n_trials) + "trials")
        # fig.subplots_adjust(bottom=0.2) 
        fig.legend([r"Virtual control", r"Simulated BMP4 $\bar\psi = $" + str(psi_mean)],loc="lower left", ncol=1)

        # fraction of successful trials
        frac_succ[w] = np.sum(p_values[w,:] < 0.05)/n_trials
        figs.append(fig)
        w = w+1

    return psi_mean_values,frac_succ,BMP4_mean_survival,noBMP4_mean_survival,mean_rho_BMP4,mean_rho_noBMP4,p_values,figs

# Jit doesn't work with scipystats so don't add it here
def trunc_norm(mu,sigma,n_samples):

    # Define the bounds for the truncated distribution
    amin, amax = 0, np.inf
    amin, amax = (amin - mu) / sigma, (amax - mu) / sigma

    # Generate samples
    samples = stats.truncnorm.rvs(amin, amax, loc=mu, scale=sigma, size=n_samples)

    
    return samples


@jit
def basic_growth_simulation(t_final,dt,u0,s0,Ps_max,Ps_min,n,k,ms,mv,delta_s,delta_v,max_size): 
    t = np.arange(0, t_final+dt/2, dt) # arange(a,b,c) uses the open interval [a,b) with steps c
    nt = len(t)
    u = np.zeros((nt,n+1))
    VS = np.zeros(nt)
    N = np.zeros(nt)

    tp1 = 0 # time to cross 0.1*k
    tp2 = 0 # time to cross 0.2*k

    # define IC
    u[0,:] = u0
    u[0,0] = s0
    VS[0] = np.sum(u[0,1:n+1]) # sum the 1th to nth entries
    N[0] = u[0,0] + VS[0]

    i = 0
    while N[i] < max_size or i<nt:
            
            Ps = Ps_min + (Ps_max - Ps_min)
                
            u[i+1,:] = u[i,:] + dt * gsc_model_dudt(u[i,:],n,Ps,k,ms,mv,delta_s,delta_v)


            VS[i+1] = np.sum(u[i+1,1:n+1])
            N[i+1] = u[i+1,0] + VS[i+1]
        
            # estimate doubling time
            if (N[i]<0.1*k and N[i+1]>=0.1*k):
                tp1 = t[i+1]
            if (N[i]<0.2*k and N[i+1]>=0.2*k):
                tp2 = t[i+1]

            if N[i] > max_size*k:
                break

            i = i + 1

    # cut all them off at final index
    u = u[0:i,:]
    N = N[0:i]
    VS = VS[0:i]
    t = t[0:i]


    return u,N,VS,t,tp1,tp2