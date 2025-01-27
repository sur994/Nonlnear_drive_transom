#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 12:06:00 2024

@author: ssengupta
"""

import numpy as np;
import matplotlib.pyplot as plt;
#from scipy.integrate import solve_ivp;
from odeintw import odeintw;
#import os;
#import pandas as pd;
import math;
from numpy.linalg import matrix_power;
from numpy import linalg as la
from time import perf_counter;
import qutip as qt



def full_Hamil(elem,Eq,Ej,phi_j,gamma,f,tprime,Nsteps,p,save,level,loop):
    #elem: no of states; Ej_tilde= E_j/gamma; phi_0: root(alpha) from the paper; delta: detuning; 
    #p: resonances
    

    t1_start = perf_counter()
    
    t_ini=0;
    t_fin=1000
    
    dt=(t_fin-t_ini)/Nsteps;
    Nsteps1=int(tprime/dt);
        
    #calculate a, a^+ ,n
    adagger=np.zeros((elem,elem),dtype=complex)         
    a0=np.zeros((elem,elem),dtype=complex);
    a1=np.zeros((elem,elem),dtype=complex);                                
    nph=0;
    for nrows in range(elem):
        for ncols in range(elem):
            if ncols==nrows-1:
                adagger[nrows][ncols]=np.sqrt(nph);
            else:
                adagger[nrows][ncols]=0;
            
        
        nph=nph+1;
    a=np.transpose(adagger);
    n=np.matmul(adagger,a);

    a0[0][1]=np.sqrt(1);
    adagger0=np.transpose(a0);
    
    a1[1][2]=np.sqrt(2);
    adagger1=np.transpose(a1);
    
    n_01=adagger0 @ a0;
    n_12=adagger1 @ a1;


    
    
    #calculate the Laguerre polynomials using the recursive formula
    L_n=np.zeros(elem,dtype=complex);
    B_p=np.zeros((elem,elem),dtype=complex);#B_k operator as defined in the paper
    L_n0_vec=np.zeros(elem,dtype=complex);
    L_n0=np.zeros((elem,elem),dtype=complex);
       
    L_n[0]=1;
    L_n[1]=1+p-phi_j**2;
    B_p[0][0]=(1/(math.factorial(p)))*L_n[0]; #B_k for n=0
    B_p[1][1]=(1/(math.factorial(1+p)))*L_n[1]; #B_k for n=1
    L_n0_vec[0]=1;
    L_n0_vec[1]=1-phi_j**2
    L_n0[0][0]=1
    L_n0[1][1]=1-phi_j**2
    
    
    i=1;
    for x in range(elem-2): 
        L_n[i+1]=((2*i+1+p-phi_j**2)*L_n[i]-((i+p)*L_n[i-1]))/(i+1);
        L_n0_vec[i+1]=((2*i+1-phi_j**2)*L_n0_vec[i]-((i)*L_n0_vec[i-1]))/(i+1)
        for n_cols in range(elem-2):
            if n_cols==x:
                B_p[i+1][i+1]=(math.factorial(i+1)/math.factorial(i+1+p))*L_n[i+1];#matrix for B_k
                L_n0[i+1][i+1]= L_n0_vec[i+1]
                
        i=i+1;  
        
     
    B_pdag=np.transpose(B_p);  # B_kdagger
    a_op_power=((-1j)**p)*matrix_power(a,p);  #a^k
    a_dag_op_power=((1j)**p)*matrix_power(adagger,p); #adagger^k
     
    B_p_adag=np.matmul(a_dag_op_power,B_p); #adagger^k*B_k
     
    B_pdag_a=np.matmul(B_pdag,a_op_power);
    #B_kdagger*a^k
    
 
    
    #print(L_n0)
    
   
    
    for counter in range(loop):
    
        Ej_star=(Ej/2)*np.exp(-(phi_j**2)/2)*(phi_j**p);#prefactor of the driving term
        Eq_star=Eq*np.exp(-(phi_j**2)/2);
        T_n=np.zeros(elem,dtype=complex)                                                       #with renormalised E_j
        if p==1:
            for st in range(elem):
                T_n[st]=(1j*Ej_star*L_n[st])/np.sqrt(st+1)
                
                
        #delta=0.5*Eq_star*(L_n0_vec[2]-1)
        delta=0
        print(delta)
        
      
   
         
        def drho(rho_i,t):
                
             comm_n_rho=-1j*(np.matmul(n,rho_i)-np.matmul(rho_i,n)); #commutator of rho, n
             comm_anh_rho=-1j*(np.matmul(L_n0,rho_i)-np.matmul(rho_i,L_n0));
           #  comm_adag_rho=-1j*(np.matmul(adagger,rho_i)-np.matmul(rho_i,adagger))
            # comm_a_rho=-1j*(np.matmul(a,rho_i)-np.matmul(rho_i,a))
           #  comm_ext_drive_rho=np.exp(1j*(-Omega)*t)*comm_adag_rho+np.exp(-1j*(-Omega)*t)*comm_a_rho
            
                                
                                
              
               #np.exp(-1j*(delta-Omega)*t)*
                 #commutator of rho, #adagger^k*B_k and c.c.  
             com_op_rho=-1j*((np.matmul(B_p_adag,rho_i)-np.matmul(rho_i,B_p_adag))+\
                                   (np.matmul(B_pdag_a,rho_i)-np.matmul(rho_i, B_pdag_a)));
                     
             comm=(-delta)*comm_n_rho+Eq_star*comm_anh_rho+Ej_star*com_op_rho
             #+(f/2)*comm_ext_drive_rho; #total commutator of the hamiltonian with rho  +Ej*np.exp(-(phi_j**2)/2)*L_n0
                     
             A=np.matmul(a,rho_i);
                
                 #rho_dot with the lindblad term
             drho_dt=comm+(gamma/2)*((2*np.matmul(A,adagger))-np.matmul(n,rho_i)-\
                                   np.matmul(rho_i,n));
                                   
                     
             return drho_dt  
        
      
                 #input matrix for rho
        rho_input=np.zeros((elem,elem),dtype=complex);
        rho_input[0][0]=1;
        
        
        
        t_for_ss=np.linspace(t_ini,t_fin,Nsteps) #time grid for steady state
        rho_soln=odeintw(drho,rho_input,t_for_ss) #solver for steady state
        
        xvec=np.linspace(-5,5,300)
        yvec=np.linspace(-5,5,300)
        
        rho_final=rho_soln[-1,:,:]
        
        #print(len(rho_final))
        
        N = int(len(rho_final))
        rho_final = rho_final.reshape((N, N))
        
        #print(rho_final)
        
        
        
        rho_qutip = qt.Qobj(rho_final)
        
        wigner_function = qt.wigner(rho_qutip, xvec, yvec);
        
        
        wmap = plt.contourf(xvec, yvec, wigner_function, 100, cmap='magma')
        #wmap = qt.wigner_cmap(wigner_function) 
        
        
        
       # plt1 = plt.contourf(xvec, xvec,wigner_function ,100, cmap=wmap)
       # axes=plt.gca();
        plt.xlabel('x');
        plt.ylabel('p');
        #cb = plt.colorbar(plt1,ax=axes)
        plt.colorbar(wmap)

        plt.show();