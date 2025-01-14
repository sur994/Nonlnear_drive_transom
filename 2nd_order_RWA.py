#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 12:08:57 2024

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



def full_Hamil(elem,Eq,Ej,phi_j,gamma,delta,ratio,tprime,Nsteps,save,loop):
    #elem: no of states; Ej_tilde= E_j/gamma; phi_0: root(alpha) from the paper; delta: detuning; 
    #p: resonances
    

    omega_dc=ratio*(Eq)
    t1_start = perf_counter()
    
    t_ini=0;
    t_fin=1000
    
    dt=(t_fin-t_ini)/Nsteps;
    Nsteps1=int(tprime/dt);
        
    #calculate a, a^+ ,n
    adagger=np.zeros((elem,elem),dtype=complex)         
                 
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

   

    
    
    #calculate the Laguerre polynomials using the recursive formula
    L_n=np.zeros(elem,dtype=complex);
    B_p=np.zeros((elem,elem),dtype=complex);#B_k operator as defined in the paper
    L_n0_vec=np.zeros(elem,dtype=complex);
    L_n0=np.zeros((elem,elem),dtype=complex);
       
    L_n[0]=1;
    
     #B_k for n=0
    Fano=np.zeros(loop);
    Ej_pl=np.zeros(loop);
      
    
     
    for countiiii in range(loop):
    
        A_p_matrices=[];
        A_pdag_matrices=[];
        
        Comm_Ap_eq_sum = np.zeros((elem, elem))
        Comm_Ap_Ej_sum=np.zeros((elem, elem))
        Comm_Ap_Ap2_sum=np.zeros((elem, elem))
    
        p=1;
        for count in range(elem):
            
            B_p[0][0]=(1/(math.factorial(p)))*L_n[0]
            
            L_n[1]=1+p-phi_j**2;
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
            a_op_power=matrix_power(a,p);  #a^k
            a_dag_op_power=matrix_power(adagger,p); #adagger^k
             
            A_p=np.matmul(a_dag_op_power,B_p); #adagger^k*B_k
             
            A_pdag=np.matmul(B_pdag,a_op_power);
            
            A_p_matrices.append(A_p)
            A_pdag_matrices.append(A_pdag)
            
            Comm_Ap_Eq=((phi_j**p)/p)*(A_p @ A_pdag-A_pdag @A_p)*(1+(-1)**p);
           
            Comm_Ap_eq_sum=Comm_Ap_eq_sum+Comm_Ap_Eq
            
            if p>1:
                Comm_Ap_Ej=((2*p*phi_j**p)/(p**2-1))*(A_pdag @ A_p-A_p @A_pdag)
                Comm_Ap_Ej_sum=Comm_Ap_Ej_sum+Comm_Ap_Ej
            
            p=p+1;
            
    
        Comm_A0_A2=2*phi_j**2*(L_n0 @ (A_pdag_matrices[1]-A_p_matrices[1]));
        Comm_A1=(phi_j/2)*(A_pdag_matrices[0]@A_p_matrices[0]-A_p_matrices[0]@A_pdag_matrices[0])
        Comm_A1_A3=(phi_j**4/2)*(A_pdag_matrices[0]@A_p_matrices[2]-A_p_matrices[0]@A_pdag_matrices[2])
        
       
        for counter in range(elem-3):
            p=2;
            Comm_Ap_Ap2=(((-1)**p*phi_j**(2*p+2))/(1+p))*(A_p_matrices[counter+1]@A_pdag_matrices[counter+3]-A_pdag_matrices[counter+3]@A_p_matrices[counter+1]+A_p_matrices[counter+3]@A_pdag_matrices[counter+1]\
                                                          -A_pdag_matrices[counter+1]@A_p_matrices[counter+3]);
            Comm_Ap_Ap2_sum=Comm_Ap_Ap2_sum+Comm_Ap_Ap2;
            p=p+1;
       
        #B_kdagger*a^k
        
     
        Comm_Ej_2nd_approx=Comm_A0_A2+Comm_A1+Comm_A1_A3+Comm_Ap_Ej_sum+Comm_Ap_Ap2_sum;
        
        
       
        
        # for counter in range(loop):
        
        Ej_star=Ej/2*np.exp(-(phi_j**2)/2)#prefactor of the driving term
        Eq_star=Eq*np.exp(-(phi_j**2)/2);
        
        Eq_prime=Eq*phi_j**2
         
        Exp_n=np.zeros(Nsteps,dtype=complex);  
          
          
       
             
        def drho(rho_i,t):
                
              comm_n_rho=-1j*(np.matmul(n,rho_i)-np.matmul(rho_i,n)); #commutator of rho, n
              comm_anh_rho=-1j*(np.matmul(L_n0,rho_i)-np.matmul(rho_i,L_n0));
            #  comm_adag_rho=-1j*(np.matmul(adagger,rho_i)-np.matmul(rho_i,adagger))
            # comm_a_rho=-1j*(np.matmul(a,rho_i)-np.matmul(rho_i,a))
            #  comm_ext_drive_rho=np.exp(1j*(-Omega)*t)*comm_adag_rho+np.exp(-1j*(-Omega)*t)*comm_a_rho
            
                                
                                
              
                #np.exp(-1j*(delta-Omega)*t)*
                  #commutator of rho, #adagger^k*B_k and c.c.  
              com_op_rho=-1j*((np.matmul(A_p_matrices[0],rho_i)-np.matmul(rho_i,A_p_matrices[0]))+\
                                    (np.matmul(A_pdag_matrices[0],rho_i)-np.matmul(rho_i, A_pdag_matrices[0])));
                  
              comm_Eq_2nd_approx_rho=-1j*(Comm_Ap_eq_sum @ rho_i-rho_i @ Comm_Ap_eq_sum)
              comm_Ej_2nd_approx_rho=-1j*( Comm_Ej_2nd_approx @ rho_i-rho_i @  Comm_Ej_2nd_approx)
                     
              comm=(-delta-(Eq_prime**2/omega_dc))*comm_n_rho+Eq_star*comm_anh_rho+Ej_star*phi_j*com_op_rho-(Eq_star**2/(2*omega_dc))*comm_Eq_2nd_approx_rho-(Ej_star**2/(4*omega_dc))*comm_Ej_2nd_approx_rho
              #comm=-delta*comm_n_rho+Eq_star*comm_anh_rho+Ej_star*phi_j*com_op_rho
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
        
        t_for_spec=np.linspace(t_fin,t_fin+tprime,Nsteps1)
        
        a_rho_adag=np.matmul(a,np.matmul(rho_soln[-1,:],adagger));
        rho_g2=odeintw(drho,a_rho_adag,t_for_spec);
        
        xvec=np.linspace(-5,5,3000)
        yvec=np.linspace(-5,5,3000)
        
        rho_final=rho_soln[-1,:,:]
        
        #print(len(rho_final))
        
        N = int(len(rho_final))
        rho_final = rho_final.reshape((N, N))
        
        #print(rho_final)
        
        
        
        rho_qutip = qt.Qobj(rho_final)
        
        wigner_function = qt.wigner(rho_qutip, xvec, yvec);
        
        wmap = qt.wigner_cmap(wigner_function) 
        
        
        
        #im=plt.imshow(wigner_function, extent=[xvec[0], xvec[-1], yvec[0], yvec[-1]], cmap=wmap, interpolation='nearest')
        plt1 = plt.contourf(xvec, xvec,wigner_function ,100, cmap=wmap)
        axes=plt.gca();
        plt.xlabel('x');
        plt.ylabel('p');
        cb = plt.colorbar(plt1,ax=axes)
        plt.show();
        
        
        for trr in range(Nsteps):
             Exp_n[trr]=np.trace(np.matmul(n,rho_soln[trr,:]));
             
        fano_int=0;     
        g2=np.zeros(Nsteps1,dtype=complex);
        for tr in range(Nsteps1):
          n_g2=np.matmul(n,rho_g2[tr,:])
          g2[tr]=np.trace(n_g2)
          fano_int=fano_int+(g2[tr].real/Exp_n[-1].real**2-1)*dt  
          
        Fano[countiiii]=1+2*gamma*Exp_n[-1].real*fano_int;
        Ej_pl[countiiii]=Ej
        Ej=Ej+1
          
     
    if save==1: 
         Fano_factor=open('2nd_RWA/Fano_Eq_'+str(Eq)+'_phi_j_'+str(round(phi_j))+'_del_'+str(delta)+'_ratio_eq_wdc_'+str(ratio)+' _save.txt',"w")
         
         for ps in range(loop):
             Fano_factor.write('%.11f\t'%Ej_pl[ps]+'%.11f\n'%Fano[ps].real);
             
             
             
        
    #plt.plot(Ej_pl,Fano.real)
    #axes=plt.gca();
    #axes.set_xlim(0,50)