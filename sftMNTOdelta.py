# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 09:30:31 2025

@author: Thalia
"""

# =============================================================================
# BIBLIOTECAS E CONFIGURAÇÕES INICIAIS
# =============================================================================
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import quad, dblquad
from scipy.optimize import differential_evolution
import time
import io
import warnings

# Suprimir avisos de integração que podem poluir a saída.
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# =============================================================================
# CSS PERSONALIZADO (COMPONENTES VERDES PARA COMBINAR COM O RANDOM)
# =============================================================================
st.markdown("""
<style>
    /* Cor da checkbox */
    .stCheckbox [data-baseweb="checkbox"] > div:first-child {
        border-color: green !important;
    }
    .stCheckbox [data-baseweb="checkbox"] > div > svg {
        fill: green !important;
    }

    /* Cor do botão principal */
    .stButton>button {
        border: 2px solid #2E8B57;
        background-color: #2E8B57;
        color: white;
    }
    .stButton>button:hover {
        border-color: #3CB371;
        background-color: #3CB371;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# LAYOUT SUPERIOR – CABEÇALHO
# =============================================================================
col1, col2 = st.columns([1, 4])
with col1:
    st.image("assets/logo_random.png", use_container_width=True)

with col2:
    st.markdown("""
        <div style='display: flex; align-items: center; height: 100%;'>
            <h1 style='color: darkgreen; text-align: left; font-size: 28px;'>
                Política de Inspeção e Manutenção Preventiva com Aproveitamento de Oportunidades (MNTδ)
            </h1>
        </div>
    """, unsafe_allow_html=True)

def policy(L,M,N,T,delta,beta_x,eta_x,beta_h,eta_h,lbda,Cp,Cop,Ci,Coi,Cf,Cep_max,delta_min,delta_lim,Dp,Df):
    
    C1 = (Cp - Cep_max)/(delta_lim - delta_min)
    C2 = Cep_max - C1*delta_min
    C3 = Cp
     
    def Cep(time_lag):
        if time_lag <= delta_lim:
            Cep_ = C1*time_lag + C2
        else:
            Cep_ = C3
        return (Cep_)
    
    Z = int(delta / T)
    Y = max(0, N - Z - 1)
    
    # Functions for X (time to defect arrival)
    def fx(x):
        return (beta_x / eta_x) * ((x / eta_x) ** (beta_x - 1)) * np.exp(-((x / eta_x) ** beta_x))
    def Rx(x):
        return np.exp(-((x / eta_x) ** beta_x))
    def Fx(x):
        return 1 - np.exp(-((x / eta_x) ** beta_x))

    # Functions for H (delay-time)
    def fh(h):
        return (beta_h / eta_h) * ((h / eta_h) ** (beta_h - 1)) * np.exp(-((h / eta_h) ** beta_h))
    def Rh(h):
        return np.exp(-((h / eta_h) ** beta_h))
    def Fh(h):
        return 1 - np.exp(-((h / eta_h) ** beta_h))

    # Functions for W (time between two consecutive opportunities)
    def fw(w):
        return lbda * np.exp(- lbda * w)
    def Rw(w):
        return np.exp(- lbda * w)
    def Fw(w):
        return 1 - np.exp(- lbda * w)
    
    def scenario_1(): 
        # Preventive replacement at NT, with system in good state
        P1 = Rx(N*T)*Rw((N-M)*T)
        EC1 = (Y*Ci + (M-L)*T*lbda*Coi + Cp)*P1
        EV1 = (N*T + Dp)*P1
        ED1 = Dp*P1
        return (P1, EC1, EV1, ED1)
    
    def scenario_2():
        # Opportunistic preventive replacement between MT and NT, with system in good state
        if (M < N) and (M < Y):
            P2_1 = 0; EC2_1 = 0; EV2_1 = 0
            for i in range(1, Y-M+1):
                prob2_1 = quad(lambda w: fw(w)*Rx(M*T + w), (i-1)*T, i*T)[0] 
                P2_1 = P2_1 + prob2_1
                EC2_1 = EC2_1 + ((M+i-1)*Ci + (M-L)*T*lbda*Coi + Cop)*prob2_1
                EV2_1 = EV2_1 + quad(lambda w: (M*T + w + Dp)*fw(w)*Rx(M*T + w), (i-1)*T, i*T)[0]
            
            P2_2 = quad(lambda w: fw(w)*Rx(M*T + w), (Y-M)*T, (N-M)*T)[0]
            EC2_2 = (Y*Ci + (M-L)*T*lbda*Coi + Cop)*P2_2
            EV2_2 = quad(lambda w: (M*T + w + Dp)*fw(w)*Rx(M*T + w), (Y-M)*T, (N-M)*T)[0]
            
            P2 = P2_1 + P2_2
            EC2 = EC2_1 + EC2_2
            EV2 = EV2_1 + EV2_2
            
            #EV2 = quad(lambda w: (M*T + w + Dp)*fw(w)*Rx(M*T + w), 0, (N-M)*T)[0]
            ED2 = Dp*P2
            
        if (M < N) and (M >= Y):
            P2 = quad(lambda w: fw(w)*Rx(M*T + w), 0, (N-M)*T)[0]       
            EC2 = (Y*Ci + (M-L)*T*lbda*Coi + Cop)*P2
            EV2 = quad(lambda w: (M*T + w + Dp)*fw(w)*Rx(M*T + w), 0, (N-M)*T)[0]
            ED2 = Dp*P2
        
        if (M == N):
            P2 = 0; EC2 = 0; EV2 = 0; ED2 = 0
        
        return (P2, EC2, EV2, ED2)
    
    def scenario_3():
        # Early preventive replacement after a positive in-house inspection (time lag delta)
        if (L >= 0) and (L < M) and (M < N) and (M < Y):
            P3_1 = 0; EC3_1 = 0; EV3_1 = 0
            for i in range(1, L+1):
                prob3_1 = quad(lambda x: fx(x)*Rh(i*T + delta - x)*Rw(delta), (i-1)*T, i*T)[0]
                P3_1 = P3_1 + prob3_1
                EC3_1 = EC3_1 + (i*Ci + Cep(delta))*prob3_1
                EV3_1 = EV3_1 + (i*T + delta + Dp)*prob3_1
                
            P3_2 = 0; EC3_2 = 0; EV3_2 = 0
            for i in range(L+1, M+1):
                prob3_2 = quad(lambda x: fx(x)*Rh(i*T + delta - x)*Rw(i*T + delta - x), (i-1)*T, i*T)[0]
                P3_2 = P3_2 + prob3_2
                EC3_2 = EC3_2 + quad(lambda x: (i*Ci + (x-L*T)*lbda*Coi + Cep(delta))*fx(x)*Rh(i*T + delta - x)*Rw(i*T + delta - x), (i-1)*T, i*T)[0]
                EV3_2 = EV3_2 + (i*T + delta + Dp)*prob3_2
            
            P3_3 = 0; EC3_3 = 0; EV3_3 = 0
            for i in range(M+1, Y+1):
                prob3_3 = quad(lambda x: fx(x)*Rh(i*T + delta - x)*Rw(i*T + delta - M*T), (i-1)*T, i*T)[0]
                P3_3 = P3_3 + prob3_3
                EC3_3 = EC3_3 + (i*Ci + (M-L)*T*lbda*Coi + Cep(delta))*prob3_3
                EV3_3 = EV3_3 + (i*T + delta + Dp)*prob3_3
            
            P3 = P3_1 + P3_2 + P3_3
            EC3 = EC3_1 + EC3_2 + EC3_3
            EV3 = EV3_1 + EV3_2 + EV3_3
            ED3 = Dp*P3
            
        if (L >= 0) and (L < M) and (M >= Y) and (L < Y):
            P3_1 = 0; EC3_1 = 0; EV3_1 = 0
            for i in range(1, L+1):
                prob3_1 = quad(lambda x: fx(x)*Rh(i*T + delta - x)*Rw(delta), (i-1)*T, i*T)[0]
                P3_1 = P3_1 + prob3_1
                EC3_1 = EC3_1 + (i*Ci + Cep(delta))*prob3_1
                EV3_1 = EV3_1 + (i*T + delta + Dp)*prob3_1
                
            P3_2 = 0; EC3_2 = 0; EV3_2 = 0
            for i in range(L+1, Y+1):
                prob3_2 = quad(lambda x: fx(x)*Rh(i*T + delta - x)*Rw(i*T + delta - x), (i-1)*T, i*T)[0]
                P3_2 = P3_2 + prob3_2
                EC3_2 = EC3_2 + quad(lambda x: (i*Ci + (x-L*T)*lbda*Coi + Cep(delta))*fx(x)*Rh(i*T + delta - x)*Rw(i*T + delta - x), (i-1)*T, i*T)[0]
                EV3_2 = EV3_2 + (i*T + delta + Dp)*prob3_2
            
            P3 = P3_1 + P3_2
            EC3 = EC3_1 + EC3_2
            EV3 = EV3_1 + EV3_2
            ED3 = Dp*P3
            
        if (L >= 0) and (L == M) and (M < Y):
            P3_1 = 0; EC3_1 = 0; EV3_1 = 0
            for i in range(1, L+1):
                prob3_1 = quad(lambda x: fx(x)*Rh(i*T + delta - x)*Rw(delta), (i-1)*T, i*T)[0]
                P3_1 = P3_1 + prob3_1
                EC3_1 = EC3_1 + (i*Ci + Cep(delta))*prob3_1
                EV3_1 = EV3_1 + (i*T + delta + Dp)*prob3_1
            
            P3_3 = 0; EC3_3 = 0; EV3_3 = 0
            for i in range(M+1, Y+1):
                prob3_3 = quad(lambda x: fx(x)*Rh(i*T + delta - x)*Rw(i*T + delta - M*T), (i-1)*T, i*T)[0]
                P3_3 = P3_3 + prob3_3
                EC3_3 = EC3_3 + (i*Ci + (M-L)*T*lbda*Coi + Cep(delta))*prob3_3
                EV3_3 = EV3_3 + (i*T + delta + Dp)*prob3_3
            
            P3 = P3_1 + P3_3
            EC3 = EC3_1 + EC3_3
            EV3 = EV3_1 + EV3_3
            ED3 = Dp*P3
            
        if (L >= Y) and (Y >= 1):
            P3 = 0; EC3 = 0; EV3 = 0
            for i in range(1, Y+1):
                prob3 = quad(lambda x: fx(x)*Rh(i*T + delta - x)*Rw(delta), (i-1)*T, i*T)[0]
                P3 = P3 + prob3
                EC3 = EC3 + (i*Ci + Cep(delta))*prob3
                EV3 = EV3 + (i*T + delta + Dp)*prob3
            ED3 = Dp*P3
            
        if (Y == 0):
            P3 = 0
            EC3 = 0
            EV3 = 0
            ED3 = 0
        
        return (P3, EC3, EV3, ED3)
    
    def scenario_4():
        #Opportunistic preventive replacement of a defective system
        if (L >= 0) and (L < M) and (M < N) and (M < Y):
            P4_1 = 0; EC4_1 = 0; EV4_1 = 0
            P4_2 = 0; EC4_2 = 0; EV4_2 = 0
            for i in range(1, L+1):
                #prob4_1 = 0
                prob4_2 = dblquad(lambda w, x: fx(x)*fw(w)*Rh(i*T+w-x), (i-1)*T, i*T, lambda x: 0, lambda x: delta)[0]
                #P4_1 = P4_1 + prob4_1
                P4_2 = P4_2 + prob4_2
                #EC4_1 = EC4_1 + ((i-1)*Ci + Cop)*prob4_1
                EC4_2 = EC4_2 + (i*Ci + Cop)*prob4_2
                #EV4_1 = EV4_1 + dblquad(lambda w, x: (x + w + Dp)*fx(x)*fw(w)*Rh(w), (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                EV4_2 = EV4_2 + dblquad(lambda w, x: (i*T + w + Dp)*fx(x)*fw(w)*Rh(i*T+w-x), (i-1)*T, i*T, lambda x: 0, lambda x: delta)[0]
                      
            P4_3 = sum(
                dblquad(lambda w, x: fx(x)*fw(w)*Rh(w), (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                for i in range(L+1,M+1))
            P4_4 = sum(
                dblquad(lambda w, x: fx(x)*fw(w)*Rh(w), (i-1)*T, i*T, lambda x: i*T-x, lambda x: i*T+delta-x)[0]
                for i in range(L+1,M+1))
            EC4_3 = sum(
                dblquad(lambda w, x: ((i-1)*Ci + (x-L*T)*lbda*Coi + Cop)*fx(x)*fw(w)*Rh(w), 
                        (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                for i in range(L+1,M+1))
            EC4_4 = sum(
                dblquad(lambda w, x: (i*Ci + (x-L*T)*lbda*Coi + Cop)*fx(x)*fw(w)*Rh(w), 
                        (i-1)*T, i*T, lambda x: i*T-x, lambda x: i*T+delta-x)[0]
                for i in range(L+1,M+1))
            EV4_3 = sum(
                dblquad(lambda w, x: (x + w + Dp)*fx(x)*fw(w)*Rh(w), (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                for i in range(L+1,M+1))
            EV4_4 = sum(
                dblquad(lambda w, x: (x + w + Dp)*fx(x)*fw(w)*Rh(w), (i-1)*T, i*T, lambda x: i*T-x, lambda x: i*T+delta-x)[0]
                for i in range(L+1,M+1))
            
            P4_5 = 0; EC4_5 = 0; EV4_5 = 0
            P4_6 = 0; EC4_6 = 0; EV4_6 = 0
            for i in range(M+1, Y+1):
                prob4_5 = dblquad(lambda w, x: fx(x)*fw(w)*Rh(M*T + w - x), (i-1)*T, i*T, lambda x: x-M*T, lambda x: (i-M)*T)[0]
                prob4_6 = dblquad(lambda w, x: fx(x)*fw(w)*Rh(M*T + w - x), (i-1)*T, i*T, lambda x: (i-M)*T, lambda x: (i-M)*T+delta)[0]
                P4_5 = P4_5 + prob4_5
                P4_6 = P4_6 + prob4_6
                EC4_5 = EC4_5 + ((i-1)*Ci + (M-L)*T*lbda*Coi + Cop)*prob4_5
                EC4_6 = EC4_6 + (i*Ci + (M-L)*T*lbda*Coi + Cop)*prob4_6
                EV4_5 = EV4_5 + dblquad(lambda w, x: (M*T + w + Dp)*fx(x)*fw(w)*Rh(M*T + w - x), (i-1)*T, i*T, lambda x: x-M*T, lambda x: (i-M)*T)[0]
                EV4_6 = EV4_6 + dblquad(lambda w, x: (M*T + w + Dp)*fx(x)*fw(w)*Rh(M*T + w - x), (i-1)*T, i*T, lambda x: (i-M)*T, lambda x: (i-M)*T+delta)[0]

            P4_7 = dblquad(lambda w, x: fx(x)*fw(w)*Rh(M*T + w - x), Y*T, N*T, lambda x: x-M*T, lambda x: (N-M)*T)[0]
            EC4_7 = (Y*Ci + (M-L)*T*lbda*Coi + Cop)*P4_7
            EV4_7 = dblquad(lambda w, x: (M*T + w + Dp)*fx(x)*fw(w)*Rh(M*T + w - x), Y*T, N*T, lambda x: x-M*T, lambda x: (N-M)*T)[0]
                 
            P4 = P4_1 + P4_2 + P4_3 + P4_4 + P4_5 + P4_6 + P4_7
            EC4 = EC4_1 + EC4_2 + EC4_3 + EC4_4 + EC4_5 + EC4_6 + EC4_7
            EV4 = EV4_1 + EV4_2 + EV4_3 + EV4_4 + EV4_5 + EV4_6 + EV4_7
            ED4 = Dp*P4
            
        if (L >= 0) and (L < M) and (M >= Y) and (Y > L):
            P4_1 = 0; EC4_1 = 0; EV4_1 = 0
            P4_2 = 0; EC4_2 = 0; EV4_2 = 0
            for i in range(1, L+1):
                #prob4_1 = dblquad(lambda w, x: fx(x)*fw(w)*Rh(w), (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                prob4_2 = dblquad(lambda w, x: fx(x)*fw(w)*Rh(i*T+w-x), (i-1)*T, i*T, lambda x: 0, lambda x: delta)[0]
                #P4_1 = P4_1 + prob4_1
                P4_2 = P4_2 + prob4_2
                #EC4_1 = EC4_1 + ((i-1)*Ci + Cop)*prob4_1
                EC4_2 = EC4_2 + (i*Ci + Cop)*prob4_2
                #EV4_1 = EV4_1 + dblquad(lambda w, x: (x + w + Dp)*fx(x)*fw(w)*Rh(w), (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                EV4_2 = EV4_2 + dblquad(lambda w, x: (i*T + w + Dp)*fx(x)*fw(w)*Rh(i*T+w-x), (i-1)*T, i*T, lambda x: 0, lambda x: delta)[0]
                      
            P4_3 = sum(
                dblquad(lambda w, x: fx(x)*fw(w)*Rh(w), (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                for i in range(L+1,Y+1))
            P4_4 = sum(
                dblquad(lambda w, x: fx(x)*fw(w)*Rh(w), (i-1)*T, i*T, lambda x: i*T-x, lambda x: i*T+delta-x)[0]
                for i in range(L+1,Y+1))
            EC4_3 = sum(
                dblquad(lambda w, x: ((i-1)*Ci + (x-L*T)*lbda*Coi + Cop)*fx(x)*fw(w)*Rh(w), 
                        (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                for i in range(L+1,Y+1))
            EC4_4 = sum(
                dblquad(lambda w, x: (i*Ci + (x-L*T)*lbda*Coi + Cop)*fx(x)*fw(w)*Rh(w), 
                        (i-1)*T, i*T, lambda x: i*T-x, lambda x: i*T+delta-x)[0]
                for i in range(L+1,Y+1))
            EV4_3 = sum(
                dblquad(lambda w, x: (x + w + Dp)*fx(x)*fw(w)*Rh(w), (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                for i in range(L+1,Y+1))
            EV4_4 = sum(
                dblquad(lambda w, x: (x + w + Dp)*fx(x)*fw(w)*Rh(w), (i-1)*T, i*T, lambda x: i*T-x, lambda x: i*T+delta-x)[0]
                for i in range(L+1,Y+1))
            
            
            P4_5 = dblquad(lambda w, x: fx(x)*fw(w)*Rh(w), Y*T, M*T, lambda x: 0, lambda x: N*T-x)[0]
            P4_6 = dblquad(lambda w, x: fx(x)*fw(w)*Rh(M*T + w - x), M*T, N*T, lambda x: x-M*T, lambda x: (N-M)*T)[0]
            EC4_5 = dblquad(lambda w, x: (Y*Ci + (x-L*T)*lbda*Coi + Cop)*fx(x)*fw(w)*Rh(w), Y*T, M*T, lambda x: 0, lambda x: N*T-x)[0]
            EC4_6 = (Y*Ci + (M-L)*T*lbda*Coi + Cop)*P4_6
            EV4_5 = dblquad(lambda w, x: (x + w + Dp)*fx(x)*fw(w)*Rh(w), Y*T, M*T, lambda x: 0, lambda x: N*T-x)[0]
            EV4_6 = dblquad(lambda w, x: (M*T + w + Dp)*fx(x)*fw(w)*Rh(M*T + w - x), M*T, N*T, lambda x: x-M*T, lambda x: (N-M)*T)[0]
                
            P4 = P4_1 + P4_2 + P4_3 + P4_4 + P4_5 + P4_6
            EC4 = EC4_1 + EC4_2 + EC4_3 + EC4_4 + EC4_5 + EC4_6
            EV4 = EV4_1 + EV4_2 + EV4_3 + EV4_4 + EV4_5 + EV4_6 
            ED4 = Dp*P4
            
        if (L >= 0) and (L == M) and (M < Y):
            P4_1 = 0; EC4_1 = 0; EV4_1 = 0
            P4_2 = 0; EC4_2 = 0; EV4_2 = 0
            for i in range(1, L+1):
                #prob4_1 = dblquad(lambda w, x: fx(x)*fw(w)*Rh(w), (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                prob4_2 = dblquad(lambda w, x: fx(x)*fw(w)*Rh(i*T+w-x), (i-1)*T, i*T, lambda x: 0, lambda x: delta)[0]
                #P4_1 = P4_1 + prob4_1
                P4_2 = P4_2 + prob4_2
                #EC4_1 = EC4_1 + ((i-1)*Ci + Cop)*prob4_1
                EC4_2 = EC4_2 + (i*Ci + Cop)*prob4_2
                #EV4_1 = EV4_1 + dblquad(lambda w, x: (x + w + Dp)*fx(x)*fw(w)*Rh(w), (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                EV4_2 = EV4_2 + dblquad(lambda w, x: (i*T + w + Dp)*fx(x)*fw(w)*Rh(i*T+w-x), (i-1)*T, i*T, lambda x: 0, lambda x: delta)[0]
            
            P4_3 = 0; EC4_3 = 0; EV4_3 = 0
            P4_4 = 0; EC4_4 = 0; EV4_4 = 0
            for i in range(L+1, Y+1):
                prob4_3 = dblquad(lambda w, x: fx(x)*fw(w)*Rh(M*T + w - x), (i-1)*T, i*T, lambda x: x-M*T, lambda x: (i-M)*T)[0]
                prob4_4 = dblquad(lambda w, x: fx(x)*fw(w)*Rh(M*T + w - x), (i-1)*T, i*T, lambda x: (i-M)*T, lambda x: (i-M)*T+delta)[0]
                P4_3 = P4_3 + prob4_3
                P4_4 = P4_4 + prob4_4
                EC4_3 = EC4_3 + ((i-1)*Ci + Cop)*prob4_3
                EC4_4 = EC4_4 + (i*Ci + Cop)*prob4_4
                EV4_3 = EV4_3 + dblquad(lambda w, x: (M*T + w + Dp)*fx(x)*fw(w)*Rh(M*T + w - x), (i-1)*T, i*T, lambda x: x-M*T, lambda x: (i-M)*T)[0]
                EV4_4 = EV4_4 + dblquad(lambda w, x: (M*T + w + Dp)*fx(x)*fw(w)*Rh(M*T + w - x), (i-1)*T, i*T, lambda x: (i-M)*T, lambda x: (i-M)*T+delta)[0] 
                
            P4_5 = dblquad(lambda w, x: fx(x)*fw(w)*Rh(M*T + w - x), Y*T, N*T, lambda x: x-M*T, lambda x: (N-M)*T)[0]
            EC4_5 = (Y*Ci + Cop)*dblquad(lambda w, x: fx(x)*fw(w)*Rh(M*T + w - x), Y*T, N*T, lambda x: x-M*T, lambda x: (N-M)*T)[0]
            EV4_5 = dblquad(lambda w, x: (M*T + w + Dp)*fx(x)*fw(w)*Rh(M*T + w - x), Y*T, N*T, lambda x: x-M*T, lambda x: (N-M)*T)[0]
            
            P4 = P4_1 + P4_2 + P4_3 + P4_4 + P4_5
            EC4 = EC4_1 + EC4_2 + EC4_3 + EC4_4 + EC4_5
            EV4 = EV4_1 + EV4_2 + EV4_3 + EV4_4 + EV4_5
            ED4 = Dp*P4
           
        if (Y >= 1) and (Y <= L):
            P4_1 = 0; EC4_1 = 0; EV4_1 = 0
            P4_2 = 0; EC4_2 = 0; EV4_2 = 0
            for i in range(1, Y+1):
                #prob4_1 = dblquad(lambda w, x: fx(x)*fw(w)*Rh(w), (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                prob4_2 = dblquad(lambda w, x: fx(x)*fw(w)*Rh(i*T+w-x), (i-1)*T, i*T, lambda x: 0, lambda x: delta)[0]
                #P4_1 = P4_1 + prob4_1
                P4_2 = P4_2 + prob4_2
                #EC4_1 = EC4_1 + ((i-1)*Ci + Cop)*prob4_1
                EC4_2 = EC4_2 + (i*Ci + Cop)*prob4_2
                #EV4_1 = EV4_1 + dblquad(lambda w, x: (x + w + Dp)*fx(x)*fw(w)*Rh(w), (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                EV4_2 = EV4_2 + dblquad(lambda w, x: (i*T + w + Dp)*fx(x)*fw(w)*Rh(i*T+w-x), (i-1)*T, i*T, lambda x: 0, lambda x: delta)[0]
                
            P4_3 = dblquad(lambda w, x: fx(x)*fw(w)*Rh(L*T+w-x), Y*T, L*T, lambda x: 0, lambda x: (N-L)*T)[0]
            P4_4 = dblquad(lambda w, x: fx(x)*fw(w)*Rh(w), L*T, M*T, lambda x: 0, lambda x: N*T-x)[0]
            P4_5 = dblquad(lambda w, x: fx(x)*fw(w)*Rh(M*T + w - x), M*T, N*T, lambda x: x-M*T, lambda x: (N-M)*T)[0]
            P4 = P4_1 + P4_2 + P4_3 + P4_4 + P4_5
            
            EC4_3 = (Y*Ci + Cop)*P4_3
            EC4_4 = dblquad(lambda w, x: (Y*Ci + (x-L*T)*lbda*Coi + Cop)*fx(x)*fw(w)*Rh(w), L*T, M*T, lambda x: 0, lambda x: N*T-x)[0]
            EC4_5 = (Y*Ci + (M-L)*T*lbda*Coi + Cop)*P4_5
            EC4 = EC4_1 + EC4_2 + EC4_3 + EC4_4 + EC4_5
            
            EV4_3 = dblquad(lambda w, x: (L*T + w + Dp)*fx(x)*fw(w)*Rh(L*T+w-x), Y*T, L*T, lambda x: 0, lambda x: (N-L)*T)[0]
            EV4_4 = dblquad(lambda w, x: (x + w + Dp)*fx(x)*fw(w)*Rh(w), L*T, M*T, lambda x: 0, lambda x: N*T-x)[0]
            EV4_5 = dblquad(lambda w, x: (M*T + w + Dp)*fx(x)*fw(w)*Rh(M*T + w - x), M*T, N*T, lambda x: x-M*T, lambda x: (N-M)*T)[0]
            EV4 = EV4_1 + EV4_2 + EV4_3 + EV4_4 + EV4_5
            
            ED4 = Dp*P4
              
        if (Y == 0):
            P4_1 = dblquad(lambda w, x: fx(x)*fw(w)*Rh(L*T+w-x), 0, L*T, lambda x: 0, lambda x: (N-L)*T)[0]
            P4_2 = dblquad(lambda w, x: fx(x)*fw(w)*Rh(w), L*T, M*T, lambda x: 0, lambda x: N*T-x)[0]
            P4_3 = dblquad(lambda w, x: fx(x)*fw(w)*Rh(M*T + w - x), M*T, N*T, lambda x: x-M*T, lambda x: (N-M)*T)[0]
                
            P4 = P4_1 + P4_2 + P4_3
            
            EC4_1 = Cop*P4_1
            EC4_2 = dblquad(lambda w, x: ((x-L*T)*lbda*Coi + Cop)*fx(x)*fw(w)*Rh(w), L*T, M*T, lambda x: 0, lambda x: N*T-x)[0]
            EC4_3 = ((M-L)*T*lbda*Coi + Cop)*P4_3
                
            EC4 = EC4_1 + EC4_2 + EC4_3
            
            EV4_1 = dblquad(lambda w, x: (L*T + w + Dp)*fx(x)*fw(w)*Rh(L*T+w-x), 0, L*T, lambda x: 0, lambda x: (N-L)*T)[0]
            EV4_2 = dblquad(lambda w, x: (x + w + Dp)*fx(x)*fw(w)*Rh(w), L*T, M*T, lambda x: 0, lambda x: N*T-x)[0]
            EV4_3 = dblquad(lambda w, x: (M*T + w + Dp)*fx(x)*fw(w)*Rh(M*T + w - x), M*T, N*T, lambda x: x-M*T, lambda x: (N-M)*T)[0]
                
            EV4 = EV4_1 + EV4_2 + EV4_3
            
            ED4 = Dp*P4
        
        return (P4, EC4, EV4, ED4)
    
    def scenario_5():
        # Preventive replacement at N.T with system in defective state
        if (Y <= L):
            P5_1 = quad(lambda x: fx(x)*Rh(N*T-x)*Rw((N-L)*T), Y*T, L*T)[0]
            P5_2 = quad(lambda x: fx(x)*Rh(N*T-x)*Rw(N*T-x), L*T, M*T)[0]
            P5_3 = quad(lambda x: fx(x)*Rh(N*T-x)*Rw((N-M)*T), M*T, N*T)[0]
            
            P5 = P5_1 + P5_2 + P5_3
            
            EC5_1 = (Y*Ci + Cp)*P5_1
            EC5_2 = quad(lambda x: (Y*Ci + (x - L*T)*lbda*Coi + Cp)*fx(x)*Rh(N*T-x)*Rw(N*T-x), L*T, M*T)[0]
            EC5_3 = (Y*Ci + (M-L)*T*lbda*Coi + Cp)*P5_3
            
            EC5 = EC5_1 + EC5_2 + EC5_3
            
            EV5 = (N*T + Dp)*P5
            ED5 = Dp*P5
            
        if (L < Y) and (Y <= M):
            P5_1 = quad(lambda x: fx(x)*Rh(N*T-x)*Rw(N*T-x), Y*T, M*T)[0]
            P5_2 = quad(lambda x: fx(x)*Rh(N*T-x)*Rw((N-M)*T), M*T, N*T)[0]
            
            P5 = P5_1 + P5_2
            
            EC5_1 = quad(lambda x: (Y*Ci + (x - L*T)*lbda*Coi + Cp)*fx(x)*Rh(N*T-x)*Rw(N*T-x), Y*T, M*T)[0]
            EC5_2 = (Y*Ci + (M-L)*T*lbda*Coi + Cp)*P5_2
            
            EC5 = EC5_1 + EC5_2
            
            EV5 = (N*T + Dp)*P5
            ED5 = Dp*P5
            
        if (Y >= M):
            P5 = quad(lambda x: fx(x)*Rh(N*T-x)*Rw((N-M)*T), Y*T, N*T)[0]

            EC5 = (Y*Ci + (M-L)*T*lbda*Coi + Cp)*P5
            
            EV5 = (N*T + Dp)*P5
            ED5 = Dp*P5
            
        return(P5, EC5, EV5, ED5)
    
    def scenario_6():
        if (L >= 0) and (L < M) and (M < N) and (M < Y):
            P6_1 = 0; EC6_1 = 0; EV6_1 = 0
            P6_2 = 0; EC6_2 = 0; EV6_2 = 0
            for i in range(1, L+1):
                prob6_1 = dblquad(lambda h, x: fx(x)*fh(h), (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                prob6_2 = dblquad(lambda h, x: fx(x)*fh(h)*Rw(x+h-i*T), (i-1)*T, i*T, lambda x: i*T-x, lambda x: i*T+delta-x)[0]
                P6_1 = P6_1 + prob6_1
                P6_2 = P6_2 + prob6_2
                EC6_1 = EC6_1 + ((i-1)*Ci + Cf)*prob6_1
                EC6_2 = EC6_2 + (i*Ci + Cf)*prob6_2
                EV6_1 = EV6_1 + dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h), (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                EV6_2 = EV6_2 + dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h)*Rw(x+h-i*T), (i-1)*T, i*T, lambda x: i*T-x, lambda x: i*T+delta-x)[0]
            
            P6_3 = sum(
                dblquad(lambda h, x: fx(x)*fh(h)*Rw(h), (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                for i in range(L+1,M+1))
            P6_4 = sum(
                dblquad(lambda h, x: fx(x)*fh(h)*Rw(h), (i-1)*T, i*T, lambda x: i*T-x, lambda x: i*T+delta-x)[0]
                for i in range(L+1,M+1))
            EC6_3 = sum(
                dblquad(lambda h, x: ((i-1)*Ci + (x- L*T)*lbda*Coi + Cf)*fx(x)*fh(h)*Rw(h), 
                        (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                for i in range(L+1,M+1))
            EC6_4 = sum(
                dblquad(lambda h, x: (i*Ci + (x- L*T)*lbda*Coi + Cf)*fx(x)*fh(h)*Rw(h), 
                        (i-1)*T, i*T, lambda x: i*T-x, lambda x: i*T+delta-x)[0]
                for i in range(L+1,M+1))
            EV6_3 = sum(
                dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h)*Rw(h), (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                for i in range(L+1,M+1))
            EV6_4 = sum(
                dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h)*Rw(h), (i-1)*T, i*T, lambda x: i*T-x, lambda x: i*T+delta-x)[0]
                for i in range(L+1,M+1))
            
            P6_5 = 0; EC6_5 = 0; EV6_5 = 0
            P6_6 = 0; EC6_6 = 0; EV6_6 = 0
            for i in range(M+1, Y+1):
                prob6_5 = dblquad(lambda h, x: fx(x)*fh(h)*Rw(x+h-M*T), (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                prob6_6 = dblquad(lambda h, x: fx(x)*fh(h)*Rw(x+h-M*T), (i-1)*T, i*T, lambda x: i*T-x, lambda x: i*T+delta-x)[0]
                P6_5 = P6_5 + prob6_5
                P6_6 = P6_6 + prob6_6
                EC6_5 = EC6_5 + ((i-1)*Ci + (M-L)*T*lbda*Coi + Cf)*prob6_5
                EC6_6 = EC6_6 + (i*Ci + (M-L)*T*lbda*Coi + Cf)*prob6_6
                EV6_5 = EV6_5 + dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h)*Rw(x+h-M*T), (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                EV6_6 = EV6_6 + dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h)*Rw(x+h-M*T), (i-1)*T, i*T, lambda x: i*T-x, lambda x: i*T+delta-x)[0]

            P6_7 = dblquad(lambda h, x: fx(x)*fh(h)*Rw(x+h-M*T), Y*T, N*T, lambda x: 0, lambda x: N*T-x)[0]
            P6 = P6_1 + P6_2 + P6_3 + P6_4 + P6_5 + P6_6 + P6_7
            
            EC6_7 = (Y*Ci + (M-L)*T*lbda*Coi + Cf)*P6_7
            EC6 = EC6_1 + EC6_2 + EC6_3 + EC6_4 + EC6_5 + EC6_6 + EC6_7
            
            EV6_7 = dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h)*Rw(x+h-M*T), Y*T, N*T, lambda x: 0, lambda x: N*T-x)[0]
            EV6 = EV6_1 + EV6_2 + EV6_3 + EV6_4 + EV6_5 + EV6_6 + EV6_7
            
            ED6 = Df*P6
            
        if (L >= 0) and (L < M) and (M >= Y) and (Y > L):
            P6_1 = 0; EC6_1 = 0; EV6_1 = 0
            P6_2 = 0; EC6_2 = 0; EV6_2 = 0
            for i in range(1, L+1):
                prob6_1 = dblquad(lambda h, x: fx(x)*fh(h), (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                prob6_2 = dblquad(lambda h, x: fx(x)*fh(h)*Rw(x+h-i*T), (i-1)*T, i*T, lambda x: i*T-x, lambda x: i*T+delta-x)[0]
                P6_1 = P6_1 + prob6_1
                P6_2 = P6_2 + prob6_2
                EC6_1 = EC6_1 + ((i-1)*Ci + Cf)*prob6_1
                EC6_2 = EC6_2 + (i*Ci + Cf)*prob6_2
                EV6_1 = EV6_1 + dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h), (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                EV6_2 = EV6_2 + dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h)*Rw(x+h-i*T), (i-1)*T, i*T, lambda x: i*T-x, lambda x: i*T+delta-x)[0]
            
            P6_3 = sum(
                dblquad(lambda h, x: fx(x)*fh(h)*Rw(h), (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                for i in range(L+1,Y+1))
            P6_4 = sum(
                dblquad(lambda h, x: fx(x)*fh(h)*Rw(h), (i-1)*T, i*T, lambda x: i*T-x, lambda x: i*T+delta-x)[0]
                for i in range(L+1,Y+1))
            P6_5 = dblquad(lambda h, x: fx(x)*fh(h)*Rw(h), Y*T, M*T, lambda x: 0, lambda x: N*T-x)[0]
            P6_6 = dblquad(lambda h, x: fx(x)*fh(h)*Rw(x+h-M*T), M*T, N*T, lambda x: 0, lambda x: N*T-x)[0]
            P6 = P6_1 + P6_2 + P6_3 + P6_4 + P6_5 + P6_6
            
            EC6_3 = sum(
                dblquad(lambda h, x: ((i-1)*Ci + (x- L*T)*lbda*Coi + Cf)*fx(x)*fh(h)*Rw(h), 
                        (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                for i in range(L+1,Y+1))
            EC6_4 = sum(
                dblquad(lambda h, x: (i*Ci + (x- L*T)*lbda*Coi + Cf)*fx(x)*fh(h)*Rw(h), 
                        (i-1)*T, i*T, lambda x: i*T-x, lambda x: i*T+delta-x)[0]
                for i in range(L+1,Y+1))
            EC6_5 = dblquad(lambda h, x: (Y*Ci + (x - L*T)*lbda*Coi + Cf)*fx(x)*fh(h)*Rw(h), Y*T, M*T, lambda x: 0, lambda x: N*T-x)[0]
            EC6_6 = (Y*Ci + (M-L)*T*lbda*Coi + Cf)*P6_6
            EC6 = EC6_1 + EC6_2 + EC6_3 + EC6_4 + EC6_5 + EC6_6
            
            EV6_3 = sum(
                dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h)*Rw(h), (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                for i in range(L+1,Y+1))
            EV6_4 = sum(
                dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h)*Rw(h), (i-1)*T, i*T, lambda x: i*T-x, lambda x: i*T+delta-x)[0]
                for i in range(L+1,Y+1))
            EV6_5 = dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h)*Rw(h), Y*T, M*T, lambda x: 0, lambda x: N*T-x)[0]
            EV6_6 = dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h)*Rw(x+h-M*T), M*T, N*T, lambda x: 0, lambda x: N*T-x)[0]
            EV6 = EV6_1 + EV6_2 + EV6_3 + EV6_4 + EV6_5 + EV6_6
            
            ED6 = Df*P6
            
        if (L >= 0) and (L == M) and (M < Y):
            P6_1 = 0; EC6_1 = 0; EV6_1 = 0
            P6_2 = 0; EC6_2 = 0; EV6_2 = 0
            for i in range(1, L+1):
                prob6_1 = dblquad(lambda h, x: fx(x)*fh(h), (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                prob6_2 = dblquad(lambda h, x: fx(x)*fh(h)*Rw(x+h-i*T), (i-1)*T, i*T, lambda x: i*T-x, lambda x: i*T+delta-x)[0]
                P6_1 = P6_1 + prob6_1
                P6_2 = P6_2 + prob6_2
                EC6_1 = EC6_1 + ((i-1)*Ci + Cf)*prob6_1
                EC6_2 = EC6_2 + (i*Ci + Cf)*prob6_2
                EV6_1 = EV6_1 + dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h), (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                EV6_2 = EV6_2 + dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h)*Rw(x+h-i*T), (i-1)*T, i*T, lambda x: i*T-x, lambda x: i*T+delta-x)[0]
            
            P6_5 = 0; EC6_5 = 0; EV6_5 = 0
            P6_6 = 0; EC6_6 = 0; EV6_6 = 0
            for i in range(M+1, Y+1):
                prob6_5 = dblquad(lambda h, x: fx(x)*fh(h)*Rw(x+h-M*T), (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                prob6_6 = dblquad(lambda h, x: fx(x)*fh(h)*Rw(x+h-M*T), (i-1)*T, i*T, lambda x: i*T-x, lambda x: i*T+delta-x)[0]
                P6_5 = P6_5 + prob6_5
                P6_6 = P6_6 + prob6_6
                EC6_5 = EC6_5 + ((i-1)*Ci + (M-L)*T*lbda*Coi + Cf)*prob6_5
                EC6_6 = EC6_6 + (i*Ci + (M-L)*T*lbda*Coi + Cf)*prob6_6
                EV6_5 = EV6_5 + dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h)*Rw(x+h-M*T), (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                EV6_6 = EV6_6 + dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h)*Rw(x+h-M*T), (i-1)*T, i*T, lambda x: i*T-x, lambda x: i*T+delta-x)[0]

            P6_7 = dblquad(lambda h, x: fx(x)*fh(h)*Rw(x+h-M*T), Y*T, N*T, lambda x: 0, lambda x: N*T-x)[0]
            P6 = P6_1 + P6_2 + P6_5 + P6_6 + P6_7
            
            EC6_7 = (Y*Ci + (M-L)*T*lbda*Coi + Cf)*P6_7
            EC6 = EC6_1 + EC6_2 + EC6_5 + EC6_6 + EC6_7
            
            EV6_7 = dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h)*Rw(x+h-M*T), Y*T, N*T, lambda x: 0, lambda x: N*T-x)[0]
            EV6 = EV6_1 + EV6_2 + EV6_5 + EV6_6 + EV6_7
            
            ED6 = Df*P6
            
        if (Y >= 1) and (Y <= L):
            P6_1 = 0; EC6_1 = 0; EV6_1 = 0
            P6_2 = 0; EC6_2 = 0; EV6_2 = 0
            for i in range(1,Y+1):
                prob6_1 = dblquad(lambda h, x: fx(x)*fh(h), (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                prob6_2 = dblquad(lambda h, x: fx(x)*fh(h)*Rw(x+h-i*T), (i-1)*T, i*T, lambda x: i*T-x, lambda x: i*T+delta-x)[0]
                P6_1 = P6_1 + prob6_1
                P6_2 = P6_2 + prob6_2
                EC6_1 = EC6_1 + ((i-1)*Ci + Cf)*prob6_1
                EC6_2 = EC6_2 + (i*Ci + Cf)*prob6_2
                EV6_1 = EV6_1 + dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h), (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                EV6_2 = EV6_2 + dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h)*Rw(x+h-i*T), (i-1)*T, i*T, lambda x: i*T-x, lambda x: i*T+delta-x)[0]

            P6_3 = dblquad(lambda h, x: fx(x)*fh(h), Y*T, L*T, lambda x: 0, lambda x: L*T-x)[0]
            P6_4 = dblquad(lambda h, x: fx(x)*fh(h)*Rw(x+h-L*T), Y*T, L*T, lambda x: L*T-x, lambda x: N*T-x)[0]
            P6_5 = dblquad(lambda h, x: fx(x)*fh(h)*Rw(h), L*T, M*T, lambda x: 0, lambda x: N*T-x)[0]
            P6_6 = dblquad(lambda h, x: fx(x)*fh(h)*Rw(x+h-M*T), M*T, N*T, lambda x: 0, lambda x: N*T-x)[0]
            P6 = P6_1 + P6_2 + P6_3 + P6_4 + P6_5 + P6_6
            
            EC6_3 = (Y*Ci + Cf)*P6_3
            EC6_4 = (Y*Ci + Cf)*P6_4
            EC6_5 = dblquad(lambda h, x: (Y*Ci + (x-L*T)*lbda*Coi + Cf)*fx(x)*fh(h)*Rw(h), L*T, M*T, lambda x: 0, lambda x: N*T-x)[0]
            EC6_6 = (Y*Ci + (M-L)*T*lbda*Coi + Cf)*P6_6
            EC6 = EC6_1 + EC6_2 + EC6_3 + EC6_4 + EC6_5 + EC6_6
            
            EV6_3 = dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h), Y*T, L*T, lambda x: 0, lambda x: L*T-x)[0]
            EV6_4 = dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h)*Rw(x+h-L*T), Y*T, L*T, lambda x: L*T-x, lambda x: N*T-x)[0]
            EV6_5 = dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h)*Rw(h), L*T, M*T, lambda x: 0, lambda x: N*T-x)[0]
            EV6_6 = dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h)*Rw(x+h-M*T), M*T, N*T, lambda x: 0, lambda x: N*T-x)[0]
            EV6 = EV6_1 + EV6_2 + EV6_3 + EV6_4 + EV6_5 + EV6_6
            
            ED6 = Df*P6
            
        if (Y == 0):
            P6_1 = dblquad(lambda h, x: fx(x)*fh(h), 0, L*T, lambda x: 0, lambda x: L*T-x)[0]
            P6_2 = dblquad(lambda h, x: fx(x)*fh(h)*Rw(x+h-L*T), 0, L*T, lambda x: L*T-x, lambda x: N*T-x)[0]
            P6_3 = dblquad(lambda h, x: fx(x)*fh(h)*Rw(h), L*T, M*T, lambda x: 0, lambda x: N*T-x)[0]
            P6_4 = dblquad(lambda h, x: fx(x)*fh(h)*Rw(x+h-M*T), M*T, N*T, lambda x: 0, lambda x: N*T-x)[0]
            P6 = P6_1 + P6_2 + P6_3 + P6_4
            
            EC6_1 = Cf*P6_1
            EC6_2 = Cf*P6_2
            EC6_3 = dblquad(lambda h, x: ((x-L*T)*lbda*Coi + Cf)*fx(x)*fh(h)*Rw(h), L*T, M*T, lambda x: 0, lambda x: N*T-x)[0]
            EC6_4 = ((M-L)*T*lbda*Coi + Cf)*P6_4
            EC6 = EC6_1 + EC6_2 + EC6_3 + EC6_4
            
            EV6_1 = dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h), 0, L*T, lambda x: 0, lambda x: L*T-x)[0]
            EV6_2 = dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h)*Rw(x+h-L*T), 0, L*T, lambda x: L*T-x, lambda x: N*T-x)[0]
            EV6_3 = dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h)*Rw(h), L*T, M*T, lambda x: 0, lambda x: N*T-x)[0]
            EV6_4 = dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h)*Rw(x+h-M*T), M*T, N*T, lambda x: 0, lambda x: N*T-x)[0]
            EV6 = EV6_1 + EV6_2 + EV6_3 + EV6_4
            
            ED6 = Df*P6
            
        return (P6, EC6, EV6, ED6)
    
    (P1, EC1, EV1, ED1) = scenario_1()
    (P2, EC2, EV2, ED2) = scenario_2()
    (P3, EC3, EV3, ED3) = scenario_3()        
    (P4, EC4, EV4, ED4) = scenario_4()        
    (P5, EC5, EV5, ED5) = scenario_5()        
    (P6, EC6, EV6, ED6) = scenario_6()
    
    P_total = P1 + P2 + P3 + P4 + P5 + P6
    EC = EC1 + EC2 + EC3 + EC4 + EC5 + EC6
    EV = EV1 + EV2 + EV3 + EV4 + EV5 + EV6
    ED = ED1 + ED2 + ED3 + ED4 + ED5 + ED6
    
    cost_rate = EC/EV
    MTBOF = EV/P6
    availability = 1 - (ED/EV)
    
    
    return (P_total, EC, EV, ED, cost_rate, MTBOF, availability, P1, P2, P3, P4, P5, P6)
def calcular_metricas_completas(T, N, M, delta, params):
    """
    Calcula Custo, Disponibilidade e MTBOF para a política.
    Retorna None se os parâmetros forem inválidos ou der erro.
    """
    try:
        # 1) Validações iniciais (CORRIGIDO: M > N em vez de M >= N)
        if M > N or N < 1 or T <= 0:
            return None
        L = M
        Z = int(delta / T)
        Y = max(0, N - Z - 1)
        if Y >= N:
            return None

        # 2) Extrai parâmetros
        betax, etax = params['betax'], params['etax']
        betah, etah = params['betah'], params['etah']
        lambd = params['lambd']
        Ci, Cp, Cop, Cf = params['Ci'], params['Cp'], params['Cop'], params['Cf']
        Dp, Df = params['Dp'], params['Df']
        Coi = Ci

        # 3) Executa os 6 cenários 
        p1, ec1, el1, ed1 = calcular_cenario1(
            T, N, M, Y, delta, Ci, Cp, Cop, Cf, Dp, Df,
            betax, etax, betah, etah, lambd
        )
        p2, ec2, el2, ed2 = calcular_cenario2(
            T, N, M, Y, delta, Ci, Cp, Cop, Cf, Dp, Df,
            betax, etax, betah, etah, lambd
        )
        p3, ec3, el3, ed3 = calcular_cenario3(
            T, N, M, Y, delta, Ci, Cp, Cop, Cf, Dp, Df,
            betax, etax, betah, etah, lambd
        )
        p4, ec4, el4, ed4 = calcular_cenario4(
            T, N, M, Y, delta, Ci, Cp, Cop, Cf, Dp, Df,
            betax, etax, betah, etah, lambd
        )
        p5, ec5, el5, ed5 = calcular_cenario5(
            T, N, M, Y, delta, Ci, Cp, Cop, Cf, Dp, Df,
            betax, etax, betah, etah, lambd
        )
        p6, ec6, el6, ed6 = calcular_cenario6(
            T, N, M, Y, delta, Ci, Cp, Cop, Cf, Dp, Df,
            betax, etax, betah, etah, lambd
        )

        # 4) Soma acumulada
        EC = ec1 + ec2 + ec3 + ec4 + ec5 + ec6
        EL = el1 + el2 + el3 + el4 + el5 + el6
        ED = ed1 + ed2 + ed3 + ed4 + ed5 + ed6

        # 5) Evita divisão por zero
        if EL <= 0:
            return None

        # 6) Calcula métricas finais
        custo = EC / EL
        disponibilidade = 1 - (ED / EL)
        # MTBOF = tempo médio até falha = EL / probabilidade de falha
        MTBOF = EL / (p6 if p6 > 0 else 1)

        return {
            "Custo": custo,
            "Disponibilidade": disponibilidade,
            "MTBOF": MTBOF
        }

    except Exception as e:
        print(f"[ERRO] Simulação falhou para T={T},N={N},M={M},δ={delta}: {e}")
        return None

# =============================================================================
# SEÇÃO DE PARÂMETROS DO MODELO
# =============================================================================
st.header("📥 Parâmetros do Modelo")

col_params1, col_params2 = st.columns(2)
with col_params1:
    betax = st.number_input(
        "Tempo até a chegada do defeito (X) – parâmetro de forma (Weibull)",
        format="%.7f", step=0.0000001
    )
    etax = st.number_input(
        "Tempo até a chegada do defeito (X) – parâmetro de escala (Weibull)",
        format="%.7f", step=0.0000001
    )
    lambd = st.number_input(
        "Taxa de chegada de oportunidades (λ)",
        format="%.7f", step=0.0000001
    )
    Cp = st.number_input(
        "Custo de substituição preventiva programada (Cp)",
        format="%.7f", step=0.0000001
    )
    Cop = st.number_input(
        "Custo de substituição preventiva oportuna (Cop)",
        format="%.7f", step=0.0000001
    )
    Dp = st.number_input(
        "Tempo de parada para substituição preventiva programada (Dp)",
        format="%.7f", step=0.0000001
    )

with col_params2:
    betah = st.number_input(
        "Tempo entre a chegada do defeito e a falha (H) – parâmetro de forma (Weibull)",
        format="%.7f", step=0.0000001
    )
    etah = st.number_input(
        "Tempo entre a chegada do defeito e a falha (H) – parâmetro de escala (Weibull)",
        format="%.7f", step=0.0000001
    )
    Cf = st.number_input(
        "Custo de substituição corretiva (Cf)",
        format="%.7f", step=0.0000001
    )
    Ci = st.number_input(
        "Custo de inspeção (Ci)",
        format="%.7f", step=0.0000001
    )
    Df = st.number_input(
        "Tempo de parada para substituição corretiva (Df)",
        format="%.7f", step=0.0000001
    )

st.markdown("---")
st.markdown("###### Parâmetros do Custo de Reposição Antecipada `Cep(δ)`")
st.markdown("Define o custo em função do tempo de postergação `δ`")

cep_col1, cep_col2, cep_col3 = st.columns(3)

with cep_col1:
    delta_min_ui = st.number_input(
        "δ Mínimo",  
        help="Define o menor tempo de resposta possível para uma manutenção após a detecção de um defeito.",
        format="%.7f",
        step=0.0000001
    )
with cep_col2:
    Cep_max_ui = st.number_input(
        "Custo para δ Mínimo (Cep_max)", 
        help="Custo da manutenção se realizada no tempo mais rápido possível (em δ Mínimo).",
        format="%.7f",
        step=0.0000001
    )
with cep_col3:
    delta_limite_ui = st.number_input(
        "δ Limite", 
        help="Limite de tempo. Para postergações (δ) maiores que este, o custo se torna o mesmo que o de uma preventiva programada (Cp).",
        format="%.7f",
        step=0.0000001
    )

# --- Montagem do dicionário 'params' ---
# 1. Primeiro, criei o dicionário com os valores lidos da interface.
params = {
    'betax': betax, 'etax': etax, 'betah': betah, 'etah': etah, 'lambd': lambd,
    'Cf': Cf, 'Cp': Cp, 'Cop': Cop, 'Ci': Ci, 'Df': Df, 'Dp': Dp,
}
# 2. DEPOIS, adicionei as chaves dos novos parâmetros de Cep ao dicionário JÁ EXISTENTE.
params['delta_min'] = delta_min_ui
params['Cep_max'] = Cep_max_ui
params['delta_limite'] = delta_limite_ui

st.markdown("---")

# =============================================================================
# SEÇÃO DE OTIMIZAÇÃO
# =============================================================================
st.header("⚙️ Otimização da Política")

if st.button("▶️ Iniciar Otimização"):
    
    # Define a função objetivo que o otimizador tentará minimizar.
    def objetivo(x):
        """
        Recebe um vetor 'x' com os parâmetros [T, M, N, delta],
        chama a simulação e retorna o custo.
        """
        # Desempacota os parâmetros da otimização
        T_val, M_val, N_val, delta_val = x

        # Converte M e N para inteiros
        M_val_int = int(round(M_val))
        N_val_int = int(round(N_val))

        # Chama a simulação
        resultado = calcular_metricas_completas(T_val, N_val_int, M_val_int, delta_val, params)

        # Se for inválido, retorna penalização alta
        if resultado is None or "Custo" not in resultado:
            return 1e9
        else:
            return resultado["Custo"]

    # Define os limites (bounds) para cada variável de otimização: [T, M, N, delta]
    # É uma boa prática garantir que o limite superior de M seja menor que o inferior de N.
    bounds = [
        (1.0, 200.0),  # T: entre 1 e 200
        (1, 20),       # M: entre 1 e 20
        (1, 20),       # N: entre 1 e 20 (permite N >= M)
        (params['delta_min'], 300.0)  # delta: entre delta_min e 300
    ]

    # Inicia a otimização com uma mensagem de espera
    with st.spinner("Otimizando a política... Por favor, aguarde. Este processo pode ser demorado."):
        start_time = time.time()
        resultado = differential_evolution(objetivo, bounds, maxiter=50, popsize=15, tol=0.01, disp=False)
        end_time = time.time()
        st.info(f"Otimização concluída em {(end_time - start_time) / 60:.2f} minutos.")

# --- EXIBIÇÃO DOS RESULTADOS ---
    # Pega os melhores valores encontrados
    T_final, M_final, N_final, delta_final = resultado.x
    custo_minimo = resultado.fun
    
    # Arredonda M e N para os valores inteiros finais
    N_final_int = int(round(N_final))
    M_final_int = int(round(M_final))
    
    # Recalcula as métricas finais com a melhor solução encontrada
    metricas_otimas = calcular_metricas_completas(T_final, N_final_int, M_final_int, delta_final, params)

    if metricas_otimas:
        # Armazena os resultados no session_state para uso posterior
        st.session_state['politica_otimizada'] = (T_final, N_final_int, M_final_int, delta_final)

        # Exibe as variáveis de decisão ótimas
        st.markdown("##### 🔍 Política Ótima Encontrada")
        r_col1, r_col2, r_col3, r_col4 = st.columns(4)
        r_col1.metric("🕒 T ótimo", f"{T_final:.2f}")
        r_col2.metric("🔢 M ótimo", f"{M_final_int}")
        r_col3.metric("🔢 N ótimo", f"{N_final_int}")
        r_col4.metric("⏱️ δ ótimo", f"{delta_final:.2f}")

        # Exibe as métricas de desempenho ótimas
        st.markdown("##### 🎯 Desempenho da Política Ótima")
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("💰 Custo Mínimo", f"{custo_minimo:.4f}")
        m_col2.metric("📈 Disponibilidade", f"{metricas_otimas['Disponibilidade']:.2%}")
        m_col3.metric("🛠️ MTBOF", f"{metricas_otimas['MTBOF']:.2f}")
    else:
        #st.error("A otimização encontrou uma combinação de parâmetros instável. Tente novamente.")
        pass

# =============================================================================
# SEÇÃO DE AVALIAÇÃO MANUAL
# =============================================================================
st.header("🧪 Avaliação de Política Pré-Definida")
col_man1, col_man2, col_man3, col_man4 = st.columns(4)
T_manual = col_man1.number_input("Valor de T", step=10.0, key="T_man")
M_manual = col_man2.number_input("Valor de M", step=1, min_value=0, key="M_man")
N_manual = col_man3.number_input("Valor de N", step=1, min_value=1, key="N_man")
# O min_value agora é dinâmico, baseado no que foi inserido na interface.
delta_manual = col_man4.number_input("Valor de δ", step=10.0, min_value=params['delta_min'], key="delta_man")

if st.button("📊 Avaliar Política"):
    # Apenas M > N deve ser um erro, M = N é permitido pela constraint M <= N
    if M_manual > N_manual:
        st.error("Erro: M não pode ser maior que N.")
    else:
        with st.spinner("Calculando desempenho..."):
            metricas_manuais = calcular_metricas_completas(T_manual, N_manual, M_manual, delta_manual, params)
        if metricas_manuais:
            st.markdown("##### 🎯 Desempenho da Política Informada")
            res_col1, res_col2, res_col3 = st.columns(3)
            res_col1.metric("💰 Taxa de Custo", f"R$ {metricas_manuais['Custo']:.4f}")
            res_col2.metric("📈 Disponibilidade", f"{metricas_manuais['Disponibilidade']:.2%}")
            res_col3.metric("🛠️ MTBOF", f"{metricas_manuais['MTBOF']:.2f}")
            st.session_state['politica_manual'] = (T_manual, N_manual, M_manual, delta_manual)
        else:
            st.error("Política inválida ou erro no cálculo.")
st.markdown("---")

# =============================================================================
# SEÇÃO DE ANÁLISE DE SENSIBILIDADE
# =============================================================================

def analise_sensibilidade_mnt(T, N, M, delta, parametros_base, n_simulacoes, variacoes_parametros):

    resultados = []
    
    # Itera para o número de simulações desejado
    for _ in range(n_simulacoes):
        parametros_simulados = parametros_base.copy()
        
        # Perturba os parâmetros selecionados
        for param, variacao in variacoes_parametros.items():
            perturbacao = np.random.uniform(1 - variacao, 1 + variacao)
            parametros_simulados[param] *= perturbacao
        
        # Calcula as métricas para a política com os parâmetros perturbados
        metricas = calcular_metricas_completas(T, N, M, delta, parametros_simulados)
        
        if metricas:
            resultados.append(metricas)

    # Retorna um DataFrame com os resultados
    return pd.DataFrame(resultados)


st.header("📉 Análise de Sensibilidade")

if 'politica_manual' in st.session_state:
    T_usado, N_usado, M_usado, delta_usado = st.session_state['politica_manual']
    st.info(f"Análise será executada para a política: **T={T_usado:.1f}, N={N_usado}, M={M_usado}, δ={delta_usado:.1f}**")
    
    st.markdown("##### Selecione os Parâmetros com Incerteza (%)")
    
    n_simulacoes = st.number_input("Tamanho da amostra de simulações", 100, 500, 200, 100)
    
    variacoes_parametros = {}
    
    # Layout de seleção de parâmetros (duas colunas por item)
    for param_key in params.keys():
        col_check, col_slider = st.columns([1, 2])
        with col_check:
            incluir = st.checkbox(f"Analisar {param_key}", key=f"check_{param_key}")
        with col_slider:
            if incluir:
                variacao = st.slider(
                    f"Nível de Incerteza para {param_key}", 
                    1, 50, 10, 
                    key=f"slider_{param_key}",
                    label_visibility="collapsed"
                )
                variacoes_parametros[param_key] = variacao / 100

    if st.button("🚀 Iniciar Análise de Sensibilidade"):
        if not variacoes_parametros:
            st.warning("Selecione pelo menos um parâmetro para a análise.")
        else:
            with st.spinner("⏳ Executando a análise de sensibilidade..."):
                parametros_base = params.copy()

                # Chama a função de lógica dedicada
                df_resultados = analise_sensibilidade_mnt(
                    T_usado, N_usado, M_usado, delta_usado,
                    parametros_base,
                    n_simulacoes=n_simulacoes,
                    variacoes_parametros=variacoes_parametros
                )

            st.subheader("Box-plots dos Resultados")

            # Cria a figura com 3 subplots para as 3 métricas
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            # --- Gráfico 1: Custo ---
            axes[0].boxplot(df_resultados['Custo'], vert=False, patch_artist=True, boxprops=dict(facecolor='skyblue'))
            media_custo = df_resultados['Custo'].mean()
            std_custo = df_resultados['Custo'].std()
            axes[0].set_title('Box-plot para Taxa de Custo', loc='left', fontsize=12, color='black')
            axes[0].text(0.02, 0.95, f"Média = {media_custo:.4f}\nDesvio Padrão = {std_custo:.4f}",
                         transform=axes[0].transAxes, fontsize=10, color='black',
                         verticalalignment='top', horizontalalignment='left')

            # --- Gráfico 2: Disponibilidade  ---
            #axes[1].boxplot(df_resultados['Disponibilidade'], vert=False, patch_artist=True, boxprops=dict(facecolor='lightcoral'))
            #media_disp = df_resultados['Disponibilidade'].mean()
            #std_disp = df_resultados['Disponibilidade'].std()
            #axes[1].set_title('Box-plot para Disponibilidade', loc='left', fontsize=12, color='black')
            #axes[1].text(0.02, 0.95, f"Média = {media_disp:.2%}\nDesvio Padrão = {std_disp:.2%}",
                         #transform=axes[1].transAxes, fontsize=10, color='black',
                         #verticalalignment='top', horizontalalignment='left')
            
            # --- Gráfico 3: MTBOF ---
            axes[2].boxplot(df_resultados['MTBOF'], vert=False, patch_artist=True, boxprops=dict(facecolor='lightgreen'))
            media_mtbof = df_resultados['MTBOF'].mean()
            std_mtbof = df_resultados['MTBOF'].std()
            axes[2].set_title('Box-plot para MTBOF', loc='left', fontsize=12, color='black')
            axes[2].text(0.02, 0.95, f"Média = {media_mtbof:.2f}\nDesvio Padrão = {std_mtbof:.2f}",
                         transform=axes[2].transAxes, fontsize=10, color='black',
                         verticalalignment='top', horizontalalignment='left')

            # Salva a figura em um buffer de memória para exibir com st.image
            fig.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            plt.close(fig)  # Fecha a figura para liberar memória
            buf.seek(0)
            st.image(buf)
    
else:
    st.warning("Primeiro, avalie uma política na seção anterior para habilitar a Análise de Sensibilidade.")

# =============================================================================
# Rodapé
# =============================================================================
st.markdown(""" 
<hr style="border:0.5px solid #333;" />

<div style='color: #aaa; font-size: 13px; text-align: left;'>
    <strong style="color: #ccc;">RANDOM - Grupo de Pesquisa em Risco e Análise de Decisão em Operações e Manutenção</strong><br>
    Criado em 2012, o grupo reúne pesquisadores dedicados às áreas de risco, manutenção e modelagem de operações.<br>
    <a href='http://random.org.br' target='_blank' style='color:#888;'>Acesse o site do RANDOM</a>
</div>
""", unsafe_allow_html=True)





