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
                Política de Inspeção e Manutenção Preventiva com Aproveitamento de Oportunidades (MNTθδ)
            </h1>
        </div>
    """, unsafe_allow_html=True)

# =============================================================================
# FUNÇÕES BASE DO MODELO MATEMÁTICO
# =============================================================================

# --- Funções de Distribuição ---
def fx(t, beta, eta):
    if t < 0: return 0
    return ((beta / eta) * ((t / eta) ** (beta - 1))) * np.exp(-((t / eta) ** beta))

def Rx(t, beta, eta):
    if t < 0: return 1
    return np.exp(-((t / eta) ** beta))

def fh(t, beta, eta):
    if t < 0: return 0
    return ((beta / eta) * ((t / eta) ** (beta - 1))) * np.exp(-((t / eta) ** beta))

def Rh(t, beta, eta):
    if t < 0: return 1
    return np.exp(-((t / eta) ** beta))

def fw(t, lam):
    if t < 0: return 0
    return lam * np.exp(-lam * t)

def Rw(t, lam):
    if t < 0: return 1
    return np.exp(-lam * t)

# CORREÇÃO: Função Cep atualizada para a nova lógica linear por partes.
def Cep(d, params):
    """
    Calcula o custo de reposição antecipada (Cep) com base no atraso 'd'.
    A lógica segue a sugestão do orientador.
    """
    delta_min = params['delta_min']
    delta_limite = params['delta_limite']
    Cep_max = params['Cep_max']
    Cp = params['Cp']

    # Garante que delta_limite seja maior que delta_min para evitar divisão por zero.
    if delta_limite <= delta_min:
        return Cp # Retorna o custo base se os limites forem inválidos.

    if d < delta_min:
        # Se o delta for menor que o mínimo prático, aplica o custo máximo.
        return Cep_max
    elif delta_min <= d <= delta_limite:
        # Lógica da reta decrescente entre (delta_min, Cep_max) e (delta_limite, Cp)
        slope = (Cp - Cep_max) / (delta_limite - delta_min)
        cost = Cep_max + slope * (d - delta_min)
        return cost
    else:  # d > delta_limite
        # Custo se torna constante e igual ao custo preventivo programado.
        return Cp

#######################################
#       IMPLEMENTAÇÃO DOS CENÁRIOS    #
#######################################

def calcular_cenario1(T, N, M, Y, delta, Ci, Cp, Cop, Cf, Dp, Df, betax, etax, betah, etah, lambd):
    """Cenário 1: Preventiva em NT, sistema em bom estado."""
    p1 = Rx(N * T, betax, etax) * Rw((N - M) * T, lambd)
    c1 = Y * Ci + Cp
    ec1 = c1 * p1
    l1 = N * T + Dp
    el1 = l1 * p1
    ed1 = Dp * p1
    return p1, ec1, el1, ed1

def calcular_cenario2(T, N, M, Y, delta, Ci, Cp, Cop, Cf, Dp, Df, betax, etax, betah, etah, lambd):
    """Cenário 2: Preventiva por oportunidade (sistema saudável)."""
    if N == M:
        return 0, 0, 0, 0

    integrand_p = lambda w: fw(w, lambd) * Rx(M * T + w, betax, etax)
    p2, _ = quad(integrand_p, 0, (N - M) * T)
    el2, _ = quad(lambda w: (M * T + w + Dp) * integrand_p(w), 0, (N - M) * T)

    ec2 = 0
    if M < Y:
        for i in range(1, Y - M + 1):
            cost_interval = (M + i - 1) * Ci + Cop 
            integral_interval, _ = quad(integrand_p, (i-1)*T, i*T)
            ec2 += cost_interval * integral_interval
        cost_final_interval = Y * Ci + Cop 
        integral_final, _ = quad(integrand_p, (Y-M)*T, (N-M)*T)
        ec2 += cost_final_interval * integral_final
    else:
        cost_total = Y * Ci + Cop 
        ec2 = cost_total * p2
    ed2 = Dp * p2
    return p2, ec2, el2, ed2

def calcular_cenario3(T, N, M, Y, delta, Ci, Cp, Cop, Cf, Dp, Df, betax, etax, betah, etah, lambd):
    """Cenário 3: Preventiva antecipada após inspeção positiva."""
    p3, ec3, el3 = 0, 0, 0
    if Y == 0:
        return 0, 0, 0, 0

    cost_cep = Cep(delta, params)
    
    if M >= 1 and N > M and M < Y:
        for i in range(1, M + 1):
            integral, _ = quad(lambda x: fx(x, betax, etax) * Rh(i*T + delta - x, betah, etah) * Rw(delta, lambd), (i-1)*T, i*T)
            p3 += integral; ec3 += (i*Ci + cost_cep)*integral; el3 += (i*T + delta + Dp)*integral
        for i in range(M + 1, Y + 1):
            integral, _ = quad(lambda x: fx(x, betax, etax) * Rh(i*T + delta - x, betah, etah) * Rw(i*T + delta - M*T, lambd), (i-1)*T, i*T)
            p3 += integral; ec3 += (i*Ci + cost_cep)*integral; el3 += (i*T + delta + Dp)*integral
    elif M >= 1 and M >= Y and Y >= 1:
        for i in range(1, Y + 1):
            integral, _ = quad(lambda x: fx(x, betax, etax) * Rh(i*T + delta - x, betah, etah) * Rw(delta, lambd), (i-1)*T, i*T)
            p3 += integral; ec3 += (i*Ci + cost_cep)*integral; el3 += (i*T + delta + Dp)*integral
    elif M == 0 and Y >= 1:
        for i in range(1, Y + 1):
            integral, _ = quad(lambda x: fx(x, betax, etax) * Rh(i*T + delta - x, betah, etah) * Rw(i*T + delta, lambd), (i-1)*T, i*T)
            p3 += integral; ec3 += (i*Ci + cost_cep)*integral; el3 += (i*T + delta + Dp)*integral
    ed3 = Dp * p3
    return p3, ec3, el3, ed3

def calcular_cenario4(T, N, M, Y, delta, Ci, Cp, Cop, Cf, Dp, Df, betax, etax, betah, etah, lambd):
    """Cenário 4: Preventiva por oportunidade (sistema defeituoso)."""
    p4, ec4, el4 = 0, 0, 0
    
    if M >= 1 and N > M and M < Y:
        for i in range(1, M + 1):
            integrand_p = lambda w, x: fx(x, betax, etax)*fw(w, lambd)*Rh(i*T+w-x, betah, etah)
            integral, _ = dblquad(integrand_p, (i-1)*T, i*T, 0, delta)
            p4 += integral; ec4 += (i*Ci + Cop)*integral
            el4 += dblquad(lambda w, x: (i*T+w+Dp)*integrand_p(w,x), (i-1)*T, i*T, 0, delta)[0]
        for i in range(M + 1, Y + 1):
            integrand_p2 = lambda w, x: fx(x, betax, etax)*fw(w, lambd)*Rh(M*T+w-x, betah, etah)
            integral1, _ = dblquad(integrand_p2, (i-1)*T, i*T, lambda x: x-M*T, lambda x: (i-M)*T)
            p4 += integral1; ec4 += ((i-1)*Ci + Cop)*integral1
            el4 += dblquad(lambda w, x: (M*T+w+Dp)*integrand_p2(w,x), (i-1)*T, i*T, lambda x: x-M*T, lambda x: (i-M)*T)[0]
            integral2, _ = dblquad(integrand_p2, (i-1)*T, i*T, lambda x: (i-M)*T, lambda x: (i-M)*T+delta)
            p4 += integral2; ec4 += (i*Ci + Cop)*integral2
            el4 += dblquad(lambda w, x: (M*T+w+Dp)*integrand_p2(w,x), (i-1)*T, i*T, lambda x: (i-M)*T, lambda x: (i-M)*T+delta)[0]
    elif M >= 1 and M >= Y and Y >= 1:
        for i in range(1, Y + 1):
            integrand_p = lambda w,x: fx(x, betax, etax)*fw(w, lambd)*Rh(i*T+w-x, betah, etah)
            integral, _ = dblquad(integrand_p, (i-1)*T, i*T, 0, delta)
            p4 += integral; ec4 += (i*Ci + Cop)*integral
            el4 += dblquad(lambda w,x: (i*T+w+Dp)*integrand_p(w,x), (i-1)*T, i*T, 0, delta)[0]
    elif M == 0 and Y >= 1:
        for i in range(1, Y + 1):
            integrand_p = lambda w,x: fx(x, betax, etax)*fw(w, lambd)*Rh(w-x, betah, etah)
            integral1, _ = dblquad(integrand_p, (i-1)*T, i*T, lambda x: x, lambda x: i*T)
            p4 += integral1; ec4 += ((i-1)*Ci + Cop)*integral1
            el4 += dblquad(lambda w,x: (w+Dp)*integrand_p(w,x), (i-1)*T, i*T, lambda x: x, lambda x: i*T)[0]
            integral2, _ = dblquad(integrand_p, (i-1)*T, i*T, i*T, lambda x: i*T+delta)
            p4 += integral2; ec4 += (i*Ci + Cop)*integral2
            el4 += dblquad(lambda w,x: (w+Dp)*integrand_p(w,x), (i-1)*T, i*T, i*T, lambda x: i*T+delta)[0]
    elif Y == 0:
        integrand_p = lambda w, x: fx(x, betax, etax)*fw(w, lambd)*Rh(M*T+w-x, betah, etah)
        integrand_l = lambda w, x: (M*T+w+Dp)*integrand_p(w,x)
        p4_1, _ = dblquad(integrand_p, 0, M*T, 0, (N-M)*T)
        p4_2, _ = dblquad(integrand_p, M*T, N*T, lambda x: x-M*T, lambda x: (N-M)*T)
        p4 = p4_1 + p4_2
        el4_1, _ = dblquad(integrand_l, 0, M*T, 0, (N-M)*T)
        el4_2, _ = dblquad(integrand_l, M*T, N*T, lambda x: x-M*T, lambda x: (N-M)*T)
        el4 = el4_1 + el4_2
        ec4 = Cop * p4
    if Y < N and Y > 0:
        w_lower_bound = (Y - M) * T if M < Y else 0
        w_upper_bound = (N - M) * T
        if w_lower_bound < w_upper_bound:
            integrand_p_final = lambda x, w: fx(x, betax, etax) * fw(w, lambd) * Rh(M*T + w - x, betah, etah)
            p4_extra, _ = dblquad(integrand_p_final, w_lower_bound, w_upper_bound, Y*T, lambda w: M*T + w)
            integrand_l_final = lambda x, w: (M*T + w + Dp) * integrand_p_final(x, w)
            el4_extra, _ = dblquad(integrand_l_final, w_lower_bound, w_upper_bound, Y*T, lambda w: M*T + w)
            ec4_extra = (Y*Ci + Cop) * p4_extra
            p4 += p4_extra; ec4 += ec4_extra; el4 += el4_extra
    ed4 = Dp * p4
    return p4, ec4, el4, ed4

def calcular_cenario5(T, N, M, Y, delta, Ci, Cp, Cop, Cf, Dp, Df, betax, etax, betah, etah, lambd):
    """Cenário 5: Preventiva em NT, sistema em estado defeituoso."""
    if Y * T >= N * T:
        return 0, 0, 0, 0
    p5, _ = quad(lambda x: fx(x, betax, etax) * Rh(N*T - x, betah, etah) * Rw((N-M)*T, lambd), Y*T, N*T)
    c5 = Y * Ci + Cp
    ec5 = c5 * p5
    el5 = (N * T + Dp) * p5
    ed5 = Dp * p5
    return p5, ec5, el5, ed5

def calcular_cenario6(T, N, M, Y, delta, Ci, Cp, Cop, Cf, Dp, Df, betax, etax, betah, etah, lambd):
    """Cenário 6: Falha."""
    p6, ec6, el6 = 0, 0, 0
    if M < Y < N and M >= 1:
        for i in range(1, M + 1):
            term1, _ = dblquad(lambda h,x: fx(x,betax,etax)*fh(h,betah,etah), (i-1)*T, i*T, 0, lambda x: i*T-x)
            p6+=term1; ec6+=((i-1)*Ci+Cf)*term1; el6+=dblquad(lambda h,x:(x+h+Df)*fx(x,betax,etax)*fh(h,betah,etah),(i-1)*T, i*T, 0, lambda x: i*T-x)[0]
            term2, _ = dblquad(lambda h,x: fx(x,betax,etax)*fh(h,betah,etah)*Rw(x+h-i*T, lambd), (i-1)*T, i*T, lambda x:i*T-x, lambda x:i*T+delta-x)
            p6+=term2; ec6+=(i*Ci+Cf)*term2; el6+=dblquad(lambda h,x:(x+h+Df)*fx(x,betax,etax)*fh(h,betah,etah)*Rw(x+h-i*T, lambd),(i-1)*T, i*T, lambda x:i*T-x, lambda x:i*T+delta-x)[0]
        for i in range(M + 1, Y + 1):
            term1, _ = dblquad(lambda h,x: fx(x,betax,etax)*fh(h,betah,etah)*Rw(x+h-M*T, lambd), (i-1)*T, i*T, 0, lambda x: i*T-x)
            p6+=term1; ec6+=((i-1)*Ci+Cf)*term1; el6+=dblquad(lambda h,x:(x+h+Df)*fx(x,betax,etax)*fh(h,betah,etah)*Rw(x+h-M*T, lambd),(i-1)*T, i*T, 0, lambda x: i*T-x)[0]
            term2, _ = dblquad(lambda h,x: fx(x,betax,etax)*fh(h,betah,etah)*Rw(x+h-M*T, lambd), (i-1)*T, i*T, lambda x: i*T-x, lambda x: i*T+delta-x)
            p6+=term2; ec6+=(i*Ci+Cf)*term2; el6+=dblquad(lambda h,x:(x+h+Df)*fx(x,betax,etax)*fh(h,betah,etah)*Rw(x+h-M*T, lambd),(i-1)*T, i*T, lambda x: i*T-x, lambda x: i*T+delta-x)[0]
        term_final, _ = dblquad(lambda h,x: fx(x,betax,etax)*fh(h,betah,etah)*Rw(x+h-M*T, lambd), Y*T, N*T, 0, lambda x: N*T-x)
        p6+=term_final; ec6+=(Y*Ci+Cf)*term_final; el6+=dblquad(lambda h,x:(x+h+Df)*fx(x,betax,etax)*fh(h,betah,etah)*Rw(x+h-M*T, lambd), Y*T, N*T, 0, lambda x: N*T-x)[0]
    elif M < Y < N and M == 0:
        for i in range(1, Y + 1):
            term1, _ = dblquad(lambda h,x: fx(x,betax,etax)*fh(h,betah,etah)*Rw(x+h, lambd), (i-1)*T, i*T, 0, lambda x: i*T-x)
            p6+=term1; ec6+=((i-1)*Ci+Cf)*term1; el6+=dblquad(lambda h,x:(x+h+Df)*fx(x,betax,etax)*fh(h,betah,etah)*Rw(x+h, lambd),(i-1)*T, i*T, 0, lambda x: i*T-x)[0]
            term2, _ = dblquad(lambda h,x: fx(x,betax,etax)*fh(h,betah,etah)*Rw(x+h, lambd), (i-1)*T, i*T, lambda x:i*T-x, lambda x: i*T+delta-x)
            p6+=term2; ec6+=(i*Ci+Cf)*term2; el6+=dblquad(lambda h,x:(x+h+Df)*fx(x,betax,etax)*fh(h,betah,etah)*Rw(x+h, lambd),(i-1)*T, i*T, lambda x:i*T-x, lambda x: i*T+delta-x)[0]
        term_final, _ = dblquad(lambda h,x: fx(x,betax,etax)*fh(h,betah,etah)*Rw(x+h, lambd), Y*T, N*T, 0, lambda x: N*T-x)
        p6+=term_final; ec6+=(Y*Ci+Cf)*term_final; el6+=dblquad(lambda h,x:(x+h+Df)*fx(x,betax,etax)*fh(h,betah,etah)*Rw(x+h, lambd), Y*T, N*T, 0, lambda x: N*T-x)[0]
    elif N > Y == M > 0:
        for i in range(1, M + 1):
            term1, _ = dblquad(lambda h,x: fx(x,betax,etax)*fh(h,betah,etah), (i-1)*T, i*T, 0, lambda x: i*T-x)
            p6+=term1; ec6+=((i-1)*Ci+Cf)*term1; el6+=dblquad(lambda h,x:(x+h+Df)*fx(x,betax,etax)*fh(h,betah,etah),(i-1)*T, i*T, 0, lambda x: i*T-x)[0]
            term2, _ = dblquad(lambda h,x: fx(x,betax,etax)*fh(h,betah,etah)*Rw(x+h-i*T, lambd), (i-1)*T, i*T, lambda x: i*T-x, lambda x: i*T+delta-x)
            p6+=term2; ec6+=(i*Ci+Cf)*term2; el6+=dblquad(lambda h,x:(x+h+Df)*fx(x,betax,etax)*fh(h,betah,etah)*Rw(x+h-i*T, lambd),(i-1)*T, i*T, lambda x: i*T-x, lambda x: i*T+delta-x)[0]
        term_final, _ = dblquad(lambda h,x: fx(x,betax,etax)*fh(h,betah,etah)*Rw(x+h-M*T, lambd), M*T, N*T, 0, lambda x: N*T-x)
        p6+=term_final; ec6+=(M*Ci+Cf)*term_final; el6+=dblquad(lambda h,x:(x+h+Df)*fx(x,betax,etax)*fh(h,betah,etah)*Rw(x+h-M*T, lambd),M*T, N*T, 0, lambda x: N*T-x)[0]
    elif N > M == Y == 0:
        p6, _ = dblquad(lambda h, x: fx(x,betax,etax)*fh(h,betah,etah)*Rw(x+h, lambd), 0, N*T, 0, lambda x: N*T-x)
        ec6 = Cf * p6
        el6, _ = dblquad(lambda h, x: (x+h+Df)*fx(x,betax,etax)*fh(h,betah,etah)*Rw(x+h, lambd), 0, N*T, 0, lambda x: N*T-x)
    elif N > M > Y and Y >= 1:
        for i in range(1, Y + 1):
            term1, _ = dblquad(lambda h,x: fx(x,betax,etax)*fh(h,betah,etah), (i-1)*T, i*T, 0, lambda x: i*T-x)
            p6+=term1; ec6+=((i-1)*Ci+Cf)*term1; el6+=dblquad(lambda h,x:(x+h+Df)*fx(x,betax,etax)*fh(h,betah,etah),(i-1)*T, i*T, 0, lambda x: i*T-x)[0]
            term2, _ = dblquad(lambda h,x: fx(x,betax,etax)*fh(h,betah,etah)*Rw(x+h-i*T, lambd), (i-1)*T, i*T, lambda x:i*T-x, lambda x:i*T+delta-x)
            p6+=term2; ec6+=(i*Ci+Cf)*term2; el6+=dblquad(lambda h,x:(x+h+Df)*fx(x,betax,etax)*fh(h,betah,etah)*Rw(x+h-i*T, lambd),(i-1)*T, i*T, lambda x:i*T-x, lambda x:i*T+delta-x)[0]
        term3, _ = dblquad(lambda h,x: fx(x,betax,etax)*fh(h,betah,etah), Y*T, M*T, 0, lambda x: M*T-x)
        p6+=term3; ec6+=(Y*Ci+Cf)*term3; el6+=dblquad(lambda h,x:(x+h+Df)*fx(x,betax,etax)*fh(h,betah,etah), Y*T, M*T, 0, lambda x: M*T-x)[0]
        term4, _ = dblquad(lambda h,x: fx(x,betax,etax)*fh(h,betah,etah)*Rw(x+h-M*T, lambd), Y*T, M*T, lambda x: M*T-x, lambda x: N*T-x)
        p6+=term4; ec6+=(Y*Ci+Cf)*term4; el6+=dblquad(lambda h,x:(x+h+Df)*fx(x,betax,etah)*fh(h,betah,etah)*Rw(x+h-M*T, lambd), Y*T, M*T, lambda x: M*T-x, lambda x: N*T-x)[0]
        term5, _ = dblquad(lambda h,x: fx(x,betax,etax)*fh(h,betah,etah)*Rw(x+h-M*T, lambd), M*T, N*T, 0, lambda x: N*T-x)
        p6+=term5; ec6+=(Y*Ci+Cf)*term5; el6+=dblquad(lambda h,x:(x+h+Df)*fx(x,betax,etax)*fh(h,betah,etah)*Rw(x+h-M*T, lambd), M*T, N*T, 0, lambda x: N*T-x)[0]
    elif N > M > Y == 0:
        term1, _ = dblquad(lambda h,x: fx(x,betax,etax)*fh(h,betah,etah), 0, M*T, 0, lambda x: M*T-x)
        term2, _ = dblquad(lambda h,x: fx(x,betax,etax)*fh(h,betah,etah)*Rw(x+h-M*T, lambd), 0, M*T, lambda x:M*T-x, lambda x: N*T-x)
        term3, _ = dblquad(lambda h,x: fx(x,betax,etax)*fh(h,betah,etah)*Rw(x+h-M*T, lambd), M*T, N*T, 0, lambda x: N*T-x)
        p6 = term1 + term2 + term3
        ec6 = Cf * p6
        el6_1,_=dblquad(lambda h,x:(x+h+Df)*fx(x,betax,etax)*fh(h,betah,etah),0,M*T,0,lambda x:M*T-x)
        el6_2,_=dblquad(lambda h,x:(x+h+Df)*fx(x,betax,etax)*fh(h,betah,etah)*Rw(x+h-M*T, lambd),0,M*T,lambda x:M*T-x,lambda x:N*T-x)
        el6_3,_=dblquad(lambda h,x:(x+h+Df)*fx(x,betax,etax)*fh(h,betah,etah)*Rw(x+h-M*T, lambd),M*T,N*T,0,lambda x:N*T-x)
        el6 = el6_1 + el6_2 + el6_3
    elif N == M > Y >= 1:
        for i in range(1, Y + 1):
            term1, _ = dblquad(lambda h,x: fx(x,betax,etax)*fh(h,betah,etah), (i-1)*T, i*T, 0, lambda x: i*T-x)
            p6+=term1; ec6+=((i-1)*Ci+Cf)*term1; el6+=dblquad(lambda h,x:(x+h+Df)*fx(x,betax,etax)*fh(h,betah,etah),(i-1)*T, i*T, 0, lambda x: i*T-x)[0]
            term2, _ = dblquad(lambda h,x: fx(x,betax,etax)*fh(h,betah,etah)*Rw(x+h-i*T, lambd), (i-1)*T, i*T, lambda x:i*T-x, lambda x:i*T+delta-x)
            p6+=term2; ec6+=(i*Ci+Cf)*term2; el6+=dblquad(lambda h,x:(x+h+Df)*fx(x,betax,etax)*fh(h,betah,etah)*Rw(x+h-i*T, lambd),(i-1)*T, i*T, lambda x:i*T-x, lambda x:i*T+delta-x)[0]
        term_final, _ = dblquad(lambda h,x: fx(x,betax,etax)*fh(h,betah,etah), Y*T, N*T, 0, lambda x: N*T-x)
        p6+=term_final; ec6+=(Y*Ci+Cf)*term_final; el6+=dblquad(lambda h,x:(x+h+Df)*fx(x,betax,etax)*fh(h,betah,etah), Y*T, N*T, 0, lambda x: N*T-x)[0]
    elif N == M > Y == 0:
        p6, _ = dblquad(lambda h, x: fx(x,betax,etax)*fh(h,betah,etah), 0, N*T, 0, lambda x: N*T-x)
        ec6 = Cf * p6
        el6, _ = dblquad(lambda h,x: (x+h+Df)*fx(x,betax,etax)*fh(h,betah,etah), 0, N*T, 0, lambda x: N*T-x)
    ed6 = Df * p6
    return p6, ec6, el6, ed6

def calcular_metricas_completas(T, N, M, delta, params):
    """
    Calcula Custo, Disponibilidade e MTBOF para a política.
    Retorna None se os parâmetros forem inválidos ou der erro.
    """
    try:
        # 1) Validações iniciais (CORRIGIDO: M > N em vez de M >= N)
        if M > N or N < 1 or T <= 0:
            return None
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
st.markdown("Define o custo em função do tempo de atraso `δ`")

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
        help="Limite de tempo. Para atrasos (δ) maiores que este, o custo se torna o mesmo que o de uma preventiva programada (Cp).",
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
        
        # Converte M e N para inteiros, pois o otimizador trabalha com floats
        M_val_int = int(round(M_val))
        N_val_int = int(round(N_val))
        
        # Chama a função de cálculo principal.
        # Ela já retorna 'None' se a política for inválida (ex: M >= N).
        resultados = calcular_metricas_completas(T_val, N_val_int, M_val_int, delta_val, params)

        # Se a política for inválida, retorna um custo altíssimo (penalidade).
        if resultados is None:
            return 1e9
        
        # Retorna a Taxa de Custo para ser minimizada.
        return resultados["Custo"]

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
        st.error("A otimização encontrou uma combinação de parâmetros instável. Tente novamente.")

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
            axes[1].boxplot(df_resultados['Disponibilidade'], vert=False, patch_artist=True, boxprops=dict(facecolor='lightcoral'))
            media_disp = df_resultados['Disponibilidade'].mean()
            std_disp = df_resultados['Disponibilidade'].std()
            axes[1].set_title('Box-plot para Disponibilidade', loc='left', fontsize=12, color='black')
            axes[1].text(0.02, 0.95, f"Média = {media_disp:.2%}\nDesvio Padrão = {std_disp:.2%}",
                         transform=axes[1].transAxes, fontsize=10, color='black',
                         verticalalignment='top', horizontalalignment='left')
            
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

