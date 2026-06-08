import streamlit as st
import numpy as np
import pandas as pd

# 1. 웹 화면 꾸미기 (슬라이더 입력창)
st.title("🌊 스핀 코팅 유체역학 시뮬레이터")
st.sidebar.header("공정 변수 설정")

omega = st.sidebar.slider("회전 속도 (RPM)", 1000, 6000, 3000, 500)
eta_0 = st.sidebar.slider("초기 점도 (eta_0)", 0.01, 0.5, 0.1, 0.01)
E = st.sidebar.slider("솔벤트 증발률 (E)", 1e-6, 1e-4, 1e-5, 1e-6)

# 2. 유체역학 수치해석 계산 엔진
def simulate_spin_coating(omega, eta_0, E):
    dt = 0.1       # 시간 간격 (초)
    t_max = 30.0   # 총 시뮬레이션 시간
    time_steps = np.arange(0, t_max, dt)
    
    h = 100.0      # 초기 두께 (마이크로미터 가정)
    eta = eta_0    # 초기 점도
    
    h_history = []
    
    for t in time_steps:
        # 물리 공식 1: 회전력에 의한 두께 감소율
        dh_rotation = - (2 * (omega**2) * (h**3)) / (3 * eta)
        
        # 물리 공식 2: 솔벤트 증발에 의한 두께 감소율 (Meyerhofer)
        dh_evaporation = - E
        
        # 전체 두께 변화량
        dh = (dh_rotation + dh_evaporation) * dt
        h += dh
        
        # 시간이 갈수록 솔벤트가 날아가 점도(eta)가 급격히 상승함
        eta += eta_0 * E * 1000 * dt  
        
        # 유체가 완전히 굳어버리면(t_gel) 두께 감소가 멈춤
        if h <= 0 or eta > 10.0: 
            h = max(h, 1.0) # 최소 두께 고정
            
        h_history.append(h)
        
    return time_steps, h_history

# 3. 계산 실행 및 그래프 그리기
times, thicknesses = simulate_spin_coating(omega, eta_0, E)
chart_data = pd.DataFrame({"시간(초)": times, "PR 두께(㎛)": thicknesses})

st.subheader("⏱️ 시간에 따른 PR 박막 두께 변화")
st.line_chart(chart_data.set_index("시간(초)"))
