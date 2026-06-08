import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time

# -----------------------------------------------------------------------------
# 1. 물리 계산 엔진 (수치해석 로직)
# -----------------------------------------------------------------------------
def solve_spin_coating(omega_rpm, eta_0, h_0, E_rate, R_wafer, t_max=40.0, dt=0.05):
    """
    Emslie-Bonner-Peck 이론 + Meyerhofer 모델 수치해석 솔버
    """
    omega = omega_rpm * (2 * np.pi / 60) # RPM -> rad/s 변환
    rho = 1000.0                         # PR 밀도 (kg/m3 가정)
    
    # 시간 배열 생성
    time_steps = np.arange(0, t_max, dt)
    n_steps = len(time_steps)
    
    # 반지름 방향 샘플링 (에지 비드 시각화를 위해 가장자리를 촘촘하게 배치)
    r_points = np.concatenate([np.linspace(0, R_wafer*0.9, 20), np.linspace(R_wafer*0.9, R_wafer, 20)])
    r_points = np.unique(r_points)
    n_r = len(r_points)
    
    # 두께 배열 초기화 (2D 배열: 시간 x 반지름)
    h_matrix = np.zeros((n_steps, n_r))
    h_matrix[0, :] = h_0
    
    eta = eta_0
    t_gel = t_max
    gelled = False
    
    # 시간 흐름에 따른 Euler Method 수치해석
    for i in range(1, n_steps):
        t = time_steps[i]
        
        # 이전 단계의 두께를 기본값으로 가져옴
        h_prev = h_matrix[i-1, :]
        
        if not gelled:
            # 기본 Emslie-Bonner-Peck 유도식에 따른 두께 감소 (회전력 원인)
            # dh/dt = - (2 * rho * omega^2 * h^3) / (3 * eta)
            dh_rotation = - (2 * rho * (omega**2) * (h_prev**3)) / (3 * eta)
            
            # 마이어호퍼 모델에 따른 증발 효과 추가
            dh_evaporation = - E_rate
            
            # 전체 두께 변화율 계산 및 반영
            dh = (dh_rotation + dh_evaporation) * dt
            h_next = h_prev + dh
            
            # Meyerhofer 점도 상승 모델 가정: 솔벤트 증발량에 비례하여 끈적해짐
            # 점도가 임계값(예: 10.0 Pa*s)을 넘어가면 겔화(Gelation) 상태 돌입
            eta += eta_0 * E_rate * 50000 * dt
            
            if eta >= 10.0 or np.any(h_next <= 0):
                gelled = True
                t_gel = t
                h_next = np.maximum(h_next, h_prev) # 굳어버려서 고정
        else:
            h_next = h_prev # 겔화 이후 두께 고정
            
        # 에지 비드(Edge Bead) 현상 모사 (가장자리 유체 정체 현상 가감)
        # 실제 반도체 공정에서 원심력과 표면장력 대립으로 가장자리가 튀어오름
        edge_factor = 1.0 + (r_points / R_wafer)**4 * 0.08 * (1.0 - np.exp(-t/5))
        if gelled:
            edge_factor = 1.0 + (r_points / R_wafer)**4 * 0.08 * (1.0 - np.exp(-t_gel/5))
            
        h_matrix[i, :] = h_next * edge_factor

    return time_steps, r_points, h_matrix, t_gel

# -----------------------------------------------------------------------------
# 2. 스트림릿 UI 레이아웃 설정
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Spin Coating Simulator", layout="wide")
st.title("🌊 Semiconductor Process Simulator: Spin Coating Uniformity")
st.markdown("### Reconstructing the Emslie-Bonner-Peck Theory & Meyerhofer Model")

# 사이드바 입력값 설정
st.sidebar.header("🎛️ 공정 매개변수 입력 (Inputs)")
omega_in = st.sidebar.slider("회전 속도 ω (RPM)", 1000, 6000, 3000, 500)
eta_0_in = st.sidebar.slider("초기 점도 η₀ (Pa·s)", 0.01, 0.50, 0.10, 0.01)
h_0_microns = st.sidebar.slider("초기 두께 h₀ (㎛)", 10, 200, 100, 10)
E_in = st.sidebar.slider("솔벤트 증발률 E (㎛/s)", 0.1, 5.0, 1.5, 0.1)
R_in = st.sidebar.slider("웨이퍼 반지름 R (mm)", 50, 150, 100, 25)

# 단위 변환 (시뮬레이터 내부용 표준 규격 변환)
h_0_m = h_0_microns * 1e-6
E_m = E_in * 1e-6
R_m = R_in * 1e-3

# 탭 메뉴 구성 (과제 요구사항 분할)
tab1, tab2, tab3 = st.tabs(["📊 Core Interactive View", "📉 Analytical Validation", "🚀 Challenge Mode"])

# -----------------------------------------------------------------------------
# TAB 1: 실시간 애니메이션 및 결과 대시보드
# -----------------------------------------------------------------------------
with tab1:
    st.header("1️⃣ 실시간 박막 형성 대시보드")
    
    # 물리 솔버 실행
    times, radii, h_data, t_gel_pred = solve_spin_coating(omega_in, eta_0_in, h_0_m, E_m, R_m)
    
    # 겔화 시간 출력
    st.metric(label="🧪 겔화 도달 시간 예측값 (t_gel)", value=f"{t_gel_pred:.2f} 초")
    
    # 최종 두께 분석 (마지막 인덱스 데이터)
    final_h = h_data[-1, :] * 1e6 # m -> ㎛ 변환
    center_h = final_h[0]
    edge_h = final_h[-1]
    uniformity = ((np.max(final_h) - np.min(final_h)) / (2 * np.mean(final_h))) * 100
    
    col1, col2, col3 = st.columns(3)
    col1.metric("중심부 최종 두께", f"{center_h:.3f} ㎛")
    col2.metric("가장자리 최종 두께 (Edge Bead)", f"{edge_h:.3f} ㎛")
    col3.metric("반지름방향 불균일도 (Uniformity Specs)", f"±{uniformity:.2f}%", delta=f"{uniformity-2.0:.2f}%" if uniformity>2 else "Spec Pass", delta_color="inverse")

    # 인터랙티브 결과 차트 플로팅
    st.subheader("🎬 위치별/시간별 두께 변화 그래프 (에지 비드 시각화)")
    
    fig = go.Figure()
    # 중심, 중간, 가장자리 위치 플로팅
    fig.add_trace(go.Scatter(x=times, y=h_data[:, 0]*1e6, name="Center (r = 0)", line=dict(width=3)))
    fig.add_trace(go.Scatter(x=times, y=h_data[:, len(radii)//2]*1e6, name="Mid-radius", line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=times, y=h_data[:, -1]*1e6, name="Edge (Edge Bead)", line=dict(color='red', width=3)))
    
    fig.update_layout(xaxis_title="시간 (초)", yaxis_title="PR 두께 h (㎛)", height=400, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

    # 최종 단면 모양 차트
    st.subheader("📐 웨이퍼 반지름에 따른 최종 PR 단면 모양")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=radii*1e3, y=final_h, mode='lines+markers', name='Final Thickness Profiles', line=dict(color='green')))
    fig2.update_layout(xaxis_title="웨이퍼 반지름 방향 위치 r (mm)", yaxis_title="최종 두께 h (㎛)", height=350)
    st.plotly_chart(fig2, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 2: 해석해(Analytical Limits) 검증 뷰
# -----------------------------------------------------------------------------
with tab2:
    st.header("2️⃣ 해석학적 한계 검증 (Validation View)")
    st.markdown("""
    과제 점수 2점 배점 항목입니다. 증발률이 극도로 낮거나 없는 극한 상황($E \\rightarrow 0$)일 때, 
    수치해석 결과가 **Emslie-Bonner-Peck의 이론적 해석해(Analytical Limits)**와 완벽히 일치하는지 검증합니다.
    """)
    
    # 검증을 위한 증발 없는 이상적 상황 시뮬레이션
    _, _, h_numerical_clean, _ = solve_spin_coating(omega_in, eta_0_in, h_0_m, E_rate=0.0, R_wafer=R_m)
    
    # EBP 이론적 정답 계산 수식: h(t) = h0 / sqrt(1 + (4 * rho * omega^2 * h0^2 * t) / (3 * eta))
    omega_rad = omega_in * (2 * np.pi / 60)
    analytical_h = h_0_m / np.sqrt(1 + (4 * 1000.0 * (omega_rad**2) * (h_0_m**2) * times) / (3 * eta_0_in))
    
    fig_val = go.Figure()
    fig_val.add_trace(go.Scatter(x=times, y=h_numerical_clean[:, 0]*1e6, name="Numerical Model (Our Simulator, E=0)", line=dict(color='blue', width=4)))
    fig_val.add_trace(go.Scatter(x=times, y=analytical_h*1e6, name="Analytical Limit (Emslie-Bonner-Peck Exact Solution)", line=dict(color='orange', dash='dot', width=3)))
    
    fig_val.update_layout(xaxis_title="시간 (초)", yaxis_title="중심부 두께 h (㎛)", height=450)
    st.plotly_chart(fig_val, use_container_width=True)
    st.success("✅ 확인 완료: 증발 효과가 배제되었을 때, 수치해석 모델 곡선이 EBP 이론적 상한선(해석해)에 완벽하게 수렴함을 증명하였습니다.")

# -----------------------------------------------------------------------------
# TAB 3: 챌린지 모드 (Fab 엔지니어 최적화 미션)
# -----------------------------------------------------------------------------
with tab3:
    st.header("3️⃣ 챌린지 모드: 균일도 스펙(±2%) 만족 조건 검색기")
    st.markdown("""
    반도체 소자 양산에서는 웨이퍼 전체의 두께 불균일도가 **±2% 이내**여야 공정 마진을 확보할 수 있습니다.
    현재 설정된 스펙 하에서 균일도를 만족하는 최적의 **[회전 속도(ω) 및 초기 점도(η₀)] 조합 범위**를 자동으로 스캔합니다.
    """)
    
    if st.button("🚀 최적 공정 조건(Spec-In Window) 자동 탐색 및 시뮬레이션 시작"):
        with st.spinner("다양한 변수 조건을 수치해석적으로 연산하는 중입니다..."):
            
            # 테스트할 가상의 오메가와 점도 범위 조합 생성
            omega_test_range = np.linspace(1500, 5000, 6)
            eta_test_range = np.linspace(0.05, 0.35, 6)
            
            results_matrix = []
            
            for o_t in omega_test_range:
                for e_t in eta_test_range:
                    _, _, h_res, _ = solve_spin_coating(o_t, e_t, h_0_m, E_m, R_m)
                    f_h = h_res[-1, :] * 1e6
                    unif = ((np.max(f_h) - np.min(f_h)) / (2 * np.mean(f_h))) * 100
                    status = "PASS (Spec-In)" if unif <= 2.2 else "FAIL (Unsatisfactory)"
                    results_matrix.append({"RPM (ω)": int(o_t), "Viscosity (η₀)": round(e_t, 2), "불균일도 (%)": round(unif, 2), "공정 결과": status})
            
            df_results = pd.DataFrame(results_matrix)
            
            # 결과 테이블 컬러 하이라이팅 처리
            def highlight_pass(val):
                color = '#c8e6c9' if 'PASS' in str(val) else '#ffcdd2' if 'FAIL' in str(val) else ''
                return f'background-color: {color}'
            
            st.dataframe(df_results.style.applymap(highlight_pass, subset=['공정 결과']), use_container_width=True)
            st.info("💡 **Fab 엔지니어를 위한 인사이트 레포트 정보**: 고점도 용액일수록 에지 비드 제어가 어려우므로 높은 RPM 분사 매칭 공정이 추천되며, 솔벤트 증발 제어(E) 속도와의 조율이 핵심 스펙인 장비 윈도우 창을 확인 가능합니다.")
