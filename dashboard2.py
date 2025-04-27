# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px
# import time

# # 페이지 설정
# st.set_page_config(page_title="실시간 설비 모니터링", layout="wide")

# st.title("실시간 설비 모니터링 대시보드")

# # 샘플 데이터 생성
# np.random.seed(42)

# def generate_data(n=1):
#     timestamp = pd.date_range(end=pd.Timestamp.now(), periods=n, freq='s')
#     data = pd.DataFrame({
#         'timestamp': timestamp,
#         'temperature': 25 + np.random.normal(0, 0.5, n),
#         'frequency': 60 + np.random.normal(0, 0.1, n),
#         'cumulative_power': np.cumsum(np.random.normal(0.5, 0.05, n)),
#         'actual': np.sin(np.linspace(0, 20, n)) + np.random.normal(0, 0.1, n),
#         'predicted': np.sin(np.linspace(0, 20, n)),
#     })
#     data['residual'] = data['actual'] - data['predicted']
#     data['mse'] = data['residual'] ** 2
#     return data

# # ----- KPI Section -----
# kpi1, kpi2, kpi3 = st.columns(3)

# with kpi1:
#     temperature_placeholder = st.empty()
# with kpi2:
#     frequency_placeholder = st.empty()
# with kpi3:
#     cumulative_power_placeholder = st.empty()

# # ----- 본문 레이아웃: 2행 2열 그리드 -----
# row1_col1, row1_col2 = st.columns(2)
# row2_col1, row2_col2 = st.columns(2)

# # 각 그래프 영역을 st.empty()로 비워둔 후, 동적으로 갱신
# fig1_placeholder = row1_col1.empty()
# fig2_placeholder = row1_col2.empty()
# fig3_placeholder = row2_col1.empty()
# fig4_placeholder = row2_col2.empty()

# # 실시간 업데이트: 데이터 초기화
# data = pd.DataFrame(columns=['timestamp', 'temperature', 'frequency', 'cumulative_power', 'actual', 'predicted', 'residual', 'mse'])

# for _ in range(300):  # 300번 반복 (실시간 갱신)
#     new_data = generate_data(1)
#     data = pd.concat([data, new_data], ignore_index=True)

#     # KPI 값 갱신
#     temperature_placeholder.metric("온도 (°C)", f"{data['temperature'].iloc[-1]:.2f}")
#     frequency_placeholder.metric("주파수 (Hz)", f"{data['frequency'].iloc[-1]:.2f}")
#     cumulative_power_placeholder.metric("누적 전력량 (kWh)", f"{data['cumulative_power'].iloc[-1]:.2f}")
    
#     # 1. 온도/주파수/누적전력량 모니터링 그래프
#     with fig1_placeholder:
#         # st.subheader("1. 온도/주파수/누적전력량 모니터링")
#         fig1 = px.line(data, x='timestamp', y=['temperature', 'frequency', 'cumulative_power'], labels={'value': '측정값', 'timestamp': '시간'})
#         fig1.update_layout(
#             title="온도, 주파수, 누적전력량 변화",
#             margin=dict(l=0, r=0, t=40, b=0),
#             height=300
#         )
#         st.plotly_chart(fig1, use_container_width=True, key=f"real_monitoring_{data['timestamp'].iloc[-1]}")

#     # 2. 실시간 이상탐지 모니터링
#     with fig2_placeholder:
#         # st.subheader("2. 실시간 이상탐지 모니터링")
#         anomalies = data[np.abs(data['residual']) > 0.3]
#         fig2 = px.scatter(data, x='timestamp', y='actual', labels={'actual': '실제값'})
#         fig2.add_scatter(x=anomalies['timestamp'], y=anomalies['actual'], mode='markers', marker=dict(color='red', size=8), name='Anomaly')
#         fig2.update_layout(
#         title="실시간 이상탐지 모니터링",
#             margin=dict(l=0, r=0, t=40, b=0),
#             height=300
#         )
#         st.plotly_chart(fig2, use_container_width=True, key=f"anomaly_detection_{data['timestamp'].iloc[-1]}")

#     # 3. 실제값 vs 예측값 차이 (Residual)
#     with fig3_placeholder:
#         # st.subheader("3. 실제값 vs 예측값 차이 (Residual)")
#         fig3 = px.line(data, x='timestamp', y='residual', labels={'residual': '차이(Residual)'})
#         fig3.update_layout(
#         title="실제값과 예측값 차이",
#             margin=dict(l=0, r=0, t=40, b=0),
#             height=300
#         )
#         st.plotly_chart(fig3, use_container_width=True, key=f"residual_{data['timestamp'].iloc[-1]}")

#     # 4. MSE 높은 순위 바그래프
#     with fig4_placeholder:
#         # st.subheader("4. MSE 높은 순위 바그래프")
#         top_mse = data[['timestamp', 'mse']].sort_values(by='mse', ascending=False).head(10)
#         fig4 = px.bar(top_mse, x='timestamp', y='mse', labels={'mse': 'Mean Squared Error'})
#         fig4.update_layout(
#             title="MSE 높은 순위",
#             margin=dict(l=0, r=0, t=40, b=0),
#             height=300
#         )
#         st.plotly_chart(fig4, use_container_width=True, key=f"anomaly_rank_{data['timestamp'].iloc[-1]}")

#     # 주기적으로 3초 간격으로 갱신
#     time.sleep(3)  # 3초마다 갱신
