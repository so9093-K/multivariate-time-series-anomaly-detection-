import os
import time
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import plotly.express as px
from DetectModel import Detector
from DataLoader import Loader
from streamlit_autorefresh import st_autorefresh
import logging
import plotly.graph_objects as go

class RealTimeDashboard:
    def __init__(self):
        # 환경 변수 로드
        load_dotenv()
        self.env_data = self.load_env_variables()

        # 데이터 로드
        self.loader = Loader()
        self.detector = Detector()
        self.data_pth = 'Modeling/data/validation/Combined_LabelledData_297_전류고조파평균.json'
        self.df = self.loader.make_df(self.data_pth)
        self.df.index = pd.to_datetime(self.df.index)
        self.visual_df = pd.DataFrame()
        # Streamlit 설정
        st.set_page_config(page_title="Real-time Dashboard", layout="wide")
        st.title("Real-time Anomaly Detection Monitoring Dashboard")

        # 세션 상태 초기화
        self.initialize_session()

    def load_env_variables(self):
        def get_env_variable(var_name, default_value):
            value = os.getenv(var_name, default_value)
            if value == "":
                st.warning(f"환경 변수 {var_name}이 설정되지 않았습니다. 기본값을 사용합니다.")
            return value.split(',')

        env_vars = {
            'VOLT_COLS_PhaseVoltage': "상전압평균,선간전압평균",
            'VOLT_COLS_LinesVoltage': "R상선간전압,S상선전간압,T상선간전압",
            'VOLT_COLS_Harmonics': "전압고조파평균",
            'CURRENT_COLS_Current': "전류평균",
            'CURRENT_COLS_Harmonics': "전류고조파평균",
            'CURRENT_COLS_ActivePower': "유효전력평균",
            'CURRENT_COLS_ReactivePower': "무효전력평균",
            'POWERFACTOR_COLS': "역률평균",
            'OTHERS_COLS_AccumulatedPower': "누적전력량",
            'OTHERS_COLS_Temperature': "온도",
            'OTHERS_COLS_Frequency': "주파수"
        }
        return {key: get_env_variable(key, default_val) for key, default_val in env_vars.items()}

    def initialize_session(self):
        """세션 상태 초기화"""
        if 'group' not in st.session_state:
            st.session_state.group = 'VOLT_COLS'
        if 'iteration' not in st.session_state:
            st.session_state.iteration = 0
        if 'option' not in st.session_state:
            st.session_state.option = list(self.get_group_category_columns()[st.session_state.group].keys())[0]

    def get_group_category_columns(self):
        """그룹별 카테고리와 컬럼 반환"""
        return {
            'VOLT_COLS': {
                "상전압": self.env_data['VOLT_COLS_PhaseVoltage'],
                "선간전압": self.env_data['VOLT_COLS_LinesVoltage'],
                "전압고조파": self.env_data['VOLT_COLS_Harmonics']
            },
            'CURRENT_COLS': {
                "전류": self.env_data['CURRENT_COLS_Current'],
                "전류고조파": self.env_data['CURRENT_COLS_Harmonics'],
                "유효전력": self.env_data['CURRENT_COLS_ActivePower'],
                "무효전력": self.env_data['CURRENT_COLS_ReactivePower']
            },
            'POWERFACTOR_COLS': {
                "역률": self.env_data['POWERFACTOR_COLS']
            },
            'OTHERS_COLS': {
                "누적전력량": self.env_data['OTHERS_COLS_AccumulatedPower'],
                "온도": self.env_data['OTHERS_COLS_Temperature'],
                "주파수": self.env_data['OTHERS_COLS_Frequency']
            }
        }

    def setup_kpis(self, latest_data):
        """KPI 섹션 동적으로 업데이트"""
        # KPI 영역에 사용할 empty placeholders 생성
        kpi1, kpi2, kpi3 = st.columns(3)

        with kpi1:
            if 'temp_placeholder' not in st.session_state:
                st.session_state.temp_placeholder = st.empty()
            st.session_state.temp_placeholder.metric("온도", f"{latest_data['온도']:.1f} °C")

        with kpi2:
            if 'freq_placeholder' not in st.session_state:
                st.session_state.freq_placeholder = st.empty()
            st.session_state.freq_placeholder.metric("주파수", f"{latest_data['주파수']:.0f} Hz")

        with kpi3:
            if 'accumulated_power_placeholder' not in st.session_state:
                st.session_state.accumulated_power_placeholder = st.empty()
            st.session_state.accumulated_power_placeholder.metric("누적전력량", f"{latest_data['누적전력량']:.0f} kWh")


    def setup_group_selection(self):
        """데이터 그룹 및 항목 선택"""
        group_category_cols = self.get_group_category_columns()
        selected_group = st.selectbox("데이터 그룹 선택", list(group_category_cols.keys()), index=list(group_category_cols.keys()).index(st.session_state.group))

        if selected_group != st.session_state.group:
            st.session_state.group = selected_group
            st.session_state.option = list(group_category_cols[selected_group].keys())[0]

        categories = group_category_cols[st.session_state.group]
        options = self.get_options_from_categories(categories)

        selected_option = st.selectbox("표시할 항목 선택", options, index=options.index(st.session_state.option))
        if selected_option != st.session_state.option:
            st.session_state.option = selected_option

    def get_options_from_categories(self, categories):
        """카테고리에서 선택 가능한 항목 반환"""
        options = []
        seen = set()

        for category, columns in categories.items():
            if category not in seen:
                options.append(category)
                seen.add(category)

            for col in columns:
                if col not in seen:
                    options.append(col)
                    seen.add(col)

        return options

    def plot_data1(self, visual_df, option):
        """Plotly로 데이터를 시각화"""
        group_category_cols = self.get_group_category_columns()
        all_categories = {cat: cols for cat_dict in group_category_cols.values() for cat, cols in cat_dict.items()}
        columns_to_plot = all_categories.get(option, [option])

        fig = px.line(visual_df, x=visual_df.index, y=columns_to_plot, title=f"{option} 실시간 변화", labels={'value': 'Value', 'index': 'Time'})

        fig.update_layout(
            plot_bgcolor='rgb(50,50,50)',
            paper_bgcolor='rgb(50,50,50)',
            font=dict(color='white'),
            hovermode='closest',
            xaxis=dict(showgrid=True, gridcolor='rgba(255, 255, 255, 0.2)', gridwidth=1),
            yaxis=dict(showgrid=True, gridcolor='rgba(255, 255, 255, 0.2)', gridwidth=1),
            title_x=0.1 
        )

        return fig

    def plot_data2(self):
        """오토인코더로 이상 탐지 후 그래프 출력 (plotly.express 사용)"""
        
        # 재구성 오류(MSE) 값이 0.2보다 큰 경우, 해당 값을 20으로 치환
        self.visual_df['Reconstruction_error'] = self.visual_df.apply(
            lambda row: 0.17 if row['Reconstruction_error'] > 0.18 else row['Reconstruction_error'], axis=1
        )
        # 재구성 오류(MSE) 시각화
        fig2 = px.line(self.visual_df, x=self.visual_df.index, y='Reconstruction_error', 
                    labels={'Reconstruction_error': '재구성 오류 (MSE)', 'index': '시간'},
                    title="재구성 오류와 이상 탐지")
        
        # 선 아래 구간 색칠 (파란색)
        fig2.update_traces(fill='tozeroy', fillcolor='rgba(0, 0, 255, 0.2)')  # 파란색 반투명

        # 임계값(Threshold) 시각화 (수평선)
        fig2.add_scatter(x=[self.visual_df.index.min(), self.visual_df.index.max()], 
                        y=[self.visual_df['Threshold'].iloc[-1], self.visual_df['Threshold'].iloc[-1]], 
                        mode='lines', name='Threshold', line=dict(color='red', dash='dash'))

        # 이상값 표시 (MSE > threshold인 지점)
        anomaly_indices = self.visual_df.index[self.visual_df['Reconstruction_error'] > self.visual_df['Threshold']]
        anomaly_values = self.visual_df['Reconstruction_error'][self.visual_df['Reconstruction_error'] > self.visual_df['Threshold']]

        fig2.add_scatter(x=anomaly_indices, y=anomaly_values, mode='markers', name='Anomalies', 
                        marker=dict(color='red', size=10, symbol='x'))

        # 임계값을 기준으로 ymin과 ymax 설정
        ymin = min(self.visual_df['Reconstruction_error'].min(), self.visual_df['Threshold'].iloc[-1])
        ymax = self.visual_df['Threshold'].iloc[-1] * 1.3  # 임계값보다 20% 더 크게 설정
        
        # 레이아웃 설정
        fig2.update_layout(
            plot_bgcolor='rgb(50,50,50)',
            paper_bgcolor='rgb(50,50,50)',
            font=dict(color='white'),
            hovermode='closest',
            xaxis=dict(showgrid=True, gridcolor='rgba(255, 255, 255, 0.2)', gridwidth=1),
            yaxis=dict(
                showgrid=True, 
                gridcolor='rgba(255, 255, 255, 0.2)', 
                gridwidth=1, 
                range=[ymin, ymax]  # ymin과 ymax을 임계값을 기준으로 설정
            ),
            title_x=0.1 
        )

        return fig2

    def update_graph(self):
        """실시간 데이터 그래프 업데이트"""
        # Group selection UI를 먼저 업데이트
        self.setup_group_selection()

        col1, col2 = st.columns(2)  # 화면을 두 개 컬럼으로 나눔

        with col1:
            graph_placeholder1 = st.empty()

        with col2:
            graph_placeholder2 = st.empty()

        for i in range(st.session_state.iteration, len(self.df)):
            # self.visual_df에 데이터 추가
            new_data = self.df.iloc[i:i+1]  # 한 번에 한 행만 선택
            self.visual_df = pd.concat([self.visual_df, new_data])  # 기존 데이터에 새 데이터를 추가

            # 실시간 KPI 업데이트
            latest_data = self.df.iloc[i]
            self.setup_kpis(latest_data)  # 동적으로 KPI 갱신
            # 첫 번째 그래프
            fig1 = self.plot_data1(self.visual_df, st.session_state.option)
            max_t = self.visual_df.index.max()
            min_t = max_t - pd.Timedelta(hours=2) if (max_t - self.visual_df.index.min()).total_seconds() > 7200 else self.visual_df.index.min()
            fig1.update_layout(xaxis=dict(range=[min_t, max_t]))
            graph_placeholder1.plotly_chart(fig1, use_container_width=True, key=f"real_monitoring_{self.visual_df.index[-1]}")
            
            if len(self.visual_df) > 10:
                # 오토인코더로 이상 탐지
                detect_results = self.detector.detect(self.visual_df[self.df.columns].tail(10).values.reshape(1, 10, 35))
                # st.write(detect_results)

                # Reconstruction_error와 Threshold 값을 self.visual_df에 추가
                self.visual_df.loc[self.visual_df.index[-1], 'Reconstruction_error'] = detect_results['Reconstruction_error'][0]
                self.visual_df.loc[self.visual_df.index[-1], 'Threshold'] = detect_results['Threshold'][0]

                # 두 번째 그래프: 오토인코더로 이상 탐지
                fig2 = self.plot_data2()

                # 그래프 출력
                graph_placeholder2.plotly_chart(fig2, use_container_width=True, key=f"monitoring_{self.visual_df.index[-1]}")
            
            st.session_state.iteration = i + 1
            time.sleep(0.3)

    def run(self):
        """실시간 대시보드 실행"""
        st_autorefresh(interval=100000, limit=None, key="refresh")
        latest_data = self.df.iloc[0]  # 최초 데이터로 KPI 갱신
        self.setup_kpis(latest_data)
        self.update_graph()

# 실행
if __name__ == "__main__":
    dashboard = RealTimeDashboard()
    dashboard.run()
