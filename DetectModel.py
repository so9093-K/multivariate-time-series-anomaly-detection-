import onnxruntime as ort
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import joblib, os

load_dotenv()

class Detector:
    
    def __init__(self, 
                 model_path='Modeling/model/lstm_autoencoder.onnx', 
                 scaler_path='Modeling/model/scaler.pkl', 
                 threshold_path='Modeling/model/threshold.pkl'):
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 절대 경로로 변환
        self.model_path = os.path.join(base_dir, model_path)
        self.scaler_path = os.path.join(base_dir, scaler_path)
        self.threshold_path = os.path.join(base_dir, threshold_path)
        
        # 경로 제대로 연결
        self.model = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
        self.scaler = joblib.load(self.scaler_path)
        self.threshold = joblib.load(self.threshold_path)
        
        # 문자열을 리스트로 변환
        volt_cols = os.getenv('VOLT_COLS').split(',')
        current_cols = os.getenv('CURRENT_COLS').split(',')
        powerfactor_cols = os.getenv('POWERFACTOR_COLS').split(',')
        others_col = os.getenv('OTHERS_COLS').split(',')

        # 전체 컬럼 조합
        feature_cols = volt_cols + current_cols + powerfactor_cols + others_col
        self.cols = feature_cols

    def detect(self, x):
        """
        예측 함수: 주어진 데이터를 ONNX 모델을 통해 예측하고, 이상 여부를 판단합니다.
        
        :param x: 입력 데이터 (배치 크기, timesteps, features)
        :return: 이상 여부 (0: 정상, 1: 이상)
        """
        # 데이터를 스케일링
        x_scaled = self.scale(x)
        
        # ONNX 모델을 통한 예측
        input_name = self.model.get_inputs()[0].name
        pred = self.model.run(None, {input_name: x_scaled.astype(np.float32)})[0]
        
        # MSE 계산
        mse = np.mean(np.power(self.flatten(x_scaled) - self.flatten(pred), 2), axis=1)
        isAnomaly = mse > self.threshold
        
        # 데이터프레임으로 출력
        res = {
            'original': pd.DataFrame(self.flatten(x), columns = self.cols),  # 원본 데이터를 평탄화하여 넣기
            'prediction': pd.DataFrame(self.flatten(self.inverse_scale(pred)), columns = self.cols),  # 예측값을 평탄화하여 넣기
            'Reconstruction_error': mse,  # 재구성 오류 (MSE)
            'Threshold': self.threshold,  # 임계값
            'Anomaly': isAnomaly
        }
        
        return res

    def scale(self, X):
        """
        스케일링 함수: 입력 데이터를 미리 학습된 스케일러로 변환합니다.
        
        :param X: 입력 데이터 (배치 크기, timesteps, features)
        :return: 스케일링된 데이터
        """
        X_scaled = X.copy()
        for i in range(X.shape[0]):
            X_scaled[i, :, :] = self.scaler.transform(X[i, :, :])
        return X_scaled
    
    def inverse_scale(self, X_scaled):
        """
        인버스 스케일링 함수: 스케일링된 데이터를 원래 스케일로 복원합니다.
        
        :param X_scaled: 스케일링된 데이터 (배치 크기, timesteps, features)
        :return: 원본 스케일 데이터
        """
        X_inversed = X_scaled.copy()
        for i in range(X_scaled.shape[0]):
            X_inversed[i, :, :] = self.scaler.inverse_transform(X_scaled[i, :, :])
        return X_inversed

    def flatten(self, X):
        """
        플래튼 함수: 시계열 데이터를 1D로 변환하여 MSE 계산에 적합하게 만듭니다.
        
        :param X: 입력 데이터 (배치 크기, timesteps, features)
        :return: 평탄화된 데이터
        """
        flattened_X = np.empty((X.shape[0], X.shape[2]))  # sample x features array.
        for i in range(X.shape[0]):
            flattened_X[i] = X[i, (X.shape[1]-1), :]  # 마지막 timesteps에서 feature 추출
        return flattened_X