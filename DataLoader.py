from dotenv import load_dotenv
import numpy as np 
import pandas as pd
import os, json

load_dotenv()

class Loader:
    def __init__(self):
        # 문자열을 리스트로 변환
        volt_cols = os.getenv('VOLT_COLS').split(',')
        current_cols = os.getenv('CURRENT_COLS').split(',')
        powerfactor_cols = os.getenv('POWERFACTOR_COLS').split(',')
        others_col = os.getenv('OTHERS_COLS').split(',')

        # 전체 컬럼 조합
        feature_cols = volt_cols + current_cols + powerfactor_cols + others_col
        self.cols = ['TIMESTAMP', 'LABEL_NAME'] + feature_cols
    
    def make_df(self, pth):
            with open(pth, encoding = 'utf-8-sig') as f : data = json.load(f)
            df = pd.DataFrame(data['data'])
            df_pivot = df.pivot_table(index='TIMESTAMP', columns='ITEM_NAME', values='ITEM_VALUE', aggfunc='first')
            df_label = df[['TIMESTAMP', 'LABEL_NAME']].drop_duplicates().set_index('TIMESTAMP')
            df = df_pivot.join(df_label, on='TIMESTAMP').reset_index()
            df = df[self.cols]
            df.drop(columns=['LABEL_NAME'], inplace=True)
            df.set_index('TIMESTAMP', inplace=True)
            return df 
        
    def temporalize(self, X, lookback=10):
        '''
        각 시점에 대해 과거 lookback 기간 동안의 데이터를 통해 새로운 3차원 배열(output_X) 생성
        output_y는 lookback 기간 후의 y값을 포함
        
        입력:
        X         시간 순으로 정렬된 2D 넘파이 배열, 형태: 
                (관측치 수 x 특성 수)
        y         X와 인덱스가 일치하는 1D 넘파이 배열, 
                즉, y[i]는 X[i]와 대응해야 함. 형태: 관측치 수.
        lookback  과거 기록을 조회할 윈도우 크기. 스칼라 형태.

        출력:
        output_X  3D 넘파이 배열의 형태: 
                ((관측치 수-lookback-1) x lookback x 
                특성 수)
        output_y  1D 배열의 형태: 
                (관측치 수-lookback-1), X와 정렬됨.
        '''
        
        X = X.copy()
        output_X = []
        
        for i in range(len(X) - lookback - 1):
            t = []
            for j in range(1, lookback + 1):
                # 과거 기록을 lookback 기간 동안 수집
                t.append(X[[(i + j + 1)], :])
            output_X.append(t)

        return np.squeeze(np.array(output_X))
    