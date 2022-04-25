fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8, 
                10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7, 
                7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

import numpy as np

'''np.column_stack(([1,2,3], [4,5,6])) # 전달받은 리스트를 일렬로 세운 다음 차례대로 나란히 연결 즉 열과 열로 짝지어줌'''
'''np.concatenate(([1,2,3], [4,5,6])) # 전달받은 리스트를 가로로 붙여줌'''

'''np.column_stack((fish_length, fish_weight))
np.concatenate((fish_length, fish_weight))'''

'''np.one() 와 np.zeros() 를 통해 [1] 와 [0]으로 채울 수 있음'''
'''print(np.one(5))'''

from sklearn.model_selection import train_test_split

fish_data = np.column_stack((fish_length, fish_weight))
fish_target = np.concatenate((np.ones(35), np.zeros(14)))

'''train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, random_state=42)''' # -> 훈련 데이터가 작아 sampling bias 발생
'''print(test_target)''' # -> [1. 0. 0. 0. 1. 1. 1. 1. 1. 1. 1. 1. 1.]

train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, stratify=fish_target, random_state=42) 
#stratify 매개변수에 타깃 데이터 전달 시 클래스 비율에 맞게 데이터 나눔. 훈련 데이터가 작거나 특정 클래스의 샘플 개수가 적을 때 특히 유용. 
'''print(test_target)''' # -> [0. 0. 1. 0. 1. 0. 1. 1. 1. 1. 1. 1. 1.]

# 전달되는 리스트나 배열을 비율에 맞게 훈련 세트와 테스트 세트로 나누어 줍니다. 나누기 전에 알아서 섞어줌. random_state 는 랜덤 시드 지정

'''print(train_input.shape, test_input.shape''' # -> (36,2) (13,2)
'''print(train_target.shape, test_target.shape''' # -> (36,) (13,)




