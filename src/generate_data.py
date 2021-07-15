import numpy as np
import pandas as pd
import math

class SynDataGenerator():

    def __init__(self, density, S, T, N, weight_max=1, weight_min=-1):
        self.density = density        # 非零权重占所有权重的比重
        self.N = N                    # 概念节点个数
        self.T = T                    # 每个响应时间序列的长度
        self.S = S                    # 响应时间序列的个数
        self.weight_max = weight_max
        self.weight_min = weight_min

    def generate_data(self):
        init_states = self.init_states(self.S)
        weights = self.init_weights()
        obvs = np.zeros(shape=(self.S * self.T, self.N))
        for i in range(self.S):
            obv = self.sigmoid(init_states[i] @ weights)
            for j in range(self.T):
                obvs[i * self.T + j] = obv
                state = obv
                obv = self.sigmoid(state @ weights)
        return obvs, weights

    def generate_data_noised(self, mean=0, sigma=0.1):
        init_states = self.init_states(self.S)
        weights = self.init_weights()
        obvs = np.zeros(shape=(self.S * self.T, self.N))
        for i in range(self.S):
            obv = self.sigmoid(init_states[i] @ weights)
            for j in range(self.T):
                noise = np.random.normal(mean, sigma, size=obv.shape)
                obvs[i * self.T + j] = obv + noise
                state = obv
                obv = self.sigmoid(state @ weights)
        return obvs, weights

    def init_weights(self):
        weights = np.zeros(shape=(self.N * self.N))
        non_zeros_len = int(self.density * self.N * self.N)
        weights[:non_zeros_len] = np.random.uniform(low=self.weight_min, high=self.weight_max, size=non_zeros_len)
        np.random.shuffle(weights)
        weights = np.reshape(weights, newshape=(self.N, self.N))
        for i in range(len(weights)):
            for j in range(len(weights[i])):
                if np.abs(weights[i][j]) <= 0.05:
                    weights[i][j] = 0
        return weights

    def init_states(self, nums=1):
        states = np.random.uniform(low=0, high=1, size=(nums, self.N))
        return states

    def sigmoid(self, u, lamd = 5):
        f = 1 / (1 + math.e ** (-lamd * u))
        return f


if __name__ == '__main__':
    syn = SynDataGenerator(0.4, S=4, T=10, N=10)
    obvs, weights = syn.generate_data()
    print(obvs)
    print(weights)
    print(obvs.shape)
    print(weights.shape)