# 用合成数据用于fcm实验
import numpy as np
import math
from dataset.generate_data import SynDataGenerator
from fcms.isss2 import ISSS2

# 全局参数
density = 0.4
concept_nums = 20
response_nums = 1
data_len = 20
threshold = 0.05

def sigmoid(x):
    return 1 / (1 + math.e ** (-5*x))

# 比较权重之间的误差
def model_errors(weights_true, weigths_predicted):
    return np.mean(np.abs(weights_true - weigths_predicted))

# 比较输出的相应序列之间的误差
def data_errors(obvs, weights_predicted):
    error_sum = 0
    rsps = []
    outputs = []
    for i in range(response_nums):
        for j in range(data_len - 1):
            state = obvs[i*data_len + j, :]
            rsp = obvs[i * data_len + j + 1, :]
            rsps.append(rsp)
            output = sigmoid(state @ weights_predicted)
            outputs.append(output)
            error_sum += np.mean(np.square(rsp - output))
    return error_sum / ((data_len - 1) * response_nums)

# 矩阵编码
def weights_encode(weights):
    encoded_weights = np.zeros_like(weights)
    for i in range(len(weights)):
        for j in range(len(weights[i])):
            if abs(weights[i][j]) <= threshold:
                encoded_weights[i][j] = 0
            else:
                encoded_weights[i][j] = 1

    return encoded_weights

# 比较编码后的SS Mean指标
def SS_means(weights_true, weights_predicted):
    encoded_weights_true = weights_encode(weights_true)
    encoded_weights_predicted = weights_encode(weights_predicted)
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(encoded_weights_true)):
        for j in range(len(encoded_weights_true[i])):
            true_value = encoded_weights_true[i][j]
            predicted_value = encoded_weights_predicted[i][j]
            if true_value == 0:
                if predicted_value == 0:
                    TP += 1
                else:
                    FP += 1
            else:
                if predicted_value == 1:
                    FN += 1
                else:
                    TN += 1
    Specificity = TP / (TP + FN)
    Sensitivity = TN / (TN + FP)
    SS_mean = 2 * Specificity * Sensitivity / (Specificity + Sensitivity)
    return SS_mean


def run_fcm_isss2():
    # ------------- 合成数据产生
    syn = SynDataGenerator(density=density, S=response_nums, T=data_len, N=concept_nums)
    obvs, weights = syn.generate_data()
    # obvs, weights = syn.generate_data_noised(sigma=0.001)
    print(obvs.shape)
    # -------------- FCM训练拟合权重
    isss = ISSS2(U=obvs, S=response_nums, T=data_len, N=concept_nums, display=False, lamd=0.005, beta=0.0001, gamma=0.0001)
    predicted_weights = isss.fit()
    print(weights)
    print(predicted_weights)
    print(model_errors(weights, predicted_weights))
    print(data_errors(obvs, predicted_weights))
    print(SS_means(weights, predicted_weights))


if __name__ == '__main__':

    run_fcm_isss2()