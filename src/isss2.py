import math
import numpy as np


class ISSS2():
    """
    ISSS 算法 for FCM权重学习
    Iterative Smoothing algorithm with Structure Sparsity (ISSS)
    """

    def __init__(self, U, S, T, N, lamd=5*1e-2, beta=1e-3, gamma=1e-3, display = False):
        self.U = U                 # 观测值矩阵（隶属度矩阵）
        self.S = S                 # 响应序列个数
        self.T = T                 # 每个响应序列的长度
        self.N = N                 # 概念节点个数
        self.Pi, self.Y = self._create_Pi_Y()   # 观测矩阵以及其标签
        self.epsilon = 1e-8        # 迭代停止阈值
        self.mu = 1e-3                # 非负平滑参数，mu越小，su(Xi)越接近TV(Xi)
        self.ant = 3
        self.beta = beta            # L2系数
        self.lamd = lamd            # L1系数
        self.gamma = gamma           # 平滑全变分系数
        self.X_max = 1              # 权重上限
        self.X_min = -1             # 权重下限
        self.display = display


    def fit(self):
        """
        权重学习
        :return:
        """
        trained_X = []
        for i in range(self.N):
            k = 1  # 迭代次数
            Xi = self._init_weights()
            Yi = self.Y[:, i]  # Label (i节点下一时刻的输出)
            Yi = self._inverse_sigmod(Yi)
            Xi0 = self._init_weights()
            t0 = 1  # 上一时刻t
            Lk = 1

            while (np.linalg.norm(self._F(Xi, Yi) - self._F(Xi0, Yi), ord=2) >= self.epsilon):

                ik = 0
                while (self._F(self._soft_func(Xi, lamd=Lk), Yi) > self._Q(self._soft_func(Xi, lamd=Lk), Xi, Lk, Yi)):
                    ik += 1
                    Lk = self.ant ** ik * Lk

                t = (1 + np.sqrt(1 + 4 * t0 ** 2)) / 2
                v = Xi + (t0 - 1) / t * (Xi - Xi0)
                f_der = self.Pi.T @ (self.Pi @ v - Yi) + self.beta * v + self.gamma * self._su_der(v)

                Xi0 = Xi
                Xi = self._soft_func(v - 1 / Lk * f_der, self.lamd * (1 / Lk))

                # 更新变量
                t0 = t
                k += 1

                if self.display:
                    print(k, Xi)

            trained_X.append(Xi)
        return np.array(trained_X).T


    def _F(self, Xi, Yi):
        """
        真实的目标函数
        :return:
        """
        ols = 1 / 2 * ((np.linalg.norm(Yi - self.Pi @ Xi, ord=2)) ** 2)
        l2 = self.beta / 2 * (np.linalg.norm(Xi, ord=2) ** 2)
        tv = self.gamma * np.linalg.norm(np.diff(Xi), ord=1)
        l1 = self.lamd * np.linalg.norm(Xi, ord=1)
        return np.array([ols + l2 + tv + l1])

    def _Q(self, Xi, Xi_, Lk, Yi):
        """
        F在Xi_处的近似函数
        :param Xi:
        :param Xi_:
        :param L: L_k-1
        :param Yi:
        :return:
        """
        fu_der = self.Pi.T @ (self.Pi @ Xi_ - Yi) + self.beta * Xi_ + self.gamma * self._su_der(Xi_)
        QLk = self._fu(Xi_, Yi) + np.dot(Xi - Xi_, fu_der) + Lk/2 * np.linalg.norm(Xi - Xi_, ord=2) ** 2 + self.lamd * np.linalg.norm(Xi, ord=1)
        return QLk


    def _Fu(self, Xi, Yi):
        """
        平滑后的目标函数，
        且Fu(Xi) = F(Xi)，当k->无穷
        :return:
        """
        return self._fu(Xi, Yi) + self.lamd * np.linalg.norm(Xi, ord=1)

    def _fu(self, Xi, Yi):
        """
        fu是平滑凸函数，即是连续可导的
        :return:
        """
        ols = 1 / 2 * ((np.linalg.norm(Yi - self.Pi @ Xi, ord=2)) ** 2)
        l2 = self.beta / 2 * (np.linalg.norm(Xi, ord=2) ** 2)
        tv = self.gamma * self._su(Xi)
        return ols + l2 + tv

    def _su(self, Xi):
        """
        Su(Xi)是TV(Xi)的Nesterov平滑形式
        :return:
        """
        su = np.dot(self._zmu(Xi), self._A() @ Xi) - self.mu / 2 * (np.linalg.norm(self._zmu(Xi)) ** 2)
        return su

    def _su_der(self, Xi):
        """
        ▽Su(Xi)，Su(Xi)的导数
        :param Xi:
        :return:
        """
        return self._A().T @ self._zmu(Xi)

    def _A(self):
        """
        AG的垂直级联矩阵
        :return:
        """
        A = np.zeros(shape=(self.N - 1, self.N))
        for i in range(A.shape[0]):
            A[i][i] = -1
            A[i][i+1] = 1
        return A

    def _zmu(self, Xi):
        """
        求z*u(Xi)
        :return:z*u(Xi)
        """
        zmu = []
        for Ag in self._A():
            x = np.array([(1 / self.mu) * (Ag @ Xi)])
            if np.linalg.norm(x, ord=2) <= 1:
                zmu.append(x)
            else:
                zmu.append(x / np.linalg.norm(x, ord=2))
        zmu = np.array(zmu).reshape(-1)
        return zmu

    def _init_weights(self):
        """
        初始化一列权重
        :param N:
        :return:
        """
        return np.random.uniform(self.X_min, self.X_max, size=(self.N,))

    def _create_Pi_Y(self):
        """
        生成观测值与之对应的标签
        :return:
        """
        Pi = np.zeros(shape=(self.S * (self.T - 1), self.N))
        Y = np.zeros(shape=(self.S * (self.T - 1), self.N))
        for i in range(self.S):
            Pi[(self.T - 1) * i : (self.T - 1)*(i + 1)] = self.U[self.T * i : self.T * i + self.T - 1]
            Y[(self.T - 1) * i : (self.T - 1)*(i + 1)] = self.U[self.T * i + 1 : self.T * i + self.T]
        return Pi, Y

    def _X_constraint(self, Xi):
        """
        保证X的无穷范数小于等于1
        :param Xi:
        :return:
        """
        # Xi = np.where(Xi < 1, Xi, 1)
        # Xi = np.where(Xi > -1, Xi, -1)
        Xi = np.where(Xi < 1, Xi, np.random.uniform(self.X_min, self.X_max, 1))
        Xi = np.where(Xi > -1, Xi, np.random.uniform(self.X_min, self.X_max, 1))
        return Xi

    def normalize(self, Xi):
        """
        权重归一化处理[-1,1]
        :return:
        """
        _range = np.max(abs(Xi))
        return Xi / _range

    def _inverse_sigmod(self, u, lamd=5):
        """
        sigmoid的反函数
        :param u:
        :param lamd:
        :return:
        """
        Yi = []
        for i in range(len(u)):
            # 防止出现log(x)其中x为0
            if u[i] == 1:
                u[i] = u[i] - 0.001
            Yi.append(-1 / lamd * np.log(1 / u[i] - 1))
        return np.array(Yi)


    def _soft_func(self, u, lamd = 0.):
        """
        软阈值函数
        :param u:
        :return:
        """
        ui_list = []
        for ui in u:
            if ui <= -lamd:
                ui_list.append(ui + lamd)
            elif ui >= lamd:
                ui_list.append(ui - lamd)
            else:
                ui_list.append(0)
        return np.array(ui_list)



if __name__ == '__main__':
    S = 2
    T = 3
    N = 4
    U = np.array([[0.1,0.1,0.1,0.1], [0.2,0.2,0.2,0.2], [0.3,0.3,0.3,0.3], [0.4,0.4,0.4,0.4], [0.5,0.5,0.5,0.5], [0.6,0.6,0.4,0.4]])  # 3 * 4
    isss = ISSS2(U, S, T, N, display=True)
    isss.fit()




