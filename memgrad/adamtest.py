import numpy as np


class Adam:
    def __init__(self, f, g, x0, alpha, beta1, beta2, epsilon):
        self.f = f
        self.g = g
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = np.zeros_like(x0)
        self.v = np.zeros_like(x0)
        self.path = [x0.copy()]
        self.obs = []


    def step(self, t):
        x = self.path[-1]
        f = self.f(x[0], x[1])
        g = self.g(x)
        self.obs.append([f, np.linalg.norm(g), x[1]-x[0]**2] )
        self.m = self.beta1 * self.m + (1 - self.beta1) * g
        self.v = self.beta2 * self.v + (1 - self.beta2) * (g ** 2)
        # Bias-corrected estimates
        m_hat = self.m / (1 - self.beta1 ** t)
        v_hat = self.v / (1 - self.beta2 ** t)
        x_new = x - self.alpha * m_hat / (np.sqrt(v_hat) + self.epsilon)
        self.path.append(x_new)

    def iterate(self, max_iter=100):
        for t in range(1,max_iter+1):
            self.step(t)
            l=np.linalg.norm(self.path[-1])
            if l>20:
                break
        return [np.array(self.obs), np.array(self.path)]
