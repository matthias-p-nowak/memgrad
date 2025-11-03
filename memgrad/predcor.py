import numpy as np


class PredCor:
    path = []
    obs = []

    def __init__(self, f, g, x0):
        self.f = f
        self.g = g
        self.path = [x0.copy()]

    def iterate(self, iterations=100):
        xl = self.path[-1]
        sl = 0.000001
        it = 0
        v_old = self.f(xl[0], xl[1])
        vs = v_old
        agg = 0
        impr = 0
        while it < iterations:
            it += 1

            x = self.path[-1]
            v = self.f(x[0], x[1])
            g = self.g(x)
            if v > v_old:
                sl *= 0.9
                impr = 0
            else:
                impr += 1
                if impr > 10:
                    sl *= 2
                else:
                    sl *= 1.001
            v_old = v
            x_new = x - sl / np.linalg.norm(g) * g
            self.obs.append((v, np.linalg.norm(g), sl))
            self.path.append(x_new)
            if v < vs:
                print("jumping")
                agg = 0
                vs = v
                x_new = 2 * x_new - xl
                xl = x_new
                self.path.append((None, None))
                self.path.append(x_new)
                self.obs.append((v, None, None))
            if np.linalg.norm(x_new) > 20:
                break
        return np.array(self.path), np.array(self.obs)
