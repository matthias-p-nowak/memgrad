import numpy as np


class AdamBashfort:
    def __init__(self, f, x0, stepsize=0.01):
        self.f = f
        self.x0 = x0
        self.stepsize = stepsize
        self.x = [x0.copy()]
        self.dx = []

    def step(self):
        self.dx.append(self.f(self.x[-1]))
        if len(self.dx) == 1:
            self.x.append(self.x[-1] - self.stepsize * (self.dx[-1]))
        elif len(self.dx) == 2:
            self.x.append(
                self.x[-1] - self.stepsize * (3 * self.dx[-1] - self.dx[-2]) / 2.0
            )
        elif len(self.dx) == 3:
            self.x.append(
                self.x[-1]
                - self.stepsize
                * (23 * self.dx[-1] - 16 * self.dx[-2] + 5 * self.dx[-3])
                / 12.0
            )
        elif len(self.dx) == 4:
            self.x.append(
                self.x[-1]
                - self.stepsize
                * (
                    55 * self.dx[-1]
                    - 50 * self.dx[-2]
                    + 37 * self.dx[-3]
                    - 9 * self.dx[-4]
                )
                / 24.0
            )
        elif len(self.dx) == 5:
            self.x.append(
                self.x[-1]
                - self.stepsize
                * (
                    1901 * self.dx[-1]
                    - 2774 * self.dx[-2]
                    + 2616 * self.dx[-3]
                    - 1274 * self.dx[-4]
                    + 251 * self.dx[-5]
                )
                / 720.0
            )
        else:
            self.x.append(
                self.x[-1]
                - self.stepsize
                * (
                    4277 * self.dx[-1]
                    - 7923 * self.dx[-2]
                    + 9982 * self.dx[-3]
                    - 7298 * self.dx[-4]
                    + 2877 * self.dx[-5]
                    - 475 * self.dx[-6]
                )
                / 1440.0
            )

    def iterate(self, max_order=2, max_iter=100):
        for i in range(max_iter):
            self.step()
            if np.linalg.norm(self.x[-1]) > 4:
                print("divergence")
                break
            self.dx = self.dx[-(max_order - 1) :]
        return np.array(self.x)
