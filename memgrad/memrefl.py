import numpy as np


class MemRefl:
    firstOne = True
    minStep = 0.0001

    def __init__(self, f, g, x0, decay=0.2, memory=10):
        self.f = f
        self.g = g
        self.path = [x0.copy()]
        self.obs = []
        self.memory = memory
        self.decay = decay

    def step(self):
        # momentum points into direction of descent!
        x = self.path[-1]
        v = self.f(x[0], x[1])
        gv = self.g(x)
        gvl = np.linalg.norm(gv)
        if self.firstOne:
            self.firstOne = False
            self.momentum = -self.minStep * gv
            x_new = x + self.momentum
            stepSize = np.linalg.norm(self.momentum)
            moml = stepSize
            sp = 0
        else:
            stepSize = 0
            for lx in self.path[-self.memory : -1]:
                l = np.linalg.norm(lx - x)
                if l > stepSize:
                    stepSize = l
            moml = np.linalg.norm(self.momentum)
            # calculating amount to reflect
            sp = np.dot(gv, self.momentum)  #  |momentum| * |gv| *sin(m,g)
            if sp < 0:
                # adding to direction
                self.momentum -= 0.1 * moml / gvl * gv  # increasing
            else:
                self.momentum -= 2 * sp / (gvl * gvl) * gv
            moml = np.linalg.norm(self.momentum)
            x_new = x + self.decay * stepSize / moml * self.momentum
        self.path.append(x_new)
        self.obs.append((v, gvl, stepSize, moml, sp))

    def iterate(self, iterations=100):
        for i in range(iterations):
            self.step()
            l = np.linalg.norm(self.path[-1])
            if l > 20:
                break
        return np.array(self.path), np.array(self.obs)
