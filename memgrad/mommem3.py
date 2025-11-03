import numpy as np


class MomMem3:
    minStep = 0.001
    firstOne = True
    momLinear = 5
    momBias = 0
    mom = []
    momentum = 0.8
    value = 0.0

    def __init__(self, f, g, x0, decay=0.2, memory=20, momentum=0.9):
        self.f = f
        self.g = g
        self.path = [x0.copy()]
        self.memory = memory
        self.decay = decay
        self.obs = []

    def step(self):
        x = self.path[-1]
        v = self.f(x[0], x[1])
        gv = self.g(x)
        gvl = np.linalg.norm(gv)
        if self.firstOne:
            self.firstOne = False
            x_new = x - self.minStep * gv / gvl
            self.mom = x_new
            self.value = v
            self.minGvl = gvl
        else:
            stepSize = self.minStep
            if gvl < self.minGvl:
                self.minGvl = gvl
                # self.mom *=0
            for lx in self.path[-self.memory : -1]:
                l = np.linalg.norm(lx - x)
                if l > stepSize:
                    stepSize = l
            self.mom = self.momentum * self.mom + gv / (gvl)
            gradn = np.linalg.norm(self.mom)
            x_new = x - self.decay * stepSize / gradn * self.mom
            self.obs.append((v, gradn, stepSize))
            self.value = v
        self.path.append(x_new.copy())

    def iterate(self, iterations=100):
        for i in range(iterations):
            self.step()
            l = np.linalg.norm(self.path[-1])
            if l > 20:
                break
        return np.array(self.path), np.array(self.obs)
