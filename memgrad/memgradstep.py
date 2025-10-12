import numpy as np


class MemGradStep:
    def __init__(self, f, x0, decay=0.9, memory=5):
        self.f = f
        self.path = [x0.copy()]
        self.memory = memory
        self.decay = decay

    def step(self):
        x = self.path[-1]
        grad = self.f(x)
        min = 0.0001
        for lx in self.path[-self.memory : -2]:
            l = np.linalg.norm(lx - x)
            if l > min:
                min = l
        gradn = np.linalg.norm(grad)
        x_new = x - self.decay * min / gradn * grad
        self.path.append(x_new.copy())

    def iterate(self, iterations=100):
        for i in range(iterations):
            self.step()
            l = np.linalg.norm(self.path[-1])
            if l > 20:
                break
        return np.array(self.path)
