import numpy as np

class MomMem2:

    obs = []
    start = True
    startStep = 0.000001
    first = True  
    ov =[]


    def __init__(self, f, g, x0, memory=10):
        self.f = f
        self.g = g
        self.path = [x0.copy()]
        self.memory=memory
        

    def step(self,i: int):
        x = self.path[-1]
        v = self.f(x[0], x[1])
        self.ov.append(v)
        g = self.g(x)
        gl = np.linalg.norm(g)
        moml=gl
        beta=0
        if self.start:
            if self.first:
                self.first=False
                self.fValue=v
            else:
                if v > self.fValue:
                    self.start=False
                    self.mom = g
                    # self.stepSize *= 0.1
                else:
                    self.stepSize *= 2.0
            self.fValue=v
            x_new = x - self.stepSize / gl * g  
        else:
            vmin= min(self.ov[-self.memory:])
            vmax = max(self.ov[-self.memory:])
            beta=(vmax -v )/ (vmax - vmin) * 0.9 + 0.1
            mom = beta*self.mom + g
            moml = np.linalg.norm(mom)
            x_new = x - self.stepSize / moml * mom 
            self.mom=mom
        self.obs.append((v, gl, self.stepSize, beta, moml))
        self.path.append(x_new)

    def iterate(self, iterations=100):
        self.stepSize = self.startStep  
        self.iterations = iterations
        for i in range(iterations):
            self.step(i)
            l = np.linalg.norm(self.path[-1])
            if l > 20:
                break
        return np.array(self.path), np.array(self.obs)