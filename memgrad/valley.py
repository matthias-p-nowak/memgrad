import numpy as np

class Valley:
    firstOne = True
    minStep = 0.0001

    def __init__(self, f, g, x0, memory=10):
        self.f = f
        self.g = g
        self.vx = x0.copy()
        self.path = [x0]
        self.obs = []
        self.memory = memory
        self.objs =[]
        self.restart=False
        self.moml = self.minStep
        self.rejected =[]

    def step(self):
        x = self.path[-1]
        v = self.f(x[0], x[1])
        gv = self.g(x)
        gvl = np.linalg.norm(gv)
        sp = 0
        if self.firstOne:
            self.firstOne = False
            self.momentum = -self.minStep * gv
            self.vx = x
            self.objs.append(v)
        else:
            mv = max(self.objs[-self.memory:])
            if v > mv:
                self.momentum *= 0.3
                self.restart=True
                self.rejected.append(x)
            else:
                # self.restart = False
                self.vx = x
                if self.restart:
                    self.restart=False
                    moml = np.linalg.norm(self.momentum)
                    self.momentum = - gv * moml / gvl
                    # self.momentum = -self.minStep * gv
                else:
                    # calculating amount to reflect
                    self.objs.append(v)
                    sp = np.dot(gv, self.momentum)  #  |momentum| * |gv| *sin(m,g)
                    if sp < 0:
                        self.momentum *= 1.5
                    else:
                        # the reflection on the level line at the point x 
                        self.momentum -= 2* sp / (gvl*gvl) *gv
        x_new = self.vx + self.momentum
        self.path.append(x_new)
        self.obs.append((v, gvl, sp, np.linalg.norm(self.momentum)))
        # 

    def iterate(self, iterations=100):
        for i in range(iterations):
            self.step()
            l = np.linalg.norm(self.path[-1])
            if l > 20:
                break
        return np.array(self.path), np.array(self.obs), np.array(self.rejected)
