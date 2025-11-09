import numpy as np

class Valley:
    firstOne = True
    minStep = 0.0001

    def __init__(self, f, g, x0, memory=10, eva = 0.9):
        self.f = f
        self.g = g
        self.vx = x0.copy()
        self.path = [x0]
        self.obs = []
        self.memory = memory
        self.eva = eva
        self.objs =[]
        self.restart=False
        self.moml = self.minStep
        self.rejected =[]
        self.restarted=[]

    def step(self):
        x = self.path[-1]
        v = self.f(x[0], x[1])
        gv = self.g(x)
        gvl = np.linalg.norm(gv)
        sp = 0
        mult=1.0
        if self.firstOne:
            self.firstOne = False
            self.momentum = -self.minStep * gv
            self.vx = x
            self.objs.append(v)
            self.meangrad = gvl
        else:
            mv = max(self.objs[-self.memory:])
            self.meangrad = self.eva * self.meangrad + (1 - self.eva) * gvl
            if v > mv:
                self.momentum *= 0.5
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
                    self.restarted.append(x)
                else:
                    # calculating amount to reflect
                    self.objs.append(v)
                    sp = np.dot(gv, self.momentum)  #  |momentum| * |gv| *sin(m,g)
                    if sp < 0:
                        self.momentum *= 1.5
                    else:
                        # the reflection on the level line at the point x 
                        mult = np.sqrt( gvl / self.meangrad)
                        self.momentum -= 2* sp / (gvl*gvl) *gv
        x_new = self.vx + mult * self.momentum
        self.path.append(x_new)
        self.obs.append((v, gvl, sp, np.linalg.norm(self.momentum)))
        # 

    def iterate(self, iterations=100):
        for i in range(iterations):
            self.step()
            l = np.linalg.norm(self.path[-1])
            if l > 20:
                break
        return np.array(self.path), np.array(self.obs), np.array(self.rejected), np.array(self.restarted)
