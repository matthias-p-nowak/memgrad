import matplotlib.pyplot as plt
import numpy as np


class TailPlot:
    m: int
    n: int
    m0: int=0
    n0: int=0
    markers: dict[int,np.ndarray] = {}
    traces: dict[int, list] = {}
    plotLines: list[list] = []

    def __init__(self, m: int, n: int):
        """
        Initialize the TailPlot class with the given parameters.

        Parameters
        ----------
        m : int
            iterations between markers.
        n : int
            number of markers.
        """
        self.m = m
        self.n = n
      


    def add_path(self, path: np.ndarray):
        """
        Add a path to the plot.

        Parameters
        ----------
        path : np.ndarray
            The path to be added to the plot.
        """
        for i,x in enumerate(path):
            for j,m in self.markers.items():
                dist = np.linalg.norm(x - m)
                self.traces[j].append((i, float(dist)))
            if self.m0 == 0:
                oldTrace = self.traces.get(self.n0,None)
                if oldTrace is not None:
                    self.plotLines.append(oldTrace)
                self.traces[self.n0] = []
                self.markers[self.n0] = x
                self.n0 += 1
                if self.n0 >= self.n:
                    self.n0 = 0
            self.m0 += 1
            if self.m0 >= self.m:
                self.m0 = 0
        for t in self.traces.values():
            if t is None or len(t)==0:
                continue
            self.plotLines.append(t)


    def show(self):
        """
        Show the plot.
        """
        fig = plt.figure(figsize=(10, 6))
        plotLines = self.plotLines
        colors = plt.cm.hsv(np.linspace(0, 1, self.n))
        for i, pl in enumerate(plotLines):
            pl = np.array(pl)
            c = colors[i % len(colors)]
            plt.semilogy(pl[:,0], pl[:,1], "-", color=c)
            plt.semilogy(pl[0,0], pl[0,1], "o", color=c, markersize=5)
        # plt.semilogy(plotLines[:,0], plotLines[:,1], "-", label="tail distance")
        plt.title(f" ({self.m}-{self.n}) Tail plot")
        plt.legend()
        plt.grid(True)
        plt.show()