import adambash

import adamtest
import matplotlib.pyplot as plt
import memgradstep
import memrefl
import mommem

import mommem2
import mommem3
import numpy as np
import predcor
import tailplot
from scipy.integrate import RK45
import valley


# Rosenbrock function
def rosenbrock(x, y, a=1, b=100):
    return (a - x) ** 2 + b * (y - x**2) ** 2


# Gradient of the Rosenbrock function
def grad_rosenbrock(xy, a=1, b=100):
    x, y = xy
    dx = -2 * (a - x) - 4 * b * x * (y - x**2)
    dy = 2 * b * (y - x**2)
    return np.array([dx, dy])


# Steepest descent method
def steepest_descent(f_grad, x0, lr=0.001, max_iter=100, tol=1e-6):
    x = x0
    path = [x0.copy()]
    for i in range(max_iter):
        grad = f_grad(x)
        if np.linalg.norm(grad) > 20000:
            break
        x_new = x - lr * grad
        path.append(x_new.copy())
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    return np.array(path)


x0 = np.array([-1.0, 2.0])


def prepPlot():
    # Plot the Rosenbrock function and path
    x = np.linspace(-2, 2, 400)
    y = np.linspace(-1, 3, 400)
    X, Y = np.meshgrid(x, y)
    Z = rosenbrock(X, Y)
    fig = plt.figure(figsize=(10, 6))
    levels = np.array([i**2 for i in range(20)])
    plt.contour(X, Y, Z, levels=levels, cmap="rainbow")
    plt.plot(1, 1, "bo", label="Minimum (1,1)")
    plt.plot(-1, 2, "co", label="Start")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Rosenbrock Function")
    plt.legend()
    plt.grid(True)
    return fig


def Steepest():
    prepPlot()
    # Initial point and optimization
    path = steepest_descent(grad_rosenbrock, x0, lr=0.0013, max_iter=500)
    plt.plot(path[:, 0], path[:, 1], "r.-", label="Steepest Descent Path")
    plt.show()
    tp = tailplot.TailPlot(30, 8)
    tp.add_path(path)
    tp.show()


def gradFlow(t, xy):
    xy_bar = -1 * grad_rosenbrock(xy, 1, 100)
    return xy_bar


def RKLpath(max_iter=10):
    rk45 = RK45(gradFlow, 0, x0.copy(), 20, max_step=20000)
    path = [x0.copy()]
    for s in range(max_iter):
        rk45.step()
        path.append(rk45.y.copy())
    return np.array(path)


def RungeKutta():
    prepPlot()
    # Solve using Runge-Kutta 4(5) (RK45)
    path = RKLpath(1000)
    plt.plot(path[:, 0], path[:, 1], "r.-", label="calculated")
    plt.title("Runge-Kutta 4(5),1000 iterations")
    plt.show()


def AdamBash():
    prepPlot()
    ab = adambash.AdamBashfort(grad_rosenbrock, x0.copy(), 0.0005)
    path = ab.iterate(max_order=3, max_iter=1000)
    plt.plot(path[:, 0], path[:, 1], "r.-", label="calculated")
    plt.title("Adam-Bashforth,1000 iterations")
    plt.show()


def MemGrad():
    prepPlot()
    mgs = memgradstep.MemGradStep(grad_rosenbrock, x0.copy(), decay=0.4, memory=20)
    path = mgs.iterate(1000)
    plt.plot(path[:, 0], path[:, 1], "r.-", label="calculated")
    plt.legend()
    plt.title("MemGradStep,1000 iterations")
    plt.show()
    tp = tailplot.TailPlot(30, 8)
    tp.add_path(path)
    tp.show()


def AdamTest():
    fig1 = prepPlot()
    a = adamtest.Adam(rosenbrock, grad_rosenbrock, x0.copy(), 0.25, 0.9, 0.99, 1e-8)
    obs, path, vh, mh = a.iterate(1000)
    plt.plot(path[:, 0], path[:, 1], "r.-", label="calculated")
    plt.legend()
    plt.title("Adam, 1000 iterations")
    fig2 = plt.figure(figsize=(10, 6))
    # plt.legend()
    plt.plot(obs[:, 2], "g.", label="side")
    plt.plot(path[:, 0], "m.", label="x")
    plt.plot(path[:, 1], "c.", label="y")
    plt.legend()
    plt.grid(True)
    plt.title("Adam, 1000 iterations")
    fig3 = plt.figure(figsize=(10, 6))
    # plt.plot(vh, "r.-", label="vh")
    plt.semilogy(obs[:, 0], "r.-", label="objective")
    plt.semilogy(obs[:, 1], "b.-", label="gradient")
    plt.semilogy(mh, "c.-", label="mh")
    plt.semilogy(vh, "m.-", label="vh")
    plt.legend()
    plt.grid(True)
    tp = tailplot.TailPlot(50, 8)
    tp.add_path(path)
    tp.show()
    plt.show(block=True)


def MomMemTest():
    fig1 = prepPlot()
    mms = mommem.MomMem(
        # rosenbrock, grad_rosenbrock, x0, decay=0.1, memory=20, momentum=0.95
        rosenbrock,
        grad_rosenbrock,
        x0,
        decay=0.4,
        memory=5,
        momentum=0.4,
    )
    path, obs = mms.iterate(1000)
    plt.plot(path[:, 0], path[:, 1], "r.-", label="calculated")
    plt.legend()
    plt.title("MomMem,1000 iterations")
    fig2 = plt.figure(figsize=(10, 6))
    plt.semilogy(obs[:, 0], "g.-", label="objective")
    plt.semilogy(obs[:, 1], "r.-", label="gradn")
    plt.semilogy(obs[:, 2], "b.-", label="stepsize")
    plt.legend()
    # plt.show()
    tp = tailplot.TailPlot(50, 8)
    tp.add_path(path)
    tp.show()



def MomMem2Test():
    fig1 = prepPlot()
    mms = mommem2.MomMem2(rosenbrock, grad_rosenbrock, x0)
    path, obs = mms.iterate(300)
    plt.plot(path[:, 0], path[:, 1], "b.-", label="calculated")
    plt.legend()
    plt.title("MomMem2, 1000 iterations")
    fig2 = plt.figure(figsize=(10, 6))
    plt.semilogy(obs[:, 0], "g.-", label="objective")
    plt.semilogy(obs[:, 1], "r.-", label="gradn")
    plt.semilogy(obs[:, 2], "b.-", label="stepsize")
    plt.legend()
    plt.grid(True)
    plt.title("MomMem2, values")
    plt.twinx()
    plt.plot(obs[:, 3], "m.-", label="beta")
    plt.grid(True)
    plt.show()


def PredCorTest():
    fig1 = prepPlot()
    mms = predcor.PredCor(rosenbrock, grad_rosenbrock, x0)
    path, obs = mms.iterate(1000)
    plt.plot(path[:, 0], path[:, 1], "b.-", label="calculated")
    plt.legend()
    plt.title("PredCor, 1000 iterations")
    fig2 = plt.figure(figsize=(10, 6))
    plt.semilogy(obs[:, 0], "g.-", label="objective")
    plt.semilogy(obs[:, 1], "r.-", label="gradn")
    plt.semilogy(obs[:, 2], "b.-", label="stepsize")
    plt.legend()
    plt.show()


def MomMem3Test():
    fig1 = prepPlot()
    mms = mommem3.MomMem3(
        # rosenbrock, grad_rosenbrock, x0, decay=0.1, memory=20, momentum=0.95
        rosenbrock,
        grad_rosenbrock,
        x0,
        decay=0.2,
        memory=10,
        momentum=0.8,
    )
    path, obs = mms.iterate(500)
    plt.plot(path[:, 0], path[:, 1], "r.-", label="calculated")
    plt.legend()
    plt.title("MomMem3, 200 iterations")
    fig2 = plt.figure(figsize=(10, 6))
    plt.semilogy(obs[:, 0], "g.-", label="objective")
    plt.semilogy(obs[:, 1], "r.-", label="gradn")
    plt.semilogy(obs[:, 2], "b.-", label="stepsize")
    plt.legend()
    # plt.twinx()
    # plt.plot(path[:, 0], "m.-", label="x")
    # plt.legend()
    plt.show()


def MemReflTest(iterations=1000):
    fig1 = prepPlot()
    mms = memrefl.MemRefl(rosenbrock, grad_rosenbrock, x0, decay=0.2, memory=30)
    path, obs = mms.iterate(iterations)
    plt.plot(path[:, 0], path[:, 1], "r.-", label="calculated")
    plt.legend()
    plt.title(f"MemRefl, {iterations} iterations")
    fig2 = plt.figure(figsize=(10, 6))
    plt.semilogy(obs[:, 0], "g.-", label="objective")
    plt.semilogy(obs[:, 1], "r.-", label="gradn")
    plt.semilogy(obs[:, 2], "b.-", label="stepsize")
    plt.semilogy(obs[:, 3], "m.-", label="moml")
    plt.legend()
    plt.grid(True)
    plt.twinx()
    plt.plot(obs[:, 4], "c.-", label="sp")
    plt.legend()
    plt.grid(True)
    plt.show()

def ValleyTest(iterations=1000):
    fig1 = prepPlot()
    mms = valley.Valley(rosenbrock, grad_rosenbrock, x0, memory=20)
    path, obs, rejected = mms.iterate(iterations)
    plt.plot(path[:, 0], path[:, 1], "r.-", label="calculated")
    plt.plot(rejected[:, 0], rejected[:, 1], "mo", markersize=5, label="rejected")
    plt.legend()
    plt.title(f"Valley, {iterations} iterations")
    fig2 = plt.figure(figsize=(10, 6))
    plt.semilogy(obs[:, 0], "g.-", label="objective")
    plt.semilogy(obs[:, 1], "r.-", label="gradn")
    plt.semilogy(obs[:, 3], "b.-", label="stepsize")
    plt.legend()
    # plt.twinx()
    # plt.plot(obs[:, 2], "c.-", label="sp")
    plt.legend()
    tp = tailplot.TailPlot(20, 8)
    tp.add_path(path)
    tp.show()
    plt.show()

if __name__ == "__main__":
    # Steepest()
    # RungeKutta()
    # AdamBash()
    # MemGrad()
    # AdamTest()
    # MomMem2Test()
    # MomMemTest()
    # MomMem3Test()
    # PredCorTest()
    # MemReflTest(80)
    ValleyTest(300)
    plt.show(block=True)
    print("all done")
