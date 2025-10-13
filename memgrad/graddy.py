import adambash
import matplotlib.pyplot as plt
import memgradstep
import numpy as np
from scipy.integrate import RK45

import adamtest


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


x0 = np.array([-1, 2])


def prepPlot():
    # Plot the Rosenbrock function and path
    x = np.linspace(-2, 2, 400)
    y = np.linspace(-1, 3, 400)
    X, Y = np.meshgrid(x, y)
    Z = rosenbrock(X, Y)
    fig=plt.figure(figsize=(10, 6))
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
    path = steepest_descent(grad_rosenbrock, x0, lr=0.0019, max_iter=250)
    plt.plot(path[:, 0], path[:, 1], "r.-", label="Steepest Descent Path")
    plt.show()


def gradFlow(t, xy):
    xy_bar = -1 * grad_rosenbrock(xy, 1, 100)
    return xy_bar


def RKLpath(max_iter=10):
    rk45 = RK45(gradFlow, 0, x0.copy(), 10, max_step=20000)
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

def AdamTest():
    fig1=prepPlot()
    a= adamtest.Adam(rosenbrock, grad_rosenbrock, x0.copy(), 0.9, 0.9, 0.99, 1e-8)
    obs, path=a.iterate(1000)
    plt.plot(path[:, 0], path[:, 1], "r.-", label="calculated")
    plt.legend()
    plt.title("Adam, 1000 iterations")
    fig2=plt.figure(figsize=(10, 6))
    plt.plot(range(len(obs)), obs[:, 0], "r.-", label="objective")
    plt.plot(range(len(obs)), obs[:, 1], "b.-", label="gradient")  
    plt.legend()
    plt2 = plt.twinx()
    plt2.plot(range(len(obs)), obs[:, 2], "g.", label="side") 
    plt2.plot(range(len(path)), path[:, 0], "m.", label="x")
    plt2.plot(range(len(path)), path[:, 1], "c.", label="y")
    plt2.grid(True)
    plt.title("Adam, 1000 iterations")
    plt.show(block=True)


if __name__ == "__main__":
    AdamTest()
    plt.show(block=True)
    print("all done")
