from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np


def errBnds1(X, Y, name):
    Ym = np.nanmedian(Y, axis=0)
    q1 = np.nanpercentile(Y, 25, axis=0)
    q3 = np.nanpercentile(Y, 75, axis=0)
    plt.plot(X, Ym, c="b", mfc="b", ls="-", marker="o", label=name)
    plt.fill_between(X, q1, q3, color='g', alpha=0.2)

def errBnds2(X, Y, name):
    Ym = np.nanmedian(Y, axis=0)
    q1 = np.nanpercentile(Y, 25, axis=0)
    q3 = np.nanpercentile(Y, 75, axis=0)
    q1 = q1.reshape(1, len(q1)) 
    q3 = q1.reshape(1, len(q3)) 
    q =  np.vstack((q1, q3))
    plt.errorbar(X, Ym,  yerr=q, c="r", mfc="r", ls="--", marker="X", label=name)

if __name__ == "__main__":
    # datetime object containing current date and time
    # now = datetime.now()
    # dt_string = now.strftime("%d%m%Y_%H%M%S")

    # Load the data
    train_infer = np.loadtxt("data/fhn/train_err_infer.csv", delimiter=",", dtype=float)
    dims = train_infer[0,:]
    train_infer = train_infer[1:,:]
    train_intru = np.loadtxt("data/fhn/train_err_intru.csv", delimiter=",", dtype=float)[1:,:]
    test1_infer = np.loadtxt("data/fhn/test1_err_infer.csv", delimiter=",", dtype=float)[1:,:]
    test1_intru = np.loadtxt("data/fhn/test1_err_intru.csv", delimiter=",", dtype=float)[1:,:]
    test2_infer = np.loadtxt("data/fhn/test2_err_infer.csv", delimiter=",", dtype=float)[1:,:]
    test2_intru = np.loadtxt("data/fhn/test2_err_intru.csv", delimiter=",", dtype=float)[1:,:]

    # Training data
    fig1, ax1 = plt.subplots()
    errBnds1(dims, train_intru, "Intrusive")
    errBnds2(dims, train_infer, "LnL")
    plt.yscale('log')
    plt.title("Error over training trajectories")
    plt.xlabel("dimension n")
    plt.ylabel("Relative state error")
    plt.legend()
    # plt.savefig("plots/fhn_train_err" + dt_string + ".png")
    plt.savefig("plots/fhn_train_err.png")
    plt.show()

    # Test1 data
    fig2, ax2 = plt.subplots()
    errBnds1(dims, test1_intru, "Intrusive")
    errBnds2(dims, test1_infer, "LnL")
    plt.yscale("log")
    plt.title("Test1 error over new trajectories")
    plt.xlabel("dimension n")
    plt.ylabel("Relative state error")
    plt.legend()
    # plt.savefig("plots/fhn_test1_err" + dt_string + ".png")
    plt.savefig("plots/fhn_test1_err.png")
    plt.show()

    fig3, ax3 = plt.subplots()
    errBnds1(dims, test2_intru, "Intrusive")
    errBnds2(dims, test2_infer, "LnL")
    plt.yscale("log")
    plt.title("Test2 errors over new trajectories")
    plt.xlabel("dimension n")
    plt.ylabel("Relative state error")
    plt.legend()
    # plt.savefig("plots/fhn_test2_err" + dt_string + ".png")
    plt.savefig("plots/fhn_test2_err.png")
    plt.show()
