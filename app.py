import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    print("App Started")

    t = np.arange(0., 5., 0.2)
    plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
    plt.show()

    print("App Finished")