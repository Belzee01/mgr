import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    a = np.random.random((224, 224))
    print(a)
    plt.imshow(a, cmap='hot', interpolation='nearest')
    plt.show()
