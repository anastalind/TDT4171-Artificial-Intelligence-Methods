import numpy as np

def normalize(element):
    return (1/np.sum(element)) * element

def forward(T, O, f, ev):
    return normalize(np.dot(O[ev], np.dot(np.transpose(T), f)))

def backward(T, O, b, ev):
    return np.dot(T, np.dot(O[ev], b))

def filtering(T, O, ev, prior):
    f = [prior]
    t = len(ev)

    for i in range(t):
        f.append(forward(T, O, f[i], ev[i]))
        #print("Forward message ", i, ":", f[i])

    return f

def smoothing(T, O, ev, prior):
    f = filtering(T, O, ev, prior)
    b = np.array([1.0, 1.0])
    s = []

    t = len(ev)

    for i in range(t - 1, -1, -1):
        s.append(normalize(np.multiply(f[i + 1], b)))
        b = backward(T, O, b, ev[i])

        #print("Backward message ", i, ": ", b)
    return s


if __name__ == "__main__":
    # State matrix
    T = np.array([
        [0.7, 0.3],
        [0.3, 0.7]
        ])

    # Observation matrix
    O = np.array([
        [[0.1, 0.0],
         [0.0, 0.8]],

        [[0.9, 0.0],
         [0.0, 0.2]]
    ])
    ev2 = np.array([True, True])
    ev5 = np.array([True, True, False, True, True])
    prior = np.array([0.5, 0.5])

    # Part B
    filtering(T, O, ev2, prior)[len(ev2) - 1]
    filtering(T, O, ev5, prior)

    # Part C
    smoothing(T, O, ev2, prior)[len(ev2) - 1]
    smoothing(T, O, ev5, prior)
