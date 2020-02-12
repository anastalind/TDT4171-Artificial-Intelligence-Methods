import numpy as np

# Function that normalizes
def normalize(element):
    return (1/np.sum(element)) * element

# Function that calculates equation 15.12
def forward(T, O, f, ev):
    return normalize(np.dot(O[ev], np.dot(np.transpose(T), f)))

# Function that calculates equation 15.13
def backward(T, O, b, ev):
    return np.dot(T, np.dot(O[ev], b))

# Function that filters using the forward algorithm
def filtering(T, O, ev):
    # Initialising f
    f = [np.array([0.5, 0.5])]

    # Number of time steps
    t = len(ev)

    # Computing equation 15.12 for all time steps
    for i in range(t):
        f.append(forward(T, O, f[i], ev[i]))
        #print("Forward message ", i, ":", f[i])

    return f

# Function that smoothes using the forward-backward algorithm
def smoothing(T, O, ev):
    # Forward step of algorithm
    f = filtering(T, O, ev)

    # Initialising b to ones
    b = np.array([1.0, 1.0])

    s = []

    # Number of time steps
    t = len(ev)

    # Computing equation 15.13 for all time steps and multiplying forward and backward steps
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

    # Vector with evidences for day 1-2 and day 1-5
    ev2 = np.array([True, True])
    ev5 = np.array([True, True, False, True, True])

    # Part B
    filtering(T, O, ev2)
    filtering(T, O, ev5)

    # Part C
    smoothing(T, O, ev2)
    smoothing(T, O, ev5)
