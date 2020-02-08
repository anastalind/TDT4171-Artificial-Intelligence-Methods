import numpy as np

# Function which implements filtering using the FORWARD operation
# as described in equations 15.5 and 15.12 in 'Artifical Intelligence - A Modern Approach'
# @param numpy array T - Conditional probability matrix.
# @param numpy array evidences - Array of evidence of size t.
# @param int t - Number of time steps.
# @return numpy array f - The probabilty of rain on day t.
def forward(T, evidences, t):
    # Initial value for f_1:0
    if t == 0:
        return np.array([0.5, 0.5])

    # Finding the right observation conditional probabilty matrix based on evidence
    if evidences[t - 1] == True:
        O = np.array([[0.9, 0.0], [0.0, 0.2]])
    else:
        O = np.array([[0.1, 0.0], [0.0, 0.8]])

    # Calculating f_1:t recursively, as the forward function returns f_1:t-1
    f = np.dot(O, np.dot(np.transpose(T), forward(T, evidences, t-1)))

    # Normalization constant
    alpha = 1 / np.sum(f)

    f = alpha * f

    return f


if __name__ == "__main__":
    T = np.array([[0.7, 0.3], [0.3, 0.7]])

    evidences_a = np.array([True, True])
    forward(T, evidences_a, 2)

    evidences_b = np.array([True, True, False, True, True])
    forward(T, evidences_b, 5)
