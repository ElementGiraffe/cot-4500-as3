import numpy as np

# takes in function f(t, y), the range [a, b], the number of iterations N, and f(a) = alpha
def eulers_method(func, a, b, N, alpha):

    ans = []

    h = (b-a)/N
    t = a
    w = alpha

    ans.append(eval_fty(func, t, w))

    for i in range(1, N+1):
        w += h*eval_fty(func, t, w)
        t += h
        ans.append(eval_fty(func, t, w))

    #print(ans)
    return w

def eval_fty(f, t, y):
    # print(f)
    # print("t: " + str(t))
    # print("y: " + str(y))
    # print(eval(f))
    return eval(f)

# this is of order four
def runge_kutta(func, a, b, N, alpha):
    ans = []

    h = (b-a)/N
    t = a
    w = alpha

    ans.append(eval_fty(func, t, w))

    for i in range(1, N+1):
        K1 = h*eval_fty(func, t, w)
        K2 = h*eval_fty(func, t + h/2, w + K1/2)
        K3 = h*eval_fty(func, t + h/2, w + K2/2)
        K4 = h*eval_fty(func, t + h, w + K3)

        w += (K1 + 2*K2 + 2*K3 + K4)/6
        t += h

        ans.append(eval_fty(func, t, w))

    # print(ans)

    return w

# performs gaussian elimination and returns an array of the values of x
def gaussian(A):
    n = A.shape[0]

    for i in range(n-1):
        p = -1
        for p_test in range(i, n):
            if A[p_test][i] != 0:
                p = p_test
                break

        if p == -1:
            # no solution was found
            return None

        if p != i:
            # swap rows
            A[[p, i]] = A[[i, p]]

        for j in range(i+1, n):
            m = A[j][i]/A[i][i]
            A[j] = A[j] - m*A[i]

    if A[n-1][n-1] == 0:
        # infinite solutions
        return None

    # print(A)
    x = np.zeros(n)

    x[n-1] = A[n-1][n]/A[n-1][n-1]

    for i in range(n-2, -1, -1):
        x[i] = A[i][n]
        for j in range(i+1, n):
            x[i] -= A[i][j]*x[j]
        x[i] /= A[i][i]

    return x

def LU_factorization(A):
    n = A.shape[0]
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    L[0][0] = 1
    U[0][0] = A[0][0]

    if U[0][0] == 0:
        return None, None

    for j in range(1, n):
        U[0][j] = A[0][j]/L[0][0]
        L[j][0] = A[j][0]/U[0][0]

    for i in range(1, n-1):
        L[i][i] = 1
        U[i][i] = A[i][i]
        for k in range(0, i):
            U[i][i] -= L[i][k] * U[k][i]

        if U[i][i] == 0:
            return None, None

        for j in range(i+1, n):
            U[i][j] = A[i][j]
            L[j][i] = A[j][i]

            for k in range(0, i):
                U[i][j] -= L[i][k] * U[k][j]
                L[j][i] -= L[j][k] * U[k][i]

            U[i][j] /= L[i][i]
            L[j][i] /= U[i][i]

    L[n-1][n-1] = 1
    U[n-1][n-1] = A[n-1][n-1]
    for k in range(0, n-1):
        U[n-1][n-1] -= L[n-1][k] * U[k][n-1]

    return L, U

def is_diag_dom(A):
    n = A.shape[0]

    for i in range(n):
        sum = 0
        for j in range(n):
            if i == j:
                continue
            sum += abs(A[i][j])
        if abs(A[i][i]) < sum:
            return False

    return True


def is_pos_def(A):
    n = A.shape[0]

    # checking that A is symmetric
    for i in range(n):
        for j in range(i+1, n):
            if A[i][j] != A[j][i]:
                return False

    for i in range(n):
        A_sub = A[:i+1, :i+1]

        # print(i)
        # print(A_sub)

        if np.linalg.det(A_sub) <= 0:
            return False

    return True


if __name__ == "__main__":
    np.set_printoptions(precision=7, suppress=True, linewidth=100)

    # PART ONE
    euler_ans = eulers_method("t - y**2", 0, 2, 10, 1)
    print("%.5f" % euler_ans)

    print("")

    # PART TWO
    runge_kutta_ans = runge_kutta("t - y**2", 0, 2, 10, 1)
    print("%.5f" % runge_kutta_ans)

    print("")
    # PART THREE
    A = np.array([[2, -1, 1, 6],
                  [1, 3, 1, 0],
                  [-1, 5, 4, -3]], dtype=np.double)

    sols = gaussian(A)
    print(sols)

    print("")

    #PART FOUR

    A = np.array([[1, 1, 0, 3],
                  [2, 1, -1, 1],
                  [3, -1, -1, 2],
                  [-1, 2, 3, -1]], dtype=np.double)

    print("%.5f" % np.linalg.det(A))

    L, U = LU_factorization(A)

    print("")

    print(L)

    print("")

    print(U)

    #print(np.matmul(L, U))

    print("")

    # PART FIVE

    A = np.array([[9, 0, 5, 2, 1],
                  [3, 9, 1, 2, 1],
                  [0, 1, 7, 2, 3],
                  [4, 2, 3, 12, 2],
                  [3, 2, 4, 0, 8]])

    print(is_diag_dom(A))

    print("")

    # PART SIX

    A = np.array([[2, 2, 1],
                  [2, 3, 0],
                  [1, 0, 2]])

    print(is_pos_def(A))
