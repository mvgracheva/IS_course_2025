import numpy as np

def uniform_intervals(a, b, n):
    return np.linspace(a, b, n)
def test1():
    assert np.allclose(uniform_intervals(-1.2, 2.4, 7), np.array([-1.2, -0.6,  0. ,  0.6,  1.2,  1.8,  2.4]))
test1()
print("1 лайк")
    

def cyclic123_array(n): 
    array_1 = np.array([1, 2, 3])
    return np.tile(array_1, n)
def test2():
    assert np.allclose(cyclic123_array(4), np.array([1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3])) 
test2()
print("2 лайк")


def first_n_odd_number(n):
    return 2 * np.arange(n) + 1
def test3():
    assert np.allclose(first_n_odd_number(3), np.array([1, 3, 5]))
test3()
print("3 лайк")


def zeros_array_with_border(n):
    a = np.zeros((n, n))
    a[0, :] = 1
    a[-1, :] = 1
    a[:, 0] = 1
    a[:, -1] = 1
    return a
def test4():
    assert np.allclose(zeros_array_with_border(4), np.array([[1., 1., 1., 1.],
                                                             [1., 0., 0., 1.],
                                                             [1., 0., 0., 1.],
                                                             [1., 1., 1., 1.]]))
test4()
print("4 лайк")


def chess_board(n):
    base = np.array([[0, 1], [1, 0]])
    num_tiles = (n + 1) // 2
    tiled = np.tile(base, (num_tiles, num_tiles))  
    return tiled[:n, :n]  
def test5():
    expected = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    assert np.allclose(chess_board(3), expected)
test5()
print("5 лайк")


def matrix_with_sum_index(n):
    """6. Создаёт 𝑛 × 𝑛 матрицу с (𝑖,𝑗)-элементами равным 𝑖+𝑗."""
    rows = np.arange(n).reshape(n, 1)
    cols = np.arange(n).reshape(1, n)
    return rows + cols
def test6():
    assert np.allclose(matrix_with_sum_index(3), np.array([[0, 1, 2],
                                                           [1, 2, 3],
                                                           [2, 3, 4]]))
test6()
print("6 лайк")


def cos_sin_as_two_rows(a, b, dx):
    x = np.arange(a, b, dx)
    cos_x = np.cos(x)
    sin_x = np.sin(x)
    return np.vstack((cos_x, sin_x))
def test7():
    assert np.allclose(cos_sin_as_two_rows(0, 1, 0.25), np.array([[1.        , 0.96891242, 0.87758256, 0.73168887],
                                                                  [0.        , 0.24740396, 0.47942554, 0.68163876]]))
test7()
print("7 лайк")


def compute_mean_rowssum_columnssum(A):
    mean = np.mean(A)
    rows_sum = np.sum(A, axis=1)
    columns_sum = np.sum(A, axis=0)
    return mean, rows_sum, columns_sum
def test8():
    np.random.seed(42)
    A = np.random.rand(3, 5)
    mean, rows_sum, columns_sum = compute_mean_rowssum_columnssum(A)
    assert np.abs(mean - 0.49456456164468965) < 1e-12
    assert np.allclose(rows_sum, np.sum(A, axis=1))
    assert np.allclose(columns_sum, np.sum(A, axis=0))
test8()
print("8 лайк")


def sort_array_by_column(A, j):
    return A[A[:, j].argsort()]
def test9():
    np.random.seed(42)
    A = np.random.rand(5, 5)
    assert np.allclose(sort_array_by_column(A, 1), np.array([[0.15599452, 0.05808361, 0.86617615, 0.60111501, 0.70807258],
                                                             [0.61185289, 0.13949386, 0.29214465, 0.36636184, 0.45606998],
                                                             [0.18340451, 0.30424224, 0.52475643, 0.43194502, 0.29122914],
                                                             [0.37454012, 0.95071431, 0.73199394, 0.59865848, 0.15601864],
                                                             [0.02058449, 0.96990985, 0.83244264, 0.21233911, 0.18182497]]))
test9()
print("9 лайк")


def compute_integral(a, b, f, dx, method):
    """10. Считает определённый интеграл функции f на отрезке [a, b] с шагом dx 3-мя методами:  
    method == 'rectangular' - методом прямоугольника   
    method == 'trapezoidal' - методом трапеций   
    method == 'simpson' - методом Симпсона  
    """
    x = np.arange(a, b, dx)
    if method == 'rectangular':
       return np.sum(f(x) * dx)
    elif method == 'trapezoidal':
       return (dx/2) * (f(a) + 2*np.sum(f(x[1:])) + f(b-dx))
    elif method == 'simpson':
        n = len(x)
        if n % 2 == 0:
            x = x[:-1] 
            n = len(x)
        h = (b - a) / (n - 1) # 
        return (h/3) * (f(a) + 4*np.sum(f(x[1::2])) + 2*np.sum(f(x[2:-1:2])) + f(b-dx))

def test10():
    f1 = lambda x: (x**2 + 3) / (x - 2)
    assert np.allclose(compute_integral(3, 4, f1, 0.001, method="rectangular"), 10.352030263919616, rtol=0.01)
    assert np.allclose(compute_integral(3, 4, f1, 0.001, method="trapezoidal"), 10.352030263919616, rtol=0.01)
    assert np.allclose(compute_integral(3, 4, f1, 0.001, method="simpson"), 10.352030263919616, rtol=0.001)

    f2 = lambda x: np.cos(x)**3
    assert np.allclose(compute_integral(0, np.pi/2, f2, 0.001, method="rectangular"), 2/3, rtol=0.01)
    assert np.allclose(compute_integral(0, np.pi/2, f2, 0.001, method="trapezoidal"), 2/3, rtol=0.01)
    assert np.allclose(compute_integral(0, np.pi/2, f2, 0.001, method="simpson"), 2/3, rtol=0.001)

test10()
print("10 лайк") 
    
