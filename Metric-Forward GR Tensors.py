from sympy import MutableDenseNDimArray, Symbol, eye, Matrix, reshape, S, diff
from sympy import det, tensorcontraction, latex, Eq, sympify, simplify
from dataclasses import dataclass
from IPython.display import display as Idisplay
from IPython.display import Math


@dataclass(frozen=False, order=True)
class Tensor:
    name: str
    symbol: str
    key: str
    components: list
    
    def rank(self):
        return self.key.count('*')
    
    def tensor_full_of(self, x):
        t = x
        for i in range(self.rank()):
            t = [t,] * n
        return MutableDenseNDimArray(t)
    
    def coord_id(self, component_num):
        indices = []
        for i in range(self.rank()):
            ith_index = int(component_num/(n**(self.rank() - i - 1)))
            indices.append(coordinates[ith_index])
            if any(symbol in indices[i] for symbol in GREEK_SYMBOLS):
                indices[i] = '\\' + indices[i] + ' '
            component_num -= ith_index * (n**(self.rank() - i - 1))  
        index_key = self.key
        umth_star = 0
        for i in index_key:
            if i == '*':
                index_key = index_key.replace('*', indices[umth_star], 1)
                umth_star += 1
        return self.symbol + index_key
    
    def print_tensor(self):
        for o in range(len(self.components)):
            if self.components[o] != 0:
                Idisplay(Math(latex(Eq(Symbol(self.coord_id(o)),
                                       self.components[o]))))
        print('\n\n')


def get_dimension():
    global n
    n = input('Enter the number of dimensions:\n')


def check_dimension():
    global n
    if n.isnumeric():
        n = int(n)
        return n
    else:
        print('Number of dimensions needs to be an integer!')
        get_dimension()
        check_dimension()


def get_coordinates():
    global coordinates
    coordinates = []
    for i in range(n):
        coordinates.append(input('Enter coordinate label %d:\n' % i))


def check_coordinates():
    for i in range(len(coordinates)):
        if any(char.isdigit() for char in coordinates[i]):
            print("You shouldn't have numbers in coordinate labels!")
            get_coordinates()
            check_coordinates()


def ask_diagonal():
    global diagonal
    diagonal = input('Is metric diagonal?\n').lower()


def check_diagonal():
    if diagonal not in OK_RESPONSES:
        print('It was a yes or no question...')
        ask_diagonal()
        check_diagonal()


def get_metric():
    global g
    g = eye(n).tolist()
    for i in range(n):
        for j in range(n):
            g[i][j] = '0'
    if diagonal[0] == 'y':
        for i in range(n):
            g[i][i] = input('What is g_[%s%s]?\n' 
                            % (coordinates[i], coordinates[i]))
    else:
        for i in range(n):
            for j in range(i, n):
                g[i][j] = input('What is g_[%s%s]?\n' 
                                % (coordinates[i], coordinates[j]))


def format_metric():
    global g
    for i in range(n):
        for j in range(i, n):
            g[i][j] = g[i][j].replace('^', '**')
            for k in range(len(g[i][j])-1):
                if g[i][j][k].isnumeric() and (g[i][j][k+1].isalpha() or 
                                               g[i][j][k+1] == '('):
                    g[i][j] = g[i][j][:k+1] + '*' + g[i][j][k+1:]
            g[j][i] = g[i][j]
    g = Matrix(g)


def check_metric():
    if g.det() == 0:
        print('\nMetric is singular, try again!\n')
        get_metric()
        format_metric()
        check_metric()


def compile_metric():
    get_dimension()
    check_dimension()
    get_coordinates()
    check_coordinates()
    ask_diagonal()
    check_diagonal()
    get_metric()
    format_metric()
    check_metric()


def assign(instance, tensor_var_name):
    instance.components = tensor_var_name.reshape(
        len(tensor_var_name)).tolist()


metric = Tensor('metric tensor', 'g', '_**', [])
metric_inv = Tensor('inverse of metric tensor', 'g', '__**', [])
metric_d = Tensor('partial derivative of metric tensor', 'g', '_**,*', [])
Christoffel = Tensor('Christoffel symbol - 2nd kind', 'Gamma', '__*_**', [])
Christoffel_d = Tensor('partial derivative of Christoffel symbol',
                       'Gamma', '__*_**,*', [])
Riemann = Tensor('Riemann curvature tensor', 'R', '__*_***', [])
Ricci = Tensor('Ricci curvature tensor', 'R', '_**', [])
Einstein = Tensor('Einstein tensor', 'G', '_**', [])
Einstein_alt = Tensor('Einstein tensor', 'G', '__*_*', [])

GREEK_SYMBOLS = ['alpha', 'beta', 'gamma', 'Gamma', 'delta', 'Delta',
                 'epsilon', 'varepsilon', 'zeta', 'eta', 'theta', 'vartheta',
                 'Theta', 'iota', 'kappa', 'lambda', 'Lambda', 'mu', 'nu',
                 'xi', 'Xi', 'pi', 'Pi', 'rho', 'varrho', 'sigma', 'Sigma',
                 'tau', 'upsilon', 'Upsilon', 'phi', 'varphi', 'Phi', 'chi',
                 'psi', 'Psi', 'omega', 'Omega']
OK_RESPONSES = ['y', 'yes', 'n', 'no']

if __name__ == '__main__':
    compile_metric()
    
    # calculate everything:
    # inverse metric:
    g_inv = MutableDenseNDimArray(g.inv())
    assign(metric_inv, g_inv)
    g = MutableDenseNDimArray(g)
    assign(metric, g)
    # first derivatives of metric components:
    g_d = metric_d.tensor_full_of(0)
    for i in range(n):
        for j in range(i):
            for d in range(n):
                g_d[i, j, d] = g_d[j, i, d]
        for j in range(i, n):
            for d in range(n):
                g_d[i, j, d] = diff(g[i, j], coordinates[d])
    assign(metric_d, g_d)
    # Christoffel symbols for Levi-Civita connection (Gam^i_jk):
    Gamma = Christoffel.tensor_full_of(0)
    for i in range(n):
        for j in range(n):
            for k in range(j):
                Gamma[i, j, k] = Gamma[i, k, j]
            for k in range(j, n):
                for l in range(n):
                    Gamma[i, j, k] += S(1)/2 * g_inv[i, l] * (
                        -g_d[j, k, l] + g_d[k, l, j] + g_d[l, j, k]
                        )
    assign(Christoffel, Gamma)
    # first derivatives of Christoffel symbols (Gam^i_jk,d):
    Gamma_d = Christoffel_d.tensor_full_of(0)
    for i in range(n):
        for j in range(n):
            for k in range(j):
                for d in range(n):
                    Gamma_d[i, j, k, d] = Gamma_d[i, k, j, d]
            for k in range(j, n):
                for d in range(n):
                    Gamma_d[i, j, k, d] = simplify(diff(Gamma[i, j, k],
                                                        coordinates[d]))
    assign(Christoffel_d, Gamma_d)
    # Riemann curvature tensor (R^i_jkl):
    Rie = Riemann.tensor_full_of(0)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(k):
                    Rie[i, j, k, l] = -Rie[i, j, l, k]
                for l in range(k, n):
                    Rie[i, j, k, l] = Gamma_d[i, j, l, k] - Gamma_d[i, j, k, l]
                    for h in range(n):
                        Rie[i, j, k, l] += (Gamma[h, j, l] * Gamma[i, h, k]
                                        - Gamma[h, j, k] * Gamma[i, h, l])
                        Rie[i, j, k, l] = simplify(Rie[i, j, k, l])
    assign(Riemann, Rie)
    # Ricci curvature tensor (R_jl):
    Ric = simplify(tensorcontraction(Rie, (0, 2)))
    assign(Ricci, Ric)
    # Ricci curvature scalar:
    R = 0
    for i in range(n):
        for j in range(n):
            R += g_inv[i, j] * Ric[i, j]
    R = simplify(R)
    # Einstein tensor (G_ij):
    G = Einstein.tensor_full_of(0)
    for i in range(n):
        for j in range(i):
            G[i, j] = G[j, i]
        for j in range(i, n):
            G[i, j] = simplify(Ric[i, j] - S(1)/2 * R * g[i, j])
    assign(Einstein, G)
    # G^i_j:
    G_alt = Einstein_alt.tensor_full_of(0)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                G_alt[i, j] += g_inv[i, k] * G[k, j]
            G_alt[i, j] = simplify(G_alt[i, j])
    assign(Einstein_alt, G_alt)

    # print it all
    print()
    metric.print_tensor()
    metric_inv.print_tensor()
    metric_d.print_tensor()
    Christoffel.print_tensor()
    Christoffel_d.print_tensor()
    Riemann.print_tensor()
    Ricci.print_tensor()
    if R != 0:
        Idisplay(Math(latex(Eq(Symbol('R'), R))))
        print('\n\n')
    Einstein.print_tensor()
    Einstein_alt.print_tensor()
