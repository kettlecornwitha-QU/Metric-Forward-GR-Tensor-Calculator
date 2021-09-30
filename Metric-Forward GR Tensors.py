from sympy import MutableDenseNDimArray, Symbol, eye, Matrix, reshape, S, diff
from sympy import det, tensorcontraction, latex, Eq, sympify, simplify
from dataclasses import dataclass, field
from itertools import product
from IPython.display import display as disp
from IPython.display import Math


@dataclass
class Tensor:
    name: str
    symbol: str
    key: str
    use: MutableDenseNDimArray = field(default_factory=MutableDenseNDimArray)
    
    def rank(self):
        return self.key.count('*')
    
    def initialize(self, t=0):
        for i in range(self.rank()):
            t = [t,] * n
        self.use = MutableDenseNDimArray(t)
    
    def print_tensor(self):
        for i in product(range(n), repeat=self.rank()):
            if self.use[i] != 0:
                index_key = self.key
                umth_star = 0
                for k in index_key:
                    if k == '*':
                        index_key = index_key.replace('*',
                                                      coords[i[umth_star]], 1)
                        umth_star += 1
                disp(Math(latex(Eq(Symbol(self.symbol + index_key),
                                   self.use[i]))))
        if any(self.use.reshape(len(self.use)).tolist()):
            print('\n\n')


def get_dimension():
    global n
    n = input('Enter the number of dimensions:  ')


def check_dimension():
    global n
    if n.isnumeric():
        n = int(n)
    else:
        print('Number of dimensions needs to be an integer!')
        get_dimension()
        check_dimension()


def get_coordinates():
    global coordinates, coords
    coordinates = []
    for i in range(n):
        coordinates.append(input('Enter coordinate label %d:  ' % i))
    coords = coordinates[:]
    for j in range(len(coords)):
        if any(symbol in coords[j] for symbol in GREEK_SYMBOLS):
            coords[j] = '\\' + coords[j] + ' '


def check_coordinates():
    for i in range(len(coordinates)):
        if any(char.isdigit() for char in coordinates[i]):
            print("You shouldn't have numbers in coordinate labels!")
            get_coordinates()
            check_coordinates()


def ask_diagonal():
    global diagonal
    diagonal = input('Is metric diagonal?  ').lower()


def check_diagonal():
    if diagonal not in OK_RESPONSES:
        print('It was a yes or no question...')
        ask_diagonal()
        check_diagonal()


def get_metric():
    global g_m
    g_m = eye(n).tolist()
    for i in range(n):
        for j in range(n):
            g_m[i][j] = '0'
    if diagonal[0] == 'y':
        for i in range(n):
            g_m[i][i] = input('What is g_[%s%s]?  '
                              % (coordinates[i], coordinates[i]))
    else:
        for i in range(n):
            for j in range(i, n):
                g_m[i][j] = input('What is g_[%s%s]?  '
                                  % (coordinates[i], coordinates[j]))


def format_metric():
    global g_m
    for i in range(n):
        for j in range(i, n):
            g_m[i][j] = g_m[i][j].replace('^', '**')
            for k in range(len(g_m[i][j])-1):
                if (
                    g_m[i][j][k].isnumeric() and (g_m[i][j][k+1].isalpha() or
                                                  g_m[i][j][k+1] == '(')
                    ) or (g_m[i][j][k] == ')' and g_m[i][j][k+1].isalpha()):
                    g_m[i][j] = g_m[i][j][:k+1] + '*' + g_m[i][j][k+1:]
            g_m[j][i] = g_m[i][j]
    g_m = Matrix(g_m)
    g.use = MutableDenseNDimArray(g_m)


def check_metric():
    if g_m.det() == 0:
        print('\nMetric is singular, try again!\n')
        get_metric()
        format_metric()
        check_metric()


def calculate_g_d():
    g_d.initialize()
    for i in product(range(n), repeat=3):
        g_d.use[i] = diff(g.use[i[:2]], coordinates[i[2]])


def calculate_Gamma():
    Gamma.initialize()
    for i in product(range(n), repeat=3):
        for j in range(n):
            Gamma.use[i] += S(1)/2 * g_inv.use[i[0], j] * (
                g_d.use[i[2], j, i[1]]
                + g_d.use[j, i[1], i[2]]
                - g_d.use[i[1], i[2], j]
                )


def calculate_Gamma_d():
    Gamma_d.initialize()
    for i in product(range(n), repeat=4):
        Gamma_d.use[i] = simplify(diff(Gamma.use[i[:3]], coordinates[i[3]]))


def calculate_Rie():
    Rie.initialize()
    for i in product(range(n), repeat=4):
        Rie.use[i] = Gamma_d.use[i[0], i[1], i[3], i[2]] - Gamma_d.use[i]
        for j in range(n):
            Rie.use[i] += (
                Gamma.use[j, i[1], i[3]] * Gamma.use[i[0], j, i[2]]
                - Gamma.use[j, i[1], i[2]] * Gamma.use[i[0], j, i[3]]
                )
        Rie.use[i] = simplify(Rie.use[i])


def calculate_Ric():
    Ric.use = simplify(tensorcontraction(Rie.use, (0, 2)))


def calculate_R():
    global R
    R = 0
    for i in product(range(n), repeat=2):
        R += g_inv.use[i] * Ric.use[i]
    R = simplify(R)


def calculate_G():
    G.initialize()
    for i in product(range(n), repeat=2):
        G.use[i] = simplify(Ric.use[i] - S(1)/2 * R * g.use[i])


def calculate_G_alt():
    G_alt.initialize()
    for i in product(range(n), repeat=2):
        for j in range(n):
            G_alt.use[i] += g_inv.use[i[0], j] * G.use[j, i[1]]
        G_alt.use[i] = simplify(G_alt.use[i])


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
    g_inv.use = MutableDenseNDimArray(g_m.inv())
    

def calculate_GR_tensors():
    calculate_g_d()
    calculate_Gamma()
    calculate_Gamma_d()
    calculate_Rie()
    calculate_Ric()
    calculate_R()
    calculate_G()
    calculate_G_alt()


def print_GR_tensors():
    g.print_tensor()
    g_inv.print_tensor()
    g_d.print_tensor()
    Gamma.print_tensor()
    Gamma_d.print_tensor()
    Rie.print_tensor()
    Ric.print_tensor()
    if R != 0:
        disp(Math(latex(Eq(Symbol('R'), R))))
        print('\n\n')
    G.print_tensor()
    G_alt.print_tensor()


g = Tensor('metric tensor', 'g', '_**')
g_inv = Tensor('inverse of metric tensor', 'g', '__**')
g_d = Tensor('partial derivative of metric tensor', 'g', '_**,*')
Gamma = Tensor('Christoffel symbol - 2nd kind', 'Gamma', '__*_**')
Gamma_d = Tensor('partial derivative of Christoffel symbol',
                 'Gamma', '__*_**,*')
Rie = Tensor('Riemann curvature tensor', 'R', '__*_***')
Ric = Tensor('Ricci curvature tensor', 'R', '_**')
G = Tensor('Einstein tensor', 'G', '_**')
G_alt = Tensor('Einstein tensor', 'G', '__*_*')

GREEK_SYMBOLS = ['alpha', 'beta', 'gamma', 'Gamma', 'delta', 'Delta',
                 'epsilon', 'varepsilon', 'zeta', 'eta', 'theta', 'vartheta',
                 'Theta', 'iota', 'kappa', 'lambda', 'Lambda', 'mu', 'nu',
                 'xi', 'Xi', 'pi', 'Pi', 'rho', 'varrho', 'sigma', 'Sigma',
                 'tau', 'upsilon', 'Upsilon', 'phi', 'varphi', 'Phi', 'chi',
                 'psi', 'Psi', 'omega', 'Omega']
OK_RESPONSES = ['y', 'yes', 'n', 'no']

if __name__ == '__main__':
    compile_metric()
    calculate_GR_tensors()
    print_GR_tensors()
