from dataclasses import dataclass
from itertools import product
from IPython import get_ipython
from IPython.display import Math, display
from sympy import (
    MutableDenseNDimArray, Symbol, eye, Matrix, reshape, S, diff, det,
    tensorcontraction, latex, Eq, simplify, Expr
)


def is_notebook() -> bool:
    if get_ipython is not None:
        return True


def disp_eq(symbol, expression) -> None:
    raw_latex = latex(Eq(Symbol(symbol), expression))
    if is_notebook():
        display(Math(raw_latex))
    else:
        print(raw_latex)


@dataclass
class Coordinate:
    index: int
    label: str
    
    def __post_init__(self) -> None:
        self.symbol = Symbol(self.label)
    
    @classmethod
    def from_stdin(cls, index) -> 'Coordinate':
        while True:
            label = input('Enter coordinate label %d:  ' % index)
            if not any(char.isdigit() for char in label):
                return cls(index, label)
            print("You shouldn't have numbers in coordinate labels!")
    
    @property
    def latex(self) -> str:
        return latex(self.symbol)


@dataclass
class Tensor:
    name: str
    symbol: str
    key: str
    use: MutableDenseNDimArray
    
    @property
    def rank(self) -> int:
        return self.key.count('*')
    
    def print_tensor(self, coords) -> None:
        for i in product(range(len(coords)), repeat=self.rank):
            if self.use[i] == 0:
                continue
            index_key = self.key.replace('*', '%s') % tuple(
                coords[j].label for j in i
            )
            disp_eq((self.symbol + index_key), self.use[i])
        if any(self.use.reshape(len(self.use)).tolist()):
            print('\n\n')


class Sys:
    def __init__(self, coords, g_m) -> None:
        self.coords = coords
        self.g = Tensor('metric tensor', 'g', '_*_*',
                        MutableDenseNDimArray(g_m))
        self.g_inv = Tensor('inverse of metric tensor', 'g', '^*^*',
                            simplify(MutableDenseNDimArray(g_m.inv())))
        self.g_d = Tensor('partial derivative of metric tensor', 'g',
                          '_*_*_,_*', self.calculate_g_d())
        self.Gamma = Tensor('Christoffel symbol - 2nd kind', 'Gamma', '^*_*_*',
                            self.calculate_Gamma())
        self.Gamma_d = Tensor('partial derivative of Christoffel symbol',
                              'Gamma', '^*_*_*_,_*', self.calculate_Gamma_d())
        self.Rie = Tensor('Riemann curvature tensor', 'R', '^*_*_*_*',
                          self.calculate_Rie())
        self.Ric = Tensor('Ricci curvature tensor', 'R', '_*_*',
                          self.calculate_Ric())
        self.R = self.calculate_R()
        self.G = Tensor('Einstein tensor', 'G', '_*_*', self.calculate_G())
        self.G_alt = Tensor('Einstein tensor', 'G', '^*_*',
                            self.calculate_G_alt())
        
    @property
    def n(self) -> int:
        return len(self.coords)

    @staticmethod
    def ask_dim() -> int:
        while True:
            n = input('Enter the number of dimensions:  ')
            if n.isnumeric():
                return int(n)
            print('Dimension needs to be a positive integer!')

    @staticmethod
    def ask_diag() -> bool:
        while True:
            diagonal = input('Is metric diagonal? (y/n)  ').lower()
            if diagonal[0] in ['y', 'n']:
                return diagonal.startswith('y')
            print('It was a yes or no question...')
    
    @staticmethod
    def metric_prompt(i, j) -> str:
        if is_notebook():
            display(Math(r'\text{What is } g_{' + i.latex + j.latex + '}?  '))
            prompt = ''
        else:
            prompt = f'What is g_[{i} {j}]?  '
        return input(prompt)
    
    @classmethod
    def get_metric(cls, coords, is_diagonal) -> list:
        g_m = eye(len(coords)).tolist()
        if is_diagonal:
            for i, coord in enumerate(coords):
                g_m[i][i] = cls.metric_prompt(coord, coord)
        else:
            for i in range(len(coords)):
                for j in range(i, len(coords)):
                    g_m[i][j] = cls.metric_prompt(coords[i], coords[j])
                    g_m[j][i] = g_m[i][j]
        return g_m
    
    @classmethod
    def get_metric_checked(cls, coords, is_diagonal) -> Matrix:
        while True:
            g_m = Matrix(cls.get_metric(coords, is_diagonal))
            if g_m.det() != 0:
                return g_m
            print('\nMetric is singular, try again!\n')
    
    @classmethod
    def from_stdin(cls) -> 'Sys':
        coords = [Coordinate.from_stdin(i) for i in range(cls.ask_dim())]
        is_diagonal = cls.ask_diag()
        return cls(
            coords=coords, g_m=cls.get_metric_checked(coords, is_diagonal)
        )
    
    def calculate_g_d(self) -> MutableDenseNDimArray:
        n, g = self.n, self.g.use
        g_d = MutableDenseNDimArray.zeros(n, n, n)
        for i, j, k in product(range(n), repeat=3):
            g_d[i, j, k] = diff(g[i, j], self.coords[k].symbol)
        return g_d

    def calculate_Gamma(self) -> MutableDenseNDimArray:
        n, g_inv, g_d = self.n, self.g_inv.use, self.g_d.use
        Gamma = MutableDenseNDimArray.zeros(n, n, n)
        for i, j, k, l in product(range(n), repeat=4):
            Gamma[i, j, k] += S(1)/2 * g_inv[i, l] * (
                g_d[k, l, j]
                + g_d[l, j, k]
                - g_d[j, k, l]
                )
        return Gamma

    def calculate_Gamma_d(self) -> MutableDenseNDimArray:
        n, Gamma = self.n, self.Gamma.use
        Gamma_d = MutableDenseNDimArray.zeros(n, n, n, n)
        for i, j, k, l in product(range(n), repeat=4):
            Gamma_d[i, j, k, l] = simplify(
                diff(Gamma[i, j, k], self.coords[l].symbol)
            )
        return Gamma_d

    def calculate_Rie(self) -> MutableDenseNDimArray:
        n, Gamma_d, Gamma = self.n, self.Gamma_d.use, self.Gamma.use
        Rie = MutableDenseNDimArray.zeros(n, n, n, n)
        for i, j, k, l in product(range(n), repeat=4):
            Rie[i, j, k, l] = Gamma_d[i, j, l, k] - Gamma_d[i, j, k, l]
            for m in range(n):
                Rie[i, j, k, l] += (
                    Gamma[m, j, l] * Gamma[i, m, k]
                    - Gamma[m, j, k] * Gamma[i, m, l]
                    )
            Rie[i, j, k, l] = simplify(Rie[i, j, k, l])
        return Rie

    def calculate_Ric(self) -> MutableDenseNDimArray:
        Rie = self.Rie.use
        return simplify(tensorcontraction(Rie, (0, 2)))
    
    def calculate_R(self) -> Expr:
        n, g_inv, Ric = self.n, self.g_inv.use, self.Ric.use
        R = 0
        for i in product(range(n), repeat=2):
            R += g_inv[i] * Ric[i]
        return simplify(R)

    def calculate_G(self) -> MutableDenseNDimArray:
        n, Ric, R, g = self.n, self.Ric.use, self.R, self.g.use
        G = MutableDenseNDimArray.zeros(n, n)
        for i in product(range(n), repeat=2):
            G[i] = simplify(Ric[i] - S(1)/2 * R * g[i])
        return G

    def calculate_G_alt(self) -> MutableDenseNDimArray:
        n, g_inv, G = self.n, self.g_inv.use, self.G.use
        G_alt = MutableDenseNDimArray.zeros(n, n)
        for i, j in product(range(n), repeat=2):
            for k in range(n):
                G_alt[i, j] += simplify(g_inv[i, k] * G[k, j])
            G_alt[i, j] = simplify(G_alt[i, j])
        return G_alt

    def print_GR_tensors(self) -> None:
        for tensor in (
            self.g, self.g_inv, self.g_d, self.Gamma,
            self.Gamma_d, self.Rie, self.Ric
        ):
            tensor.print_tensor(self.coords)
        if self.R != 0:
            disp_eq('R', self.R)
            print('\n\n')
        self.G.print_tensor(self.coords)
        self.G_alt.print_tensor(self.coords)
    
    @classmethod
    def from_demo(cls) -> 'Sys':
        return cls(
        coords=[
            Coordinate(i, name) for i, name in enumerate(
                ['t', 'l', 'theta', 'phi']
            )],
        g_m=Matrix([
            [-1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 'r(l)^2', 0],
            [0, 0, 0, 'r(l)^2*sin(theta)^2']
        ]))


if __name__ == '__main__':
    #Sys = Sys.from_demo()
    Sys = Sys.from_stdin()
    Sys.print_GR_tensors()
