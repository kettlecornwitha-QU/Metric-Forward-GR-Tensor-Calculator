from dataclasses import dataclass
from itertools import product
from IPython import get_ipython
from IPython.display import Math, display
from sympy import (
    MutableDenseNDimArray, Symbol, eye, Matrix, reshape, S,
    diff, det, tensorcontraction, latex, Eq, simplify, Expr
)


def is_notebook() -> bool:
    if get_ipython is not None:
        return True


def disp_eq(symbol: str, expression: Expr) -> None:
    raw_latex = latex(Eq(Symbol(symbol), expression))
    if is_notebook():
        display(Math(raw_latex))
    else:
        print(raw_latex)


def y_n_question(latex_Q: str, str_Q: str) -> bool:
    if is_notebook() and latex_Q is not None:
        display(Math(latex_Q))
        prompt = ''
    else:
        prompt = str_Q
    while True:
        response = input(prompt).lower()
        if response[0] in ['y', 'n']:
            return response.startswith('y')
        print('It was a yes or no question...')


@dataclass
class Coordinate:
    index: int
    label: str
    
    def __post_init__(self) -> None:
        self.symbol = Symbol(self.label)
        self.latex = latex(self.symbol)
    
    @classmethod
    def from_stdin(cls, index: int) -> 'Coordinate':
        while True:
            label = input('Enter coordinate label %d:  ' % index)
            if not any(char.isdigit() for char in label):
                return cls(index, label)
            print("You shouldn't have numbers in coordinate labels!")
    
    def __str__(self) -> str:
        return self.label


@dataclass
class Basis:
    index: int
    use: list
    
    @staticmethod
    def basis_prompt(index: int, coord: Coordinate) -> str:
        if is_notebook():
            display(Math(r'\text{What is } (\mathbf{e}_{%d})^{%s}~~?' %
                         (index, coord.latex)))
            prompt = ''
        else:
            prompt = 'What is (e_[%d])^[%s]?\n' % (index, coord)
        return input(prompt)
    
    @classmethod
    def from_stdin(cls, index: int, coords: Coordinate) -> 'Basis':
        basis_vector = [cls.basis_prompt(index, coord) for coord in coords]
        return cls(index, basis_vector)


@dataclass
class Tensor:
    name: str
    symbol: str
    key: str
    use: MutableDenseNDimArray
    
    def __post_init__(self) -> None:
        self.rank = self.key.count('*')
    
    def print_tensor(
            self, coords: Coordinate, using_coord_basis: bool) -> None:
        index = [coords[i].label for i in range(len(coords))]
        if not using_coord_basis:
            index = [f'{i}' for i in range(len(coords))]
        for i in product(range(len(coords)), repeat=self.rank):
            if self.use[i] == 0:
                continue
            index_key = self.key.replace('*', '%s') % tuple(index[j]
                                                            for j in i)
            disp_eq((self.symbol + index_key), self.use[i])
        if any(self.use.reshape(len(self.use)).tolist()):
            print('\n\n')


class Sys:
    def __init__(
            self, coords: list, basis: Matrix, g_m: Matrix,
            using_coord_basis: bool) -> None:
        self.n = len(coords)
        self.coords = coords
        self.basis = basis
        self.using_coord_basis = using_coord_basis
        self.dual_basis = basis.inv()
        self.CB_g = MutableDenseNDimArray(g_m)  # CB = coordinate basis
        self.CB_g_inv = simplify(MutableDenseNDimArray(g_m.inv()))
        self.CB_g_d = self.calculate_CB_g_d()
        self.CB_Gamma = self.calculate_CB_Gamma()
        self.CB_Gamma_d = self.calculate_CB_Gamma_d()
        self.CB_Rie = self.calculate_CB_Rie()
        self.CB_Ric = self.calculate_CB_Ric()
        self.R = self.calculate_R()
        self.CB_G = self.calculate_CB_G()
        self.CB_G_alt = self.calculate_CB_G_alt()
        self.g = Tensor('metric tensor', 'g', '_*_*',
                        MutableDenseNDimArray(g_m))
        self.g_inv = Tensor('inverse of metric tensor', 'g', '^*^*',
                            simplify(MutableDenseNDimArray(g_m.inv())))
        self.g_d = Tensor('partial derivative of metric tensor', 'g',
                          '_*_*_,_*', self.calculate_g_d())
        self.Gamma = Tensor('Christoffel symbol - 2nd kind', 'Gamma',
                            '^*_*_*', self.calculate_Gamma())
        self.Gamma_d = Tensor('partial derivative of Christoffel symbol',
                              'Gamma', '^*_*_*_,_*', self.calculate_Gamma_d())
        self.Rie = Tensor('Riemann curvature tensor', 'R', '^*_*_*_*',
                          self.calculate_Rie())
        self.Ric = Tensor('Ricci curvature tensor', 'R', '_*_*',
                          self.calculate_Ric())
        self.G = Tensor('Einstein tensor', 'G', '_*_*', self.calculate_G())
        self.G_alt = Tensor('Einstein tensor', 'G', '^*_*',
                            self.calculate_G_alt())
        
    @staticmethod
    def ask_dim() -> int:
        while True:
            n = input('Enter the number of dimensions:  ')
            if n.isnumeric():
                return int(n)
            print('Dimension needs to be a positive integer!')

    @staticmethod
    def metric_prompt(i: int or Coordinate, j: int or Coordinate) -> str:
        if isinstance(i, int):
            index_1, index_2 = f'{i}', f'{j}'
        else:
            index_1, index_2 = i.latex, j.latex
        if is_notebook():
            display(Math(r'\text{What is } g_{' + index_1 + index_2 + '}?  '))
            prompt = ''
        else:
            prompt = 'What is g_[%s %s]?  ' % (i, j)
        return input(prompt)
    
    @classmethod
    def ask_metric(
            cls, coords: Coordinate, using_coord_basis: bool,
            is_diagonal: bool) -> list:
        g_m = eye(len(coords)).tolist()
        if is_diagonal:
            indices = enumerate([i for i in range(len(coords))])
            if using_coord_basis:
                indices = enumerate(coords)
            for i, j in indices:
                    g_m[i][i] = cls.metric_prompt(j, j)
        else:
            for i in range(len(coords)):
                for j in range(i, len(coords)):
                    index_1, index_2 = i, j
                    if using_coord_basis:
                        index_1, index_2 = coords[i], coords[j]
                    g_m[i][j] = cls.metric_prompt(index_1, index_2)
                    g_m[j][i] = g_m[i][j]
        return g_m
    
    @classmethod
    def metric_checked(
            cls, coords: Coordinate, using_coord_basis: bool,
            is_diagonal: bool) -> Matrix:
        while True:
            g_m = Matrix(cls.ask_metric(coords, using_coord_basis,
                                           is_diagonal))
            if g_m.det() != 0:
                return g_m
            print('\nMetric is singular, try again!\n')
    
    @staticmethod
    def using_coord_basis() -> bool:
        return y_n_question(
            r'~\text{Do you want to use the coordinate basis? (y/n)}\
            \newline\text{i.e. } (\mathbf{e}_\mu)^\nu = \delta_\mu^\nu',
            'Do you want to use the coordinate basis? (y/n)\n'
        )
    
    @staticmethod
    def is_pseudo_riemannian() -> bool:
        return y_n_question(
            r'~\text{Is manifold pseudo-Riemannian? (y/n)}',
            'Is manifold pseudo-Riemannian? (y/n)\n'
        )
    
    @staticmethod
    def using_orthonormal(is_pseudo_riemannian: bool) -> bool:
        latex_metric = r'\delta'
        if is_pseudo_riemannian:
            latex_metric = r'\eta'
        return y_n_question(
            r'~\text{Is basis orthonormal?}\newline\
            \text{i.e. } g_{\alpha \beta} (\mathbf{e}_\mu)^\alpha \
            (\mathbf{e}_\nu)^\beta = %s_{\mu \nu}' % latex_metric,
            'Is basis orthonormal? (y/n)\n'
        )
    
    @classmethod
    def metric_from_basis(
            cls, n: int, basis: Matrix, using_orthonormal: bool,
            is_pseudo_riemannian: bool, using_coord_basis: bool,
            coords: Coordinate, is_diagonal: bool) -> Matrix:
        M = Matrix([
            [k*l for k, l in product(i, j)]
            for i, j in product(basis.tolist(), repeat=2)
        ])
        if using_orthonormal:
            v = Matrix([int(i % (n+1) == 0) for i in range(n**2)])
            if is_pseudo_riemannian:
                v[0] = -1
        else:
            v = cls.metric_checked(coords, using_coord_basis,
                                   is_diagonal).reshape(n**2, 1)
        return M.LUsolve(v).reshape(n, n)

    @classmethod
    def from_stdin(cls) -> 'Sys':
        coords = [Coordinate.from_stdin(i) for i in range(cls.ask_dim())]
        n = len(coords)
        using_coord_basis = cls.using_coord_basis()
        if using_coord_basis:
            basis = eye(n)
            is_diagonal = y_n_question(r'\text{Is metric diagonal? (y/n)}',
                                       'Is metric diagonal? (y/n)  ')
            g_m = cls.metric_checked(coords, using_coord_basis, is_diagonal)
            using_orthonormal = False
        else:
            basis = Matrix([Basis.from_stdin(i, coords).use
                            for i in range(n)]).T
            is_pseudo_riemannian = cls.is_pseudo_riemannian()
            using_orthonormal = cls.using_orthonormal(is_pseudo_riemannian)
            is_diagonal = True
            if not using_orthonormal:
                is_diagonal = y_n_question(
                    r'~g_{ij} = g_{\mu \nu} (\mathbf{e}_i)^\mu\
                    (\mathbf{e}_j)^\nu\newline\
                    \text{is } g_{ij} \text{ diagonal?}',
                    'Is the basis metric diagonal? (y/n)  ')
            g_m = cls.metric_from_basis(
                n, basis, using_orthonormal, is_pseudo_riemannian, 
                using_coord_basis, coords, is_diagonal
            )
        return cls(coords, basis, g_m, using_coord_basis)
    
    def calculate_CB_g_d(self) -> MutableDenseNDimArray:
        g_d = MutableDenseNDimArray.zeros(self.n, self.n, self.n)
        for i, j, k in product(range(self.n), repeat=3):
            g_d[i, j, k] = diff(self.CB_g[i, j], self.coords[k].symbol)
        return g_d
    
    def calculate_g_d(self) -> MutableDenseNDimArray:
        if self.using_coord_basis:
            return self.CB_g_d
        return MutableDenseNDimArray.zeros(self.n, self.n, self.n)
        
    def calculate_CB_Gamma(self) -> MutableDenseNDimArray:
        g_d = self.CB_g_d
        Gamma = MutableDenseNDimArray.zeros(self.n, self.n, self.n)
        for i, j, k, l in product(range(self.n), repeat=4):
            Gamma[i, j, k] += S(1)/2 * self.CB_g_inv[i, l] * (
                g_d[k, l, j] + g_d[l, j, k] - g_d[j, k, l]
            )
        return Gamma
    
    def calculate_Gamma(self) -> MutableDenseNDimArray:
        if self.using_coord_basis:
            return self.CB_Gamma
        return MutableDenseNDimArray.zeros(self.n, self.n, self.n)

    def calculate_CB_Gamma_d(self) -> MutableDenseNDimArray:
        Gamma_d = MutableDenseNDimArray.zeros(self.n, self.n, self.n, self.n)
        for i, j, k, l in product(range(self.n), repeat=4):
            Gamma_d[i, j, k, l] = simplify(diff(self.CB_Gamma[i, j, k],
                                                self.coords[l].symbol))
        return Gamma_d

    def calculate_Gamma_d(self) -> MutableDenseNDimArray:
        if self.using_coord_basis:
            return self.CB_Gamma_d
        return MutableDenseNDimArray.zeros(self.n, self.n, self.n, self.n)

    def calculate_CB_Rie(self) -> MutableDenseNDimArray:
        Gamma_d, Gamma = self.CB_Gamma_d, self.CB_Gamma
        Rie = MutableDenseNDimArray.zeros(self.n, self.n, self.n, self.n)
        for i, j, k, l in product(range(self.n), repeat=4):
            Rie[i, j, k, l] = Gamma_d[i, j, l, k] - Gamma_d[i, j, k, l]
            for m in range(self.n):
                Rie[i, j, k, l] += (Gamma[m, j, l] * Gamma[i, m, k]
                                    - Gamma[m, j, k] * Gamma[i, m, l])
            Rie[i, j, k, l] = simplify(Rie[i, j, k, l])
        return Rie
    
    def calculate_Rie(self) -> MutableDenseNDimArray:
        if self.using_coord_basis:
            return self.CB_Rie
        basis, dual_basis = self.basis, self.dual_basis
        Rie = MutableDenseNDimArray.zeros(self.n, self.n, self.n, self.n)
        for i, j, k, l in product(range(self.n), repeat=4):
            for m, o, p, q in product(range(self.n), repeat=4):
                Rie[i, j, k, l] += (self.CB_Rie[m, o, p, q]
                                    * dual_basis[i, m] * basis[o, j]
                                    * basis[p, k] * basis[q, l])
            Rie[i, j, k, l] = simplify(Rie[i, j, k, l])
        return Rie

    def calculate_CB_Ric(self) -> MutableDenseNDimArray:
        return simplify(tensorcontraction(self.CB_Rie, (0, 2)))
    
    def calculate_Ric(self) -> MutableDenseNDimArray:
        if self.using_coord_basis:
            return self.CB_Ric
        basis = self.basis
        Ric = MutableDenseNDimArray.zeros(self.n, self.n)
        for i, j in product(range(self.n), repeat=2):
            for k, l in product(range(self.n), repeat=2):
                Ric[i, j] += self.CB_Ric[k, l] * basis[k, i] * basis[l, j]
            Ric[i, j] = simplify(Ric[i, j])
        return Ric
    
    def calculate_R(self) -> Expr:
        R = 0
        for i in product(range(self.n), repeat=2):
            R += self.CB_g_inv[i] * self.CB_Ric[i]
        return simplify(R)

    def calculate_CB_G(self) -> MutableDenseNDimArray:
        G = MutableDenseNDimArray.zeros(self.n, self.n)
        for i in product(range(self.n), repeat=2):
            G[i] = simplify(self.CB_Ric[i] - S(1)/2 * self.R * self.CB_g[i])
        return G

    def calculate_G(self) -> MutableDenseNDimArray:
        if self.using_coord_basis:
            return self.CB_G
        basis = self.basis
        G = MutableDenseNDimArray.zeros(self.n, self.n)
        for i, j in product(range(self.n), repeat=2):
            for k, l in product(range(self.n), repeat=2):
                G[i, j] += self.CB_G[k, l] * basis[k, i] * basis[l, j]
            G[i, j] = simplify(G[i, j])
        return G
    
    def calculate_CB_G_alt(self) -> MutableDenseNDimArray:
        G_alt = MutableDenseNDimArray.zeros(self.n, self.n)
        for i, j in product(range(self.n), repeat=2):
            for k in range(self.n):
                G_alt[i, j] += self.CB_g_inv[i, k] * self.CB_G[k, j]
            G_alt[i, j] = simplify(G_alt[i, j])
        return G_alt

    def calculate_G_alt(self) -> MutableDenseNDimArray:
        if self.using_coord_basis:
            return self.CB_G_alt
        CB_G_alt = self.CB_G_alt
        basis, dual_basis = self.basis, self.dual_basis
        G_alt = MutableDenseNDimArray.zeros(self.n, self.n)
        for i, j in product(range(self.n), repeat=2):
            for k, l in product(range(self.n), repeat=2):
                G_alt[i, j] += CB_G_alt[k, l] * dual_basis[i, k] * basis[l, j]
            G_alt[i, j] = simplify(G_alt[i, j])
        return G_alt
    
    def print_GR_tensors(self) -> None:
        self.g.print_tensor(self.coords, True)
        self.g_inv.print_tensor(self.coords, True)
        for tensor in (self.g_d, self.Gamma, self.Gamma_d, self.Rie, self.Ric):
            tensor.print_tensor(self.coords, self.using_coord_basis)
        if self.R != 0:
            disp_eq('R', self.R)
            print('\n\n')
        self.G.print_tensor(self.coords, self.using_coord_basis)
        self.G_alt.print_tensor(self.coords, self.using_coord_basis)
    
    @classmethod
    def from_demo(cls) -> 'Sys':
        return cls(
            coords=[Coordinate(i, name) for i, name in enumerate(
                ['t', 'l', 'theta', 'phi']
            )],
            basis=Matrix([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, '1/r(l)', 0],
                [0, 0, 0, '1/(r(l)*sin(theta))']
            ]),
            g_m=Matrix([
                [-1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 'r(l)^2', 0],
                [0, 0, 0, 'r(l)^2*sin(theta)^2']
            ]),
            using_coord_basis=False
        )


if __name__ == '__main__':
    Sys = Sys.from_demo()
    #Sys = Sys.from_stdin()
    Sys.print_GR_tensors()
