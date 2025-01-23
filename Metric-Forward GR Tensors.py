#______________________________________________________________________________
from dataclasses import dataclass
from typing import List, Dict
from itertools import product
from IPython import get_ipython
from IPython.display import Math, display
from sympy import (
    MutableDenseNDimArray as MDMA, Symbol, eye, Matrix, reshape, S, diff,
    det, tensorcontraction as contract, tensorproduct as tensprod, latex,
    Eq, simplify, Expr, sympify
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


def var_rank_array(dim: int, rank: int) -> MDMA:
    x = 0
    for i in range(rank):
        x = [x,] * dim
    return MDMA(x)


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
            print("Don't put numbers in your coordinate labels, you goon")
    
    def __str__(self) -> str:
        return self.label


class Basis:
    def __init__(self, basis_set: list['Basis_Vector']) -> None:
        self.basis_set = basis_set
        self.dual_basis = self.calc_dual_basis()
        
    def calc_dual_basis(self) -> list['Basis_One_Form']:
        dim = len(self.basis_set)
        basis = [self.basis_set[i].use_str for i in range(dim)]
        dual_basis = []
        for i, dual_vector in enumerate(Matrix(basis).T.inv().tolist()):
            coords = self.basis_set[i].coords
            dual_basis.append(
                Basis_One_Form(
                    i, MDMA(
                        [sympify(component) for component in dual_vector]
                    ), coords
                )
            )
        return dual_basis


class GR_Array:
    def __init__(
        self, name: str, symbol: str, key: str, use: MDMA, 
        coords: list[Coordinate], alt_basis: Basis=None
    ) -> None:
        self.name = name
        self.symbol = symbol
        self.key = key
        self.use = use
        self.coords = coords
        self.n = len(coords)
    
    @property
    def rank(self) -> int:
        return self.key.count('*')
    
    def finalize_array(self, using_alt_basis: bool = False) -> dict[str, str]:
        index = [coord.label for coord in self.coords]
        base_key, use = self.key, self.use
        if isinstance(self, Tensor):
            base_key = self.disp_key
            if using_alt_basis:
                index = [f'{i}' for i in range(self.n)]
                use = self.alt_basis_use
        non_zero_components = {}
        for i in product(range(self.n), repeat=self.rank):
            if use[i] == 0:
                continue
            index_key = base_key.replace('*', '%s') % tuple(index[j] 
                                                            for j in i)
            non_zero_components[
                (latex(Symbol(self.symbol + index_key)))
            ] = latex(simplify(use[i]))
        return non_zero_components

    def print_array(self, using_alt_basis: bool = False) -> None:
        non_zero_components = self.finalize_array(using_alt_basis)
        for symbol, expression in non_zero_components.items():
            if is_notebook():
                display(Math(symbol + '=' + expression))
            else:
                print(symbol + '=' + expression)
        if any(non_zero_components):
            print('\n\n')

    def partial_derivative(self) -> 'GR_Array':
        new_key = self.key + '_,_*'
        new_use = var_rank_array(self.n, self.rank+1)
        for i in product(range(self.n), repeat=self.rank+1):
            new_use[i] = diff(self.use[i[:-1]], self.coords[i[-1]].symbol)
        return GR_Array(self.name, self.symbol, new_key, new_use, self.coords)


class Tensor(GR_Array):
    def __init__(
        self, name: str, symbol: str, key: str, 
        use: MDMA, coords: List[Coordinate], alt_basis: Basis = None
    ) -> None:
        super().__init__(name, symbol, key, use, coords)
        self.disp_key = self.generate_disp_key()
        self.alt_basis_use = self.change_basis(alt_basis)
        
    def raise_index(
        self, i: int, g_inv: MDMA, alt_basis: Basis
    ) -> 'Tensor':
        if self.key[i*2] != '_':
            if self.key[i*2] == '^':
                raise ValueError('That index is already raised')
            raise ValueError('Your key is not properly formatted')
        new_key = self.key[:i*2] + '^' + self.key[i*2+1:]
        new_use = contract(tensprod(self.use, g_inv), (i, self.rank))
        return Tensor(
            self.name, self.symbol, new_key, 
            new_use, self.coords, alt_basis
        )
    
    def lower_index(self, i: int, g: MDMA, alt_basis: Basis) -> 'Tensor':
        if self.key[i*2] != '^':
            if self.key[i*2] == '_':
                raise ValueError('That index is already lowered')
            raise ValueError('Your key is not properly formatted')
        new_key = self.key[:i*2] + '_' + self.key[i*2+1:]
        new_use = contract(tensprod(self.use, g), (i, self.rank))
        return Tensor(
            self.name, self.symbol, new_key, 
            new_use, self.coords, alt_basis
        )
    
    def change_basis(self, alt_basis: Basis) -> MDMA:
        if alt_basis is None:
            return None
        dim = len(alt_basis.basis_set)
        alt_tensor = var_rank_array(dim, self.rank)
        key = self.key.replace('*', '')
        for i in product(range(dim), repeat=self.rank):
            mixed_basis = []
            for k, l in enumerate(key):
                if l == '_':
                    mixed_basis.append(alt_basis.basis_set[i[k]].use)
                else:
                    mixed_basis.append(alt_basis.dual_basis[i[k]].use)
            temp = contract(tensprod(self.use, mixed_basis[0]), (0, self.rank))
            for j, vect in enumerate(mixed_basis[1:]):
                temp = contract(tensprod(temp, vect), (0, self.rank-1-j))
            alt_tensor[i] = simplify(temp)
        return alt_tensor

    def generate_disp_key(self) -> str:
        output = []
        prev_char = None
        consecutive_count = 0
        for i in range(0, len(self.key), 2):
            current = self.key[i:i+2]
            if prev_char is not None and current != prev_char:
                if prev_char == '^*' and current == '_*':
                    output.append('_~_~' * consecutive_count)
                elif prev_char == '_*' and current == '^*':
                    output.append('^~^~' * consecutive_count)
                consecutive_count = 0
            output.append(current)
            prev_char = current
            consecutive_count += 1
        return ''.join(output)
    
    def print_tensor(self) -> None:
        self.print_array()
        if self.alt_basis_use is not None:
            self.print_array(using_alt_basis=True)

    def finalize_tensor(self) -> dict[str, Expr]:
        non_zero_components = self.finalize_array()
        if self.alt_basis_use is not None:
            non_zero_components.update(
                self.finalize_array(using_alt_basis=True)
            )
        return non_zero_components


class Metric(Tensor):
    def __init__(
        self, g_m: Matrix, coords: list[Coordinate], alt_basis: Basis
    ) -> None:
        name = 'metric tensor'
        symbol = 'g'
        key = '_*_*'
        use = MDMA(g_m)
        super().__init__(name, symbol, key, use, coords, alt_basis)
        

class Inverse_Metric(Tensor):
    def __init__(
        self, metric: Metric, coords: list[Coordinate], alt_basis: Basis
    ) -> None:
        name = 'inverse metric'
        symbol = 'g'
        key = '^*^*'
        use = simplify(MDMA(Matrix(metric.use).inv()))
        super().__init__(name, symbol, key, use, coords, alt_basis)


class Basis_Vector(Tensor):
    def __init__(
        self, index: int, use_str: list[str], coords: list[Coordinate]
    ) -> None:
        name = f'basis vector {index}'
        symbol = 'e'
        key = '^*'
        use = MDMA([sympify(i) for i in use_str])
        super().__init__(name, symbol, key, use, coords)
        self.index = index
        self.use_str = use_str
        self.latex = r'\mathbf{e}_{%s}' % index
        

class Basis_One_Form(Tensor):
    def __init__(
        self, index: int, use: MDMA, coords: list[Coordinate]
    ) -> None:
        name = f'basis one-form {index}'
        symbol = 'omega'
        key = '_*'
        super().__init__(name, symbol, key, use, coords)
        self.index = index
        self.latex = r'\mathbf{\omega}_{%s}' % index


class Christoffel(GR_Array):
    def __init__(
        self, coords: list[Coordinate], metric: Metric, 
        inverse_metric: Inverse_Metric
    ) -> None:
        name = 'Christoffel symbol - 2nd kind'
        symbol = 'Gamma'
        key = '^*_*_*'
        use = self.calc_christoffel(metric, inverse_metric, coords)
        super().__init__(name, symbol, key, use, coords)

    def calc_christoffel(
        self, metric: Metric, inverse_metric: Inverse_Metric, 
        coords: list[Coordinate]
    ) -> MDMA:
        n = len(coords)
        g_inv, g_d = inverse_metric.use, metric.partial_derivative().use
        Gamma = MDMA.zeros(n, n, n)
        for i, j, k, l in product(range(n), repeat=4):
            Gamma[i, j, k] += S(1)/2 * g_inv[i, l] * (
                g_d[k, l, j] + g_d[l, j, k] - g_d[j, k, l]
            )
        return Gamma


class Riemann(Tensor):
    def __init__(
        self, christoffel: Christoffel, 
        coords: list[Coordinate], alt_basis: Basis
    ) -> None:
        name = 'Riemann curvature tensor'
        symbol = 'R'
        key = '^*_*_*_*'
        use = self.calc_rie(christoffel, coords)
        super().__init__(name, symbol, key, use, coords, alt_basis)

    def calc_rie(
        self, christoffel: Christoffel, coords: list[Coordinate]
    ) -> MDMA:
        n = len(coords)
        Gamma, Gamma_d = christoffel.use, christoffel.partial_derivative().use
        Rie = MDMA.zeros(n, n, n, n)
        for i, j, k, l in product(range(n), repeat=4):
            Rie[i, j, k, l] = Gamma_d[i, j, l, k] - Gamma_d[i, j, k, l]
            for m in range(n):
                Rie[i, j, k, l] += (
                    Gamma[m, j, l] * Gamma[i, m, k]
                    - Gamma[m, j, k] * Gamma[i, m, l]
                )
            Rie[i, j, k, l] = simplify(Rie[i, j, k, l])
        return Rie


class Ricci_Tensor(Tensor):
    def __init__(
        self, rie: Riemann, coords: list[Coordinate], alt_basis: Basis
    ) -> None:
        name = 'Ricci curvature tensor'
        symbol = 'R'
        key = '_*_*'
        use = simplify(contract(rie.use, (0, 2)))
        super().__init__(name, symbol, key, use, coords, alt_basis)


class Ricci_Scalar:
    def __init__(
        self, ric: Ricci_Tensor, g_inv: MDMA, alt_basis: Basis
    ) -> None:
        self.name = 'Ricci scalar'
        self.symbol = 'R'
        self.value = simplify(
            contract(ric.raise_index(0, g_inv, alt_basis).use, (0, 1))
        )


class Einstein(Tensor):
    def __init__(
        self, metric: Metric, ric_t: Ricci_Tensor, ric_s: Ricci_Scalar, 
        coords: list[Coordinate], alt_basis: Basis
    ) -> None:
        name = 'Einstein tensor'
        symbol = 'G'
        key = '_*_*'
        use = self.calc_einstein(metric, ric_t, ric_s, coords)
        super().__init__(name, symbol, key, use, coords, alt_basis)

    def calc_einstein(
        self, metric: Metric, ric_t: Ricci_Tensor, 
        ric_s: Ricci_Scalar, coords: list[Coordinate]
    ) -> MDMA:
        n, g, Ric, R = len(coords), metric.use, ric_t.use, ric_s.value
        G = MDMA.zeros(n, n)
        for i in product(range(n), repeat=2):
            G[i] = simplify(Ric[i] - S(1)/2 * R * g[i])
        return G


class User_Ins:
    def ask_dim(self) -> int:
        while True:
            n = input('Enter the number of dimensions:  ')
            if n.isnumeric():
                return int(n)
            print('Dimension needs to be a positive integer!')

    def using_alt_basis(self) -> bool:
        return y_n_question(
            r'~\text{Do you want to include an alternate basis? (y/n)}',
            'Do you want to include an alternate basis? (y/n)\n'
        )
    
    def alt_bv_prompt(self, index: int, coord: Coordinate) -> str:
        if is_notebook():
            display(Math(
                r'\text{What is } (\mathbf{e}_{%d})^{%s}~~?' %
                (index, coord.latex)
            ))
            prompt = ''
        else:
            prompt = 'What is (e_[%d])^[%s]?\n' % (index, coord.label)
        return input(prompt)
    
    def ask_diagonal(self) -> bool:
        return y_n_question(
            r'\text{Is metric diagonal? (y/n)}',
            'Is metric diagonal? (y/n)  '
        )
    
    def is_pseudo_riemannian(self) -> bool:
        return y_n_question(
            r'~\text{Is manifold pseudo-Riemannian? (y/n)}',
            'Is manifold pseudo-Riemannian? (y/n)\n'
        )
    
    def ask_orthonormal(self, latex_metric: str) -> str:
        return y_n_question(
            r'~\text{Is alternate basis orthonormal?}\newline\
            \text{i.e. } g_{\alpha \beta} (\mathbf{e}_\mu)^\alpha \
            (\mathbf{e}_\nu)^\beta = %s_{\mu \nu}' % latex_metric,
            'Is alternate basis orthonormal? (y/n)\n'
        )
    
    def ask_which_basis(self) -> str:
        return input(
            'Do you prefer to provide the metric components of the\n'
            'coordinate basis or of the alternate basis? (enter \"c\" for\n'
            'coordinate or \"a\" for alternate)\n'
        )
    
    def metric_prompt(self, i: str, j: str) -> str:
        if is_notebook():
            display(Math(r'\text{What is } g_{' + i + j + '}?  '))
            return input('')
        return input('What is g_[%s %s]?  ' % (i, j))


class Run_Calc:
    def __init__(
        self, coords: list, g_m: Matrix, alt_basis: Basis = None
    ) -> None:
        self.n = len(coords)
        self.coords = coords
        self.alt_basis = alt_basis
        self.using_alt_basis = alt_basis is not None
        self.metric = Metric(g_m, coords, alt_basis)
        self.inverse_metric = Inverse_Metric(self.metric, coords, alt_basis)
        self.christoffel = Christoffel(
            coords, self.metric, self.inverse_metric
        )
        self.riemann = Riemann(self.christoffel, coords, alt_basis)
        self.ricci_tensor = Ricci_Tensor(self.riemann, coords, alt_basis)
        self.ricci_scalar = Ricci_Scalar(
            self.ricci_tensor, self.inverse_metric.use, alt_basis
        )
        self.einstein = Einstein(
            self.metric, self.ricci_tensor, 
            self.ricci_scalar, coords, alt_basis
        )
    
    @staticmethod
    def using_orthonormal(is_pseudo_riemannian: bool) -> bool:
        latex_metric = r'\delta'
        if is_pseudo_riemannian:
            latex_metric = r'\eta'
        return User_Ins().ask_orthonormal(latex_metric)
            
    @staticmethod
    def g_cmpnts_in_CB() -> bool:
        while True:
            basis = User_Ins().ask_which_basis()
            if basis[0] in ['c', 'a']:
                return basis.startswith('c')
            print('Try that again...')
    
    @staticmethod
    def formatted_metric_prompt(
        coord_1: Coordinate, coord_2: Coordinate, using_alt_basis: bool
    ) -> str:
        i, j = str(coord_1.index), str(coord_2.index)
        if not using_alt_basis:
            i, j = coord_1.latex, coord_2.latex
            if not is_notebook():
                i, j = coord_1.label, coord_2.label
        return User_Ins().metric_prompt(i,j)
    
    @classmethod
    def ask_metric(
        cls, coords: list[Coordinate], using_alt_basis: bool, is_diagonal: bool
    ) -> list[list[str]]:
        g_m = eye(len(coords)).tolist()
        if is_diagonal:
            for i, coord in enumerate(coords):
                g_m[i][i] = cls.formatted_metric_prompt(
                    coord, coord, using_alt_basis
                )
            return g_m
        for i in range(len(coords)):
            for j in range(i, len(coords)):
                coord_1, coord_2 = coords[i], coords[j]
                g_m[i][j] = cls.formatted_metric_prompt(
                    coord_1, coord_2, using_alt_basis
                )
                g_m[j][i] = g_m[i][j]
        return g_m
    
    @classmethod
    def metric_from_stdin(
        cls, coords: list[Coordinate], using_alt_basis: bool, is_diagonal: bool
    ) -> Matrix:
        while True:
            g_m = Matrix(
                cls.ask_metric(coords, using_alt_basis, is_diagonal)
            )
            if g_m.det() != 0:
                return g_m
            print('\nMetric is singular, try again!\n')
    
    @classmethod
    def metric_from_basis(
        cls, alt_basis: Basis, using_orthonormal: bool, 
        is_pseudo_riemannian: bool, 
        coords: list[Coordinate], is_diagonal: bool
    ) -> Matrix:
        n = len(coords)
        M = Matrix([
            [k*l for k, l in product(BV1.use, BV2.use)]
            for BV1, BV2 in product(alt_basis.basis_set, repeat=2)
        ])
        if using_orthonormal:
            v = Matrix([int(i % (n+1) == 0) for i in range(n**2)])
            if is_pseudo_riemannian:
                v[0] = -1
        else:
            v = cls.metric_from_stdin(
                coords=coords, using_alt_basis=True, is_diagonal=is_diagonal
            ).reshape(n**2, 1)
        return M.LUsolve(v).reshape(n, n)

    @classmethod
    def build_from_CB(cls, coords: list[Coordinate]) -> 'Run_Calc':
        n = len(coords)
        using_alt_basis = False
        is_diagonal = User_Ins().ask_diagonal()
        g_m = cls.metric_from_stdin(coords, using_alt_basis, is_diagonal)
        return cls(coords, g_m)
    
    @classmethod
    def build_from_altB_not_ON(
        cls, coords: list[Coordinate], alt_basis: Basis, 
        is_pseudo_riemannian: bool
    ) -> 'Run_Calc':
        g_cmpnts_in_CB = cls.g_cmpnts_in_CB()
        is_diagonal = User_Ins().ask_diagonal()
        if g_cmpnts_in_CB:
            g_m = cls.metric_from_stdin(
                coords=coords, using_alt_basis=False, is_diagonal=is_diagonal
            )
            return cls(coords, g_m, alt_basis)
        g_m = cls.metric_from_basis(
            alt_basis=alt_basis, using_orthonormal=False, 
            is_pseudo_riemannian=is_pseudo_riemannian, 
            coords=coords, is_diagonal=is_diagonal
        )
        return cls(coords, g_m, alt_basis)
    
    @classmethod
    def build_from_altB_ON(
        cls, coords: list[Coordinate], alt_basis: Basis, 
        is_pseudo_riemannian: bool, is_diagonal: bool
    ) -> 'Run_Calc':
        g_m = cls.metric_from_basis(
            alt_basis=alt_basis, using_orthonormal=True, 
            is_pseudo_riemannian=is_pseudo_riemannian, 
            coords=coords, is_diagonal=is_diagonal
        )
        return cls(coords, g_m, alt_basis)
    
    @classmethod
    def from_stdin(cls) -> 'Run_Calc':
        n = User_Ins().ask_dim()
        coords = [Coordinate.from_stdin(i) for i in range(n)]
        using_alt_basis = User_Ins().using_alt_basis()
        if not using_alt_basis:
            return cls.build_from_CB(coords)
        alt_basis = Basis([
            Basis_Vector(
                i, [User_Ins().alt_bv_prompt(i, coord) for coord in coords], 
                coords
            ) for i in range(n)
        ])
        is_pseudo_riemannian = User_Ins().is_pseudo_riemannian()
        using_orthonormal = cls.using_orthonormal(is_pseudo_riemannian)
        is_diagonal = True
        if not using_orthonormal:
            return cls.build_from_altB_not_ON(
                coords, alt_basis, is_pseudo_riemannian
            )
        return cls.build_from_altB_ON(
            coords, alt_basis, is_pseudo_riemannian, is_diagonal
        )

    def print_GR_tensors(self) -> None:
        self.metric.print_tensor()
        self.inverse_metric.print_tensor()
        for array in (
            self.metric.partial_derivative(), self.christoffel, 
            self.christoffel.partial_derivative()
        ):
            array.print_array()
        self.riemann.print_tensor()
        self.ricci_tensor.print_tensor()
        if self.ricci_scalar.value != 0:
            disp_eq('R', self.ricci_scalar.value)
            print('\n\n')
        mixed_index_G = self.einstein.raise_index(
            0, self.inverse_metric.use, self.alt_basis
        )
        for tensor in (
            self.einstein, mixed_index_G, 
            mixed_index_G.raise_index(
                1, self.inverse_metric.use, self.alt_basis
            )
        ):
            tensor.print_tensor()

    def return_all_GR_tensors(self) -> dict[str, str]:
        metric = self.metric.finalize_tensor()
        inv_metric = self.inverse_metric.finalize_tensor()
        metric_d = self.metric.partial_derivative().finalize_array()
        christoffel = self.christoffel.finalize_array()
        christoffel_d = self.christoffel.partial_derivative().finalize_array()
        riemann = self.riemann.finalize_tensor()
        ricci_t = self.ricci_tensor.finalize_tensor()
        all_tensors = {
            **metric, **inv_metric, **metric_d, **christoffel, 
            **christoffel_d, **riemann, **ricci_t
        }
        if self.ricci_scalar.value != 0:
            all_tensors.update({'R': self.ricci_scalar.value})
        all_tensors.update(self.einstein.finalize_tensor())
        mixed_index_G = self.einstein.raise_index(
            0, self.inverse_metric.use, self.alt_basis
        )
        all_tensors.update(mixed_einstein = mixed_index_G.finalize_tensor())
        all_tensors.update(contravar_einstein = mixed_index_G.raise_index(
            1, self.inverse_metric.use, self.alt_basis
        ).finalize_tensor())
        return all_tensors
    
    @classmethod
    def from_demo_1(cls) -> 'Run_Calc':
        coords = [
            Coordinate(0, 't'), Coordinate(1, 'l'), 
            Coordinate(2, 'theta'), Coordinate(3, 'phi')
        ]
        return cls(
            coords = coords,
            g_m = Matrix([
                [-1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 'r(l)^2', 0],
                [0, 0, 0, 'r(l)^2*sin(theta)^2']
            ]),
            alt_basis = Basis([
                Basis_Vector(0, ['1', '0', '0', '0'], coords),
                Basis_Vector(1, ['0', '1', '0', '0'], coords),
                Basis_Vector(2, ['0', '0', '1/r(l)', '0'], coords),
                Basis_Vector(
                    3, ['0', '0', '0', '1/(r(l)*sin(theta))'], coords
                )
            ])
            
        )

    @classmethod
    def from_demo_2(cls) -> 'Run_Calc':
        coords = [Coordinate(0, 'theta'), Coordinate(1, 'phi')]
        return cls(
            coords = coords,
            g_m = Matrix([[1, 0], [0, 'sin(theta)^2']]),
            alt_basis = Basis([
                Basis_Vector(0, ['1', '0'], coords), 
                Basis_Vector(1, ['0', '1/sin(theta)'], coords)
            ])
        )

    @classmethod
    def from_demo_3(cls) -> 'Run_Calc':
        coords = [
            Coordinate(0, 't'), Coordinate(0, 'r'), 
            Coordinate(0, 'theta'), Coordinate(0, 'phi')
        ]
        return cls(
            coords = coords,
            g_m = Matrix([
                ['-f(t,r)', 0, 0, 0],
                [0, 'h(t,r)', 0, 0],
                [0, 0, 'r^2', 0],
                [0, 0, 0, 'r^2*sin(theta)^2']
            ])
        )

    @classmethod
    def from_demo_4(cls) -> 'Run_Calc':
        coords = [Coordinate(0, 'x'), Coordinate(1, 'y')]
        return cls(
            coords = coords,
            g_m = eye(2),
            alt_basis = Basis([
                Basis_Vector(0, ['cos(x)', 'sin(x)'], coords),
                Basis_Vector(0, ['cos(x+pi/2)', 'sin(x+pi/2)'], coords)
            ])
        )


if __name__ == '__main__':
    Run_Calc = Run_Calc.from_demo_1()
    #Run_Calc = Run_Calc.from_demo_2()
    #Run_Calc = Run_Calc.from_demo_3()
    #Run_Calc = Run_Calc.from_demo_4()
    #Run_Calc = Run_Calc.from_stdin()
    
    Run_Calc.print_GR_tensors()
    #print(Run_Calc.return_all_GR_tensors())
