# Metric-Forward-GR-Tensor-Calculator
A calculator that does arduous tensor analysis for you

Depending on the user's responses to a few y/n questions, this calculator takes as input, at most:
- the number of dimensions of the manifold
- the coordinate labels
- the basis vectors
- the inner products of the basis vectors (i.e. the metric components in that basis)

Again, depending on those initial responses, it outputs, at most, the non-zero components of:
- the metric and inverse metric (in the coordinate basis)
- derivatives of the metric, the Christoffel symbols, and the derivatives of the Christoffel symbols (only if using coordinate basis)
- the Riemann curvature tensor, the Ricci curvature tensor, the Ricci scalar, and the Einstein tensor

Right now the main file works best when executed in Jupyter since it's an environment that will display all the pretty LaTeX.
It will still work otherwise, it's just that the output will be raw LaTeX that will need to be copied and pasted into an environment that renders LaTeX.

Here are some useful tips on the user inputs: When inputting metric components, you can use '^' instead of '**' for exponents.
Feel free to include undefined functions- just make sure you include its arguments if you want it to be differentiated correctly
(e.g. if you want a function that will have a non-zero derivative of x, then type 'f(x)' in your expression instead of just 'f').
Also, feel free to use greek letters (spelled out in English) when inputting coordinate labels and/or functions in your metric components.
