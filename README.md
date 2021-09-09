# Metric-Forward-GR-Tensor-Calculator
A calculator that does arduous tensor analysis for you

This calculator takes as input the number of dimensions of a manifold, the coordinate labels being used, 
and the components of a metric, and outputs the non-zero components of the inverse metric, derivatives of 
the metric, the Christoffel symbols, the derivatives of the Christoffel symbols, the Riemann curvature 
tensor, the Ricci curvature tensor, the Ricci scalar, and the Einstein tensor.  Right now the main file 
should be executed in Jupyter, since it's an environment that will display all the pretty LaTeX.

Here are some useful tips on the user inputs: When inputting metric components, you can use '^' instead of 
'**' for exponents, and when multiplying a number by a symbol or something in parentheses, you don't need 
to include a '*'. Feel free to include undefined functions- just make sure you include its arguments if 
you want it to be differentiated correctly (e.g. if you want a function that will have a non-zero derivative 
of x, then type 'f(x)' in your expression instead of just 'f'). Also feel free to use greek letters (spelled 
out in English) when inputting coordinate labels and/or functions in your metric components.
