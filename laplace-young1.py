"""
Candidate: 1003876

Static Laplace-Young equation:
-div( q grad u) = u,

where q = q(grad u) = (1 + |grad u|^2)^{-1/2}

using Newton solver
"""
from dolfin import *
import mshr
import math
import numpy as np


# Solver settings:
solv_parameters = {"newton_solver": {"relative_tolerance": 1e-5,
                                "report": True,
                                "maximum_iterations": 20}}
compiler_parameters =  {"optimize": True,
                        "cpp_optimize": True,
                        "cpp_optimize_flags": "-O3 -ffast-math -march=native"}
output = File("solutions/laplace-young1/sol1.pvd")


# Mesh properties (for ellipse and rectangle)
a = 1.0;
b = 1.0;
xn = 100
yn = 100

# Rectangle
#mesh = RectangleMesh(Point(-a/2,-b/2), Point(a/2, b/2), xn, yn)
#u_0 = Expression('''0.7*(1 - x[0]*x[0]/(b*b) - x[1]*x[1]/(a*a) )''', a = a, b = b)

# Ellipse
# domain = mshr.Ellipse(Point(0.0, 0.0), a, b, xn)
# mesh = mshr.generate_mesh(domain, 50, "cgal")
# u_0 = Expression('''0.7*(x[0]*x[0]/(b*b) - x[1]*x[1]/(a*a) + (x[0]-x[1])*(x[0]-x[1]) )''', a = a, b = b)

# Triangular pizza mesh (Figure 3)
a = 1.0;
b = 3.0;
triangle = mshr.Polygon([Point(-a, 0.0), Point(a, 0.0), Point(0.0, b)])
half_circle = mshr.Ellipse(Point(0.0, 0.0), a, a, xn) - \
                mshr.Rectangle(Point(-a, 0.0), Point(a,a))
domain = triangle + half_circle
mesh = mshr.generate_mesh(domain, xn, "cgal")
u_0 = Expression('''(1 - 0.5*x[0]*x[0]+ 0.5*sin(x[1]*x[1]))''', a = a, b = b)


V = FunctionSpace(mesh, 'Lagrange', 2)

boundaries = MeshFunction('size_t',mesh, 1)
ds = Measure('ds')[boundaries]

# Solution is u0
u0 = interpolate(u_0, V);
u = TrialFunction(V)
v = TestFunction(V)
q0 = (1+inner(grad(u0),grad(u0)))**(-.5)

# Save initial guess
output << u0

S1 = assemble(sqrt(1+inner(grad(u0), grad(u0)))*dx)
print 'Initial surface:', S1

# Boundary conditions
bc = DirichletBC(V, u0, "on_boundary");
theta = Constant(pi/2)
B = cos(theta)

# Weak form formulation
F =  B*v*ds - q0*inner(grad(u0), grad(v))*dx + u0*v*dx
solve(F == 0, u0, bc,
            solver_parameters = solv_parameters,
            form_compiler_parameters = compiler_parameters)

print 'Surface area 2:', assemble(sqrt(1+inner(grad(u0),grad(u0)))*dx)
output << u0
