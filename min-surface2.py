"""
Candidate: 1003876

Evolving minimal surface equation:
-div( q grad u) = u_t,

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
                                "maximum_iterations": 10}}
compiler_parameters =  {"optimize": True,
                        "cpp_optimize": True,
                        "cpp_optimize_flags": "-O3 -ffast-math -march=native"}
output = File("solutions/min-surface2/sol1.pvd")

# Mesh properties (for ellipse and rectangle)
a = 1.0;
b = 1.0;
xn = 100
yn = 100

# Ellipse (figure 2)
domain = mshr.Ellipse(Point(0.0, 0.0), a, b, xn)
mesh = mshr.generate_mesh(domain, 50, "cgal")
u_0 = Expression('''0.7*(x[0]*x[0]/(b*b) - x[1]*x[1]/(a*a) + (x[0]-x[1])*(x[0]-x[1]) )''', a = a, b = b)

V = FunctionSpace(mesh, 'CG', 2)
u0 = interpolate(u_0, V);
u1 = Function(V)
v = TestFunction(V)

S1 = assemble(sqrt(1+inner(grad(u0),grad(u0)))*dx)
print 'Initial surface:', S1

# Change in consecutive times
dt = Constant(0.05);
t = float(dt); T = 1
# Tolerance of change of surface area
toldS = 1.e-5
# Variable to store change in surface area
dS = toldS          # to go into first for-loop

q1 = (1+inner(grad(u1),grad(u1)))**(-.5)

# Boundary settings
bc = DirichletBC(V, u0, "on_boundary");
theta = Constant(pi/2)
B = cos(theta)/q1
F = (u1-u0)*v*dx - dt*q1*B*v*ds + dt*q1*inner(grad(u1),grad(v))*dx

areas = np.array([S1])
times = np.array([t])
while dS >= toldS:
    solve(F == 0, u1, bc,
                    solver_parameters = solv_parameters,
                    form_compiler_parameters = compiler_parameters)
    # Look at change of surface
    S0 = S1
    S1 = assemble(sqrt(1+inner(grad(u1),grad(u1)))*dx)
    V1 = assemble(u1*dx)
    dS = abs(S1 - S0)
    # Print report
    print 'At time:', t, ' surface area is :',S1
    print 'Volume is :', V1
    areas = np.append(areas, [S1])
    times = np.append(times, [t])
    # Save this time solution
    output << u1
    # Prepare next iteration
    u0.assign(u1)
    t += float(dt)

print 'surface area:', assemble(sqrt(1+inner(grad(u0),grad(u0)))*dx)

np.savetxt("solutions/min-surface2/areas.csv",areas)
np.savetxt("solutions/min-surface2/times.csv",times)
