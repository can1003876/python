"""
Candidate: 1003876

Evolving Laplace-Young surface tenstion equation:
-div( q u^3 grad u)/(1-u) = u_t,

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
output = File("solutions/laplace-young2/sol5.pvd")

# Mesh properties (for ellipse and rectangle)
a = 1.0;
b = 1.0;
xn = 100
yn = 100

# Rectangle
# u_0 = Expression('''0.7*(1 - x[0]*x[0]/(b*b) - x[1]*x[1]/(a*a) )''', a = a, b = b)
u_0 = Expression('''0.0''')
mesh = RectangleMesh(Point(-a/2,-b/2), Point(a/2, b/2), xn, yn)

# Ellipse (figure 1)
domain = mshr.Ellipse(Point(0.0, 0.0), a, b, xn)
mesh = mshr.generate_mesh(domain, 50, "cgal")
# u_0 = Expression('''0.6*sqrt(1 - x[0]*x[0]/(a*a) - x[1]*x[1]/(b*b) + 0.01 )''', a = a, b = b)

# Wrinkled disc
# u_0 = Expression('''0.5*(1 - 0.5*sin(2*x[0]*x[0]/(b*b) + 10*x[1]*x[1]/(a*a)) )''', a = a, b = b)

# Triangular pizza mesh (Figure 2/3/4)
# a = 1.0;
# b = 3.0;
# triangle = mshr.Polygon([Point(-a, 0.0), Point(a, 0.0), Point(0.0, b)])
# half_circle = mshr.Ellipse(Point(0.0, 0.0), a, a, xn) - \
#                 mshr.Rectangle(Point(-a, 0.0), Point(a,a))
# domain = triangle + half_circle
# mesh = mshr.generate_mesh(domain, xn, "cgal")
# u_0 = Expression('''0.3*(1 - 0.5*x[0]*x[0]+ 0.5*sin(2*x[1]*x[1]))''', a = a, b = b)

V = FunctionSpace(mesh, 'CG', 2)
u0 = interpolate(u_0, V);
u1 = Function(V)
v = TestFunction(V)

S1 = assemble(sqrt(1+inner(grad(u0),grad(u0)))*dx)
V1 = assemble(u0*dx)
print 'Initial surface:', S1

# Change in consecutive times
dt = Constant(0.05);
t = float(dt); T = 20
# Tolerance of change of surface area
toldS = 1.e-5
# Variable to store change in surface area
dS = toldS          # to go into first for-loop

q1 = (1+inner(grad(u1),grad(u1)))**(-.5)

# Boundary settings
bc = DirichletBC(V, u0, "on_boundary");
theta = Constant(pi/4)
grad2 = ((1-u1)*grad(v) + grad(u1)*v)/(1-u1)**2
F = (u1-u0)*v*dx - dt*cos(theta)*(u1**3)*v*ds + dt*q1*(u1**3)*inner(grad(u1),grad2)*dx

areas = np.array([S1])
volumes = np.array([S1])
times = np.array([t])
while dS >= toldS and t < T:
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
    volumes = np.append(volumes, [V1])
    # Save this time solution
    output << u1
    # Prepare next iteration
    u0.assign(u1)
    if (t > 4):
        dt = Constant(0.5)
    t += float(dt)

print 'surface area:', assemble(sqrt(1+inner(grad(u0),grad(u0)))*dx)

np.savetxt("solutions/laplace-young2/areas5.csv",areas)
np.savetxt("solutions/laplace-young2/volumes5.csv",volumes)
np.savetxt("solutions/laplace-young2/times5.csv",times)
