import numpy
# import sys
from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

start_time = MPI.Wtime()

a = 0.0
b = 1.0
n = 10000

def f(x):
        return 4/(1+x*x)

def integrateRange(a, b):
    integral = ((b-a)/2)*(f((a+b) / 2 - (b-a) / (2 * numpy.math.sqrt(3))) + f((a+b) / 2 + (b-a) / (2 * numpy.math.sqrt(3))))
    return integral

h = (b-a)/n
local_n = n/size

local_a = a + rank*local_n*h
local_b = local_a + local_n*h

integral = numpy.zeros(1)
total = numpy.zeros(1)


integral[0] = integrateRange(local_a, local_b)

comm.Reduce(integral, total, op=MPI.SUM, root=0)

if comm.rank == 0:
    print("Result:", total, "time: ", MPI.Wtime() - start_time)