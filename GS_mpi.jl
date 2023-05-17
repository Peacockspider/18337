# Codeframe adopted from the MPIHaloArrays.jl module tutorial on github (https://github.com/smillerc/MPIHaloArrays.jl)
# https://github.com/smillerc/MPIHaloArrays.jl/blob/main/docs/examples/04-diffusion2d.jl
using CUDA
using MPI, MPIHaloArrays
using Plots
gr()

MPI.Init()
const comm = MPI.COMM_WORLD
const rank = MPI.Comm_rank(comm)
const nprocs = MPI.Comm_size(comm)
const root = 0 # root rank

"""Establish the initial conditions, e.g. place a (square) rectangle in the center
for U initialization, U is 1 everywhere in the domain except in the center zone where u is other value like 0.5,
for V initialization, V is 0 everywhere in the domain except in the center zone where v is other value like 0.25.
"""
function initialize!(x, y, C_0, C_c)
    C = ones(length(x), length(y))
    fill!(C, C_0)

    dx0 = 0.2 # size of the central region
    dy0 = 0.2 # size of the central region
    for j in 1:length(y)
        for i in 1:length(x)
            if -dx0 < x[i] < dx0 && -dy0 < y[j] < dy0
                C[i, j] = C_c
            end
        end
    end
    return C
end


"""Perform 2D GrayScott reaction diffusion"""
function grayscott!(U, U_new, V, V_new, ru, rv, f, k, dt, dx, dy)
    ilo, ihi, jlo, jhi = local_domain_indices(U)
    for j in jlo:jhi
        for i in ilo:ihi
            U_new[i, j] = U[i, j] + dt*(
                ((ru*(U[i+1,j] - U[i,j])/dx) - (ru*(U[i,j] - U[i-1,j])/dx))/dx
                + ((ru*(U[i,j+1] - U[i,j])/dy) - (ru*(U[i,j] - U[i,j-1])/dy))/dy
                - U[i, j]*V[i,j]^2 + f*(1-U[i,j]))
            V_new[i, j] = V[i,j] + dt*(
                ((rv*(V[i+1,j] - V[i,j])/dx) - (rv*(V[i,j] - V[i-1,j])/dx))/dx
                + ((rv*(V[i,j+1] - V[i,j])/dy) - (rv*(V[i,j] - V[i,j-1])/dy))/dy
                + U[i, j]*V[i,j]^2 - (f+k)*V[i,j])
        end
    end
end

function plot_temp(U, V, iter; root = 0)
    U_result = gatherglobal(U; root = root)
    V_result = gatherglobal(V; root = root)
    if rank == root
        println("Plotting t$(iter).png")
        p1 = contour(U_result, fill = true, color = :viridis, aspect_ratio = :equal)
        p2 = contour(V_result, fill = true, color = :viridis, aspect_ratio = :equal)
        #plot(p1)
        #savefig("t$(iter).png")
        plot(p2)
        savefig("t$(iter).png")
    end
end

# ----------------------------------------------------------------
# Initial conditions
dx = 1.0/104; dy = 1.0/104; # grid spacing
ru       = 0.1;   # Diffusion Rates for the U species
rv       = 0.05;  # Diffusion Rates for the V species
f        = 0.0545;
k        = 0.062;
dt = 1 # stable time step

x = -1:dx:1 |> collect # x grid 
y = -1:dy:1 |> collect # y grid 

U_0 = 1.0  # initial concentration of the U species
U_c = 0.5  # concetration of the U species at the center region

V_0 = 0.0  # initial concentration of the V species
V_c = 0.25 # concetration of the V species at the center region

# Initialize the concentration field for the U and V species
U_global = initialize!(x, y, U_0, U_c)
V_global = initialize!(x, y, V_0, V_c)

# ----------------------------------------------------------------
# Parallel topology construction
nhalo = 1 # only a stencil of 5 cells is needed, so 1 halo cell is sufficient
@assert nprocs == 1 "This example is designed with 1 process for testing, 
but can be changed in the topology construction..."
topology = CartesianTopology(comm, [1,1], [true, true]) # periodic boundary conditions

# ----------------------------------------------------------------
# Plot initial conditions
if rank == root
    println("Plotting initial conditions")
    p1 = contour(U_global, fill = true, color = :viridis, aspect_ratio = :equal)
    p2 = contour(V_global, fill = true, color = :viridis, aspect_ratio = :equal)
    plot(p2)
    savefig("t0.png")
end

# ----------------------------------------------------------------
# Distribute the work to each process, e.g. domain decomposition
Uⁿ = scatterglobal(U_global, root, nhalo, topology; 
                   do_corners = false) # the 5-cell stencil doesn't use corner info, 
                                       # so this saves communication time
Vⁿ = scatterglobal(V_global, root, nhalo, topology; 
                   do_corners = false) # the 5-cell stencil doesn't use corner info, 
                                       # so this saves communication time
Uⁿ⁺¹ = deepcopy(Uⁿ) # u at the next timestep
Vⁿ⁺¹ = deepcopy(Vⁿ) # v at the next timestep

niter = 30
plot_interval = 1
info_interval = 1

plot_temp(Uⁿ, Vⁿ, 0) # Plot initial conditions

# Time loop
for iter in 1:niter
    if rank == root && iter % info_interval == 0 println("Iteration: $iter") end
    if iter % plot_interval == 0 plot_temp(Uⁿ, Vⁿ, iter) end

    updatehalo!(Uⁿ)
    updatehalo!(Vⁿ)
    grayscott!(Uⁿ, Uⁿ⁺¹, Vⁿ, Vⁿ⁺¹, ru, rv, f, k, dt, dx, dy)
    Uⁿ.data .= Uⁿ⁺¹.data # update the next time-step
    Vⁿ.data .= Vⁿ⁺¹.data # update the next time-step
end

GC.gc()
MPI.Finalize()