# Codeframe adopted from the ParallelStencil.jl module tutorial on github (https://github.com/omlins/ParallelStencil.jl)
# https://github.com/omlins/ParallelStencil.jl/blob/main/examples/diffusion2D_shmem_novis.jl
const USE_GPU = true
using BenchmarkTools
using ParallelStencil
ParallelStencil.@reset_parallel_stencil()
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2);
else
    @init_parallel_stencil(Threads, Float64, 2);
end

@parallel_indices (ix,iy) function gs2D_step!(U2, U, V2, V, ru, rv, f, k, dt, _dx, _dy)
    tx  = @threadIdx().x + 1
    ty  = @threadIdx().y + 1
    U_l = @sharedMem(eltype(U), (@blockDim().x+2, @blockDim().y+2))
    U_l[tx,ty] = U[ix,iy]
    V_l = @sharedMem(eltype(V), (@blockDim().x+2, @blockDim().y+2))
    V_l[tx,ty] = V[ix,iy]

    if (ix>1 && ix<size(U2,1) && iy>1 && iy<size(U2,2))
        if (@threadIdx().x == 1)             
            U_l[tx-1,ty] = U[ix-1,iy] 
            V_l[tx-1,ty] = V[ix-1,iy] 
        end
        if (@threadIdx().x == @blockDim().x) 
            U_l[tx+1,ty] = U[ix+1,iy] 
            V_l[tx+1,ty] = V[ix+1,iy] 
        end
        if (@threadIdx().y == 1)             
            U_l[tx,ty-1] = U[ix,iy-1] 
            V_l[tx,ty-1] = V[ix,iy-1] 
        end
        if (@threadIdx().y == @blockDim().y) 
            U_l[tx,ty+1] = U[ix,iy+1] 
            V_l[tx,ty+1] = V[ix,iy+1] 
        end
        @sync_threads()
        U2[ix,iy] = U_l[tx,ty] + dt*(
                    ((ru*(U_l[tx+1,ty] - U_l[tx,ty])*_dx) - (ru*(U_l[tx,ty] - U_l[tx-1,ty])*_dx))*_dx
                    + ((ru*(U_l[tx,ty+1] - U_l[tx,ty])*_dy) - (ru*(U_l[tx,ty] - U_l[tx,ty-1])*_dy))*_dy
                    - U_l[tx, ty]*V_l[tx,ty]^2 + f*(1-U_l[tx,ty]));
        V2[ix,iy] = V_l[tx,ty] + dt*(
                    ((rv*(V_l[tx+1,ty] - V_l[tx,ty])*_dx) - (rv*(V_l[tx,ty] - V_l[tx-1,ty])*_dx))*_dx
                    + ((rv*(V_l[tx,ty+1] - V_l[tx,ty])*_dy) - (rv*(V_l[tx,ty] - V_l[tx,ty-1])*_dy))*_dy
                    + U_l[tx, ty]*V_l[tx,ty]^2 - (f+k)*V_l[tx,ty]);
    end
    return
end

function plot_temp(U, V, iter; root = 0)
    println("Plotting t$(iter).png")
    p1 = contour(U, fill = true, color = :viridis, aspect_ratio = :equal)
    p2 = contour(V, fill = true, color = :viridis, aspect_ratio = :equal)
    #plot(p1)
    #savefig("t$(iter).png")
    plot(p2)
    savefig("t$(iter).png")
end

function gs2D()
# Physics
ru       = 0.1;                                          # Diffusion Rates for the U species
rv       = 0.05;                                         # Diffusion Rates for the V species
f        = 0.0545;
k        = 0.062;
lx, ly   = 1.0, 1.0;                                     # Length of computational domain in dimension x and y

# Numerics
nx, ny   = 105, 105;                                     # Number of gridpoints in dimensions x and y
nt       = 100;                                          # Number of time steps
dx       = lx/(nx-1);                                    # Space step in x-dimension
dy       = ly/(ny-1);                                    # Space step in y-dimension
_dx, _dy = 1.0/dx, 1.0/dy;

# Array initializations
U   = @ones(nx, ny);
U2  = @ones(nx, ny);
V   = @zeros(nx, ny);
V2  = @zeros(nx, ny);

# Initial conditions
for i = 43:63
    for j = 43:63
        U[i, j] = 0.5
        V[i, j] = 0.25
    end
end

U2 .= U;                                                 # Assign also U2 to get correct boundary conditions.
V2 .= V;                                                 # Assign also V2 to get correct boundary conditions.
#
# GPU launch parameters
#threads = (32, 8)
threads = (32, 32)
blocks  = (nx, ny) .÷ threads
#
# Time loop
dt   = 1;                                                         # Time step for 2D GrayScott reaction diffusion
for it = 1:nt
    if (it == 11) GC.enable(false); global t_tic=time(); end      # Start measuring time.
    @parallel blocks threads shmem=prod(threads.+2)*sizeof(Float64) gs2D_step!(U2, U, V2, V, ru, rv, f, k, dt, _dx, _dy);
    U, U2 = U2, U;
    V, V2 = V2, V;
    #V_scaled = 255 * (V2 .- minimum(V2)) / (maximum(V2) - minimum(V2));
    #if (it % 10 == 0)
    #    plot_temp(U, V, it);
    #end
end
time_s = time() - t_tic

# Performance
A_eff = (2+2)*1/1e9*nx*ny*sizeof(eltype(U));             # Effective main memory access per iteration [GB] 
t_it  = time_s/(nt-10);                                  # Execution time per iteration [s]
T_eff = A_eff/t_it;                                      # Effective memory throughput [GB/s]
println("time_s=$time_s t_it=$t_it T_eff=$T_eff");

# Performance
A_eff = (2+2)*1/1e9*nx*ny*sizeof(eltype(U));             # Effective main memory access per iteration [GB] 
t_it = @belapsed begin @parallel $blocks $threads shmem=prod($threads.+2)*sizeof(Float64) gs2D_step!($U2, $U, $V2, $V, $ru, $rv, $f, $k, $dt, $_dx, $_dy); end
println("Benchmarktools (min): t_it=$t_it T_eff=$(A_eff/t_it)");

end

gs2D()