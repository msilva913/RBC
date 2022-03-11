
using PyPlot
using Plots
using BenchmarkTools
using LaTeXStrings, KernelDensity
using Parameters, CSV, Statistics, Random, QuantEcon
using NLsolve, Dierckx, Distributions, ArgParse
using LinearAlgebra, QuadGK, Roots, Optim, LinearInterpolations, Interpolations
using Printf
using Dierckx
using DataFrames, Pandas
include("time_series_fun.jl")

function calibrate(r=0.03, alpha=0.33, l=0.33, delta=0.025, rhox=0.974, stdx=0.009)
    " Convert r to quarterly "
    r_q = (1+r)^(1.0/4.0)-1.0
    beta = 1/(1+r_q)
    " Ratio of capital to labor supply "
    k_l = (alpha/(1/beta-(1-delta)))^(1/(1-alpha))
    " Solve for theta "
    theta = ((1-l)/l*(1-alpha)*k_l^alpha)/(k_l^alpha-delta*k_l)
    CalibratedParameters = (alpha=alpha, beta=beta, delta=delta, theta=theta,
    rhox=rhox, stdx=stdx)
    return CalibratedParameters
end


function steady_state(params)
    @unpack alpha, beta, delta, theta, rhox, stdx = params
    " capital-labor ratio "
    k_l = (alpha/(1/beta-(1-delta)))^(1/(1-alpha))
    " wage and rental rate "
    w = (1-alpha)*k_l^alpha
    R = alpha*k_l^(alpha-1)
    " consumption-labor ratio "
    c_l = k_l^alpha-delta*k_l
    " other variables "
    l = ((1-alpha)/theta*k_l^(alpha-1))/((theta+1-alpha)/theta*k_l^(alpha-1)-delta)
    c = l*c_l
    k = k_l*l
    y = k^alpha*l^(1-alpha)
    i = y-c
    lab_prod = y/l

    SteadyState = (l=l, c=c, k=k, y=y, i=i, w=w, R=R, lab_prod=lab_prod)
    return SteadyState
end


@with_kw mutable struct Para{T1, T2, T3, T4}
    # model parameters
    alpha::Float64 = 0.33
    beta::Float64 = 0.99
    delta::Float64 = 0.025
    theta::Float64 = 1.82
    rhox::Float64 = 0.974
    stdx::Float64 = 0.009

    # numerical parameter
    k_l::Float64 = 5.0
    k_u::Float64 = 15.0
    sig::Float64 = 1e-6
    max_iter::Int64 = 1000
    NK::Int64 = 50
    NS::Int64 = 20
    T::Float64 = 1e5
    mc::T1 = rouwenhorst(NS, rhox, stdx, 0)
    P::T2 = mc.p
    A::T3 = exp.(mc.state_values)
    k_grid::T4 = range(k_l, stop=k_u, length =NK)
end

function update_params(self, cal)
    @unpack alpha, beta, delta, theta = cal
    self.alpha = alpha
    self.beta = beta
    self.delta = delta
    self.theta = theta
    nothing
end


function RHS_fun_cons(l_pol::Function, para::Para)
    @unpack alpha, beta, delta, theta, P, NK, NS, k_grid, A = para
    # consumption given state and labor
    
    RHS = zeros(NK, NS)
    for (i, k) in enumerate(k_grid)
        for z in 1:NS
            # labor policy
            l_i = l_pol(k, z)
            # consumption and output
            y = A[z]*k^alpha*l_i^(1-alpha)
            c = (1-l_i)/theta*(1-alpha)*y/l_i
            #c = min(c, y)
            # update capital
            k_p = (1-delta)*k + A[z]*k^alpha*l_i^(1-alpha) - c
            for z_hat in 1:NS
                # update labor supply via interpolation
                l_p = l_pol(k_p, z_hat)
                # update consumption
                y_p = A[z_hat]*k_p^alpha*l_p^(1-alpha)
                c_p = (1-l_p)/theta*(1-alpha)*y_p/l_p
                #c_p = min(c_p, y_p)
                # future marginal utility
                RHS[i, z] += P[z, z_hat]*((1/c_p)*(alpha*y_p/k_p+1-delta))
            end
        end
    end
    RHS = (beta.*RHS)
    rhs_fun(k, z) = LinearInterpolation(k_grid, RHS[:, z], extrapolation_bc=Line())(k)
    return rhs_fun
end


function labor_supply_loss(l_i::Float64, k::Float64, z::Int64, RHS_fun::Function,  para::Para)
    
    @unpack A, alpha, theta = para
    # convert domain from (0, infty) to (0, 1)
    # optimal consumption
    y = A[z]*k^alpha*l_i^(1-alpha)
    c = (1-l_i)/theta*(1-alpha)*y/l_i
    #c = min(c, y)
    error =  1/c - RHS_fun(k, z)
    return error
end


function solve_model_time_iter(l, para::Para; tol=1e-6, max_iter=1000, verbose=true, 
                                print_skip=25)
    # Set up loop 
    @unpack k_grid, NS, A, alpha, theta= para
    # Initial consumption level
    c = similar(l)
    c_new = similar(l)
    for z in 1:NS
        c[:, z] = A[z].*k_grid.^alpha.*l[:, z].^(1-alpha)
    end

    err = 1
    iter = 1
    while (iter < max_iter) && (err > tol)
        # interpolate given labor grid l
        l_pol(k, z) = Interpolate(k_grid, @view(l[:, z]), extrapolate=:reflect)(k)
        RHS_fun = RHS_fun_cons(l_pol, para)
        for (i, k) in enumerate(k_grid)
            for z in 1:NS
                # solve for labor supply
                #l_i = l_pol(k, z)
                l_i = find_zero(l_i -> labor_supply_loss(l_i, k, z, RHS_fun, para), (1e-10, 0.99), Bisection() )
                #l_i = abs(h_i)/(1+abs(h_i))
                l[i, z] = l_i
                # implied consumption
                y = A[z]*k^alpha*l_i^(1-alpha)
                c_new[i, z] = (1-l_i)/theta*(1-alpha)*y/l_i
            end
        end
        #@printf(" %.2f", (mean(c)))
        err = maximum(abs.(c_new-c)/max.(abs.(c), 1e-10))
        if verbose && iter % print_skip == 0
            print("Error at iteration $iter is $err.")
        end
        iter += 1
        c = c_new
    
    end

    # Get convergence level
    if iter == max_iter
        print("Failed to converge!")
    end

    if verbose && (iter < max_iter)
        print("Converged in $iter iterations")
    end
    y = similar(c)
    inv = similar(c)
    for (i, k) in enumerate(k_grid)
        for z in 1:NS
            y[i, z] = A[z]*k^alpha*l[i, z]^(1-alpha)
        end
    end

    inv = y - c
    w = (1-alpha).*y./l
    R = alpha.*y./k_grid
    l_pol(k, z) = LinearInterpolation(k_grid, l[:, z], extrapolation_bc=Line())(k)
    return l, l_pol, c, y, inv, w, R
end


function simulate_series(l_mat::Array, para::Para, burn_in=200, capT=10000)

    @unpack rhox, stdx, P, mc, A, alpha, theta, delta, k_grid = para
    l_pol(k, z) = Interpolate(k_grid, @view(l_mat[:, z]), extrapolate=:reflect)(k)
    capT = capT + burn_in + 1

    # Extract indices of simualtes shocks
    z_indices = simulate_indices(mc, capT)
    z_series = A[z_indices]
    # Simulate shocks
    
    k = ones(capT+1)
    var = ones(capT, 3)
    l, y, c = [ var[:,i] for i in 1:size(var, 2) ]

    for t in 1:capT
        l[t] = l_pol(k[t], z_indices[t])
        y[t] = z_series[t]*k[t]^alpha*l[t]^(1-alpha)
        c[t] = (1-l[t])/theta*(1-alpha)*y[t]/l[t]
        k[t+1] = (1-delta)*k[t] + y[t] - c[t]
    end
    k = k[1:(end-1)]
    i = y - c
    w = (1-alpha).*y./l
    R = alpha.*y./k
    lab_prod = y./l
    Simulation = (l=l, y=y, c=c, k=k, i=i, w=w, R=R,
                 lab_prod=lab_prod, eta_x=log.(z_series), z_indices)
    return Simulation
end


function impulse_response(l_mat, para, k_init; irf_length=40, scale=1.0)

    @unpack rhox, stdx, P, mc, A, alpha, theta, delta, k_grid, NK, NS = para

    # Bivariate interpolation (AR(1) shocks, so productivity can go off grid)
    L = Spline2D(k_grid, A, l_mat)

    eta_x = zeros(irf_length)
    eta_x[1] = stdx*scale
    for t in 1:(irf_length-1)
        eta_x[t+1] = rhox*eta_x[t]
    end
    z = exp.(eta_x)
    z_bas = ones(irf_length)

    function impulse(z_series)

        k = zeros(irf_length+1)
        l = zeros(irf_length)
        c = zeros(irf_length)
        y = zeros(irf_length)

        k[1] = k_init

        for t in 1:irf_length
            # labor
            l[t] = L(k[t], z_series[t])
            y[t] = z_series[t]*k[t]^alpha*l[t]^(1-alpha)
            c[t] = (1-l[t])/theta*(1-alpha)*y[t]/l[t]
            k[t+1] = (1-delta)*k[t] + y[t] - c[t]
        end

        k = k[1:(end-1)]
        i = y - c
        w = (1-alpha).*y./l
        R = alpha.*y./k
        lab_prod = y./l
        out = [c k l i w R y lab_prod]
        return out
    end

    out_imp = impulse(z)
    out_bas = impulse(z_bas)

    irf_res = similar(out_imp)
    @. irf_res = 100*log(out_imp/out_bas)
    #out = [log.(x./mean(getfield(simul, field))) for (x, field) in
    #zip([c, k[1:(end-1)], l, i, w, R, y, lab_prod], [:c, :k, :l, :i, :w, :R, :y, :lab_prod])]
    c, k, l, i, w, R, y, lab_prod = [irf_res[:, i] for i in 1:size(irf_res, 2)]

    irf = (l=l, y=y, c=c, k=k, i=i, w=w, R=R,
                 lab_prod=lab_prod, eta_x=100*log.(z))
    return irf
end


function residual(l_pol, simul, para::Para; burn_in=200)
    capT = size(simul.c)[1]
    resids = zeros(capT)
    @unpack A, alpha, theta, P = para
    @unpack k, z_indices = simul

    " Pre-allocate arrays "
    basis_mat = zeros(2, size(P)[1])
    rhs_fun = RHS_fun_cons(l_pol, para)

    " Right-hand side of Euler equation "
    rhs = rhs_fun.(k, z_indices)
    loss = 1.0 .- simul.c .* rhs
    return loss[burn_in:end]
end  
   

para = Para()
cal = calibrate()
update_params(para, cal)
@unpack NK, NS, A, k_grid, alpha = para

l = zeros(NK, NS)
c = zeros(NK, NS)
l .= 0.5
# iterate until policy function converges
l_mat, l_pol, c, y, inv, w, R = solve_model_time_iter(l, para, verbose=false)


" Solve for steady state "
steady = steady_state(cal)

" Simulation "
simul = simulate_series(l_mat, para) 
@unpack c, k, l, i, w, R, y, lab_prod, eta_x, z_indices = simul

" Log deviations from stationary mean "
out = [log.(getfield(simul, x)./mean(getfield(simul,x))) for x in keys(steady)]
l, c, k, y, i, w, R, lab_prod = out
simul_dat = DataFrames.DataFrame(l=l, c=c, k=k, y=y, i=i, w=w, R=R, lab_prod=lab_prod)


fig, ax = subplots(1, 3, figsize=(20, 5))
t = 250:1000
ax[1].plot(t, c[t], label="c")
ax[1].plot(t, l[t], label="l")
ax[1].plot(t, i[t], label="i")
ax[1].plot(t, y[t], label="y")
ax[1].set_title("Consumption, investment, output, and labor supply")
ax[1].legend()

ax[2].plot(t, w[t], label="w")
ax[2].plot(t, R[t], label="R")
ax[2].set_title("Wage and rental rate of capital")
ax[2].legend()

ax[3].plot(t, eta_x[t], label="x")
ax[3].plot(t, lab_prod[t], label="labor productivity")
ax[3].set_title("Total factor and labor productivity")
ax[3].legend()
display(fig)
PyPlot.savefig("simulations.pdf")

" Residuals "
res = residual(l_pol, simul, para)

" Impulse responses "
k_1 = mean(simul.k)
irf = impulse_response(l_mat, para, k_1, irf_length=60)

fig, ax = subplots(1, 3, figsize=(20, 5))
ax[1].plot(irf.c, label="c")
ax[1].plot(irf.i, label="i")
ax[1].plot(irf.l, label="l")
ax[1].plot(irf.y, label="y")
ax[1].set_title("Consumption, investmnt, output, and labor supply")
ax[1].legend()

ax[2].plot(irf.w, label="w")
ax[2].plot(irf.R, label="R")
ax[2].set_title("Wage and rental rate of capital")
ax[2].legend()

ax[3].plot(irf.eta_x, label="x")
ax[3].plot(irf.lab_prod, label="Labor productivity")
ax[3].set_title("Total factor and labor productivity")
ax[3].legend()
display(fig)
PyPlot.savefig("rbc_irf.pdf")

" Moments from simulated data "
# convert to Pandas DataFrame
#simul_dat = Pandas.DataFrame(simul_dat)
mom = moments(simul_dat)
      



