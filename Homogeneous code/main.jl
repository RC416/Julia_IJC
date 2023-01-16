#=
Solving the "Rewards Program" model with the "IJC algorithm" from Imai, Jain and Ching 2009.
This version uses homogeneous individuals. 

Outline

0. Parameters and setup
    - Define model parameters and observables.
    - Precalculate the state correspondence; which states are accessible
    from each starting state and the corresponding choices/rewards.
    - Package the parameters and environment variables into data structs.

1. Simulate data
    - Simulate data according to model parameters.
    - Key steps: 
        - draw observables (price) and unobservables (unobserved demand).
        - Solve Value Function.
        - For each individual in each period, find and record optimal choice.

2. Estimate model using IJC algorithm
    - Define search parameters.
    - See notes below for outline of the algorithm.
    - Key data points, such as accepted and rejected draws are tracked along the way.
    - Optional: adjust random walk variance to target acceptance rate.

3. Display results
    - Plot accepted parameter draws.
    - Plot mixing during burn-in and main sampling draws.
    - Display mean estimates and credible intervals for model parameters.

4. Extra: solving the model with traditional maximum likelihood
    - Define the likelihood function.
    - Perform numerical optimization using Optim package.
    - Report parameter estimates and confidence intervals.

5. Extra: benchmarking, profiling, and visualzing the likelihood function
    - Benchmarking and profiling of bottleneck/hot functions.
    - Visualize the likelihood function over certain parameter values.

Notes 
    - The key innovation from JIC 2009 is to approximate the value function, rather than 
    solve through the more slow process of value function iteration. This is enclosed in
    Get_Pseudo_Value_Function(). Below each call to this function, I include a call to 
    Solve_Value_Function(). Using these calls instead is tantamount to a more tranditional
    nested fixed point procedure, where the value function is fully solved for each MH draw.
    - These functions and other essential functions are defined in the custom_functions.jl file.
    - Variables and simulated data are stored in 3 data structs to simplify passing data between
    functions. The data structures are defined in the data_structs.jl file.
    - I experimented with automatic differentation in section 4. This is not compatabile with
    the value function contraction mapping, so it doesn't work in this problem.
    - Approximate compute times:
        - IJC: 15 minutes with acceptance rate of ~0.25.
        - Maximum Likelihood: 25 seconds
    - The likelihood function and Solve_Individual_Problem() are well-optimized and are they
    bottlenecks/hot functions. The majority of this program's compute time is spent on unavoidable
    log and exponential calculations.
        - Using the fastmath version of log/exp provides a modest ~5% speed improvement.

IJC algorithm

a. Start with guess value for parameters and solve the value function.
b. Take draw of parameter values from proposal distirbution. 
c. Decide whether to accept the draw.
    - calculate the likelihood ratio with the previous draw (α on T1 slide 54)
        - solve the likelihood function using the current pseudo value function
    - if not accepted, go back to step 2.
d. Update the pseudo value function.
    - peform one iteration of value function iteration.
    - store the result in the array of recent value function. 
    - store the corresponding parameters in the array of recent parameter draws and
    the array of total parameter draws.
    - calculate the new pseudo value function using the kernel density function.
        - average of recent value functions weighted by distance between corresponding 
        parameters.
e. Repeat steps 2-4. for many iterations. 

Additional notes:
    -   Log transform the acceptance ratio (called rₙ here) to avoid numerical issues with 
        very small likelihood probabilities.

        From T1 slide 54: α = L(θ₁|Y)*k(θ₁) / L(θ₂|Y)*k(θ₂)
        becomes:          α = exp(log(L(θ₁|Y) + log(k(θ₁) - log(L(θ₂|Y) - log(k(θ₂))

        Note that the magnitude of the log of the priors is small (~20) 
        compared to the sample likelihood (10⁶)
=#

# Packages.
using Distributions
using Random
using LinearAlgebra
using Parameters
using Plots
using Dates
using DataFrames 

# Custom functions for model.
include("custom_functions.jl")
using .custom_functions

# --------------------------------------------------------------------------------
# 0. Parameters and setup.
# --------------------------------------------------------------------------------
# Model parameters (to be esimated).
α   = [-0.0, -0.0, -0.0]            # store brand intercepts
γ   = -1.0                          # price coefficient
G   = [1.0, 3.0, 8.0]               # value of gifts
σg1 = [0.0, 0.0, 0.0]               # homogeneous consumers
β   = 0.650                         # discount rate

# Environment variables.
n_stores = 3                        # number of stores
n_choices = n_stores + 1            # number of choices for consumer (stores + outside option)
s̄ = [4, 4, 6]                       # gift threshold
price_mean = [1.0, 0.75, 1.5]       # mean of observed prices 
price_stdev = [0.25 0.00 0.00;
               0.00 0.25 0.00;
               0.00 0.00 0.25]      # standard deviation (covariance matrix) of observed prices
n_price_draws = 100                 # number of draws for price integration

# Simulated data parameters.
n_individuals = 1000
n_periods = 100

# Random draws of price for each store plus a vector of 0s for the outside option.
Price_Draws = [ rand(MvNormal(price_mean, price_stdev.^2), n_price_draws)'  zeros(n_price_draws) ]

# Create a vector containing all of the possible states.
state_ranges = [0:(s-1) for s in s̄]
state_space = vec([state for state in Iterators.product(state_ranges...)])

# Precalculate the state correspondence, choices and gifts (which states are accessible from given starting state).
State_Correspondence, Store_Choice, Gift_Earned = Precalculate_State_Correspondence(state_space, s̄)

# Package variables into structs.
Model_Parameters = Model_Parameters_Struct(α, γ, G, σg1, β)
Environment = Environment_Struct{n_stores}(n_choices, s̄, n_price_draws, Price_Draws, state_space,
                                State_Correspondence, Store_Choice, Gift_Earned)


# --------------------------------------------------------------------------------
# 1. Simulate Data.
# --------------------------------------------------------------------------------
# Arrays to store dataset values.
Starting_State =    Array{Int64}(undef, n_individuals, n_periods)
State_Chosen =      Array{Int64}(undef, n_individuals, n_periods)
Store_Chosen =      Array{Int64}(undef, n_individuals, n_periods)
Observed_Prices =   Array{Float64}(undef, n_individuals, n_periods, n_choices)
Unobserved_Demand = Array{Float64}(undef,n_individuals, n_periods, n_choices)

# Assign random draws to each individual, in each period, for each store (and outside option).
for t in 1:n_periods
    Observed_Prices[:, t, :] = [ rand(MvNormal(price_mean, price_stdev.^2), n_individuals)'  zeros(n_individuals) ]
end

# Assign random draws of unobserved demand (ϵ) for each individual, in each period, for each store (and outside option).
Unobserved_Demand = rand(Gumbel(0,1), (n_individuals, n_periods, n_choices))

# For each individual.
for i in 1:n_individuals

    # Solve the Value Function using the true parameters.
    V = Solve_Value_Function(Model_Parameters, Environment)

    # Get starting state (no stamps).
    starting_state = 1

    # For each period.
    for t in 1:n_periods

        # Get prices and unobserved demand
        prices = Observed_Prices[i, t, :]
        ϵ = Unobserved_Demand[i, t, :]

        # Get the available next period states and corresponding choices.
        state_choices = State_Correspondence[starting_state, :]
        store_choices = Store_Choice[starting_state, :]
        gifts_earned = Gift_Earned[starting_state, :]

        # Calculate the alternative-specific value functions.
        Vⱼᵢ = zeros(n_choices)

        # Loop over each possible store choice.
        for choice_index in eachindex(store_choices)
            
            # Get the specific store choice, next state, and whether gift was earned.
            j = store_choices[choice_index]
            gift_earned = gifts_earned[choice_index]
            next_state = state_choices[choice_index]
            
            # If outside option is chosen (last index).
            if j == n_choices
                Vⱼᵢ[choice_index] = 0.0 + γ*prices[j] + 0.0 + ϵ[j] + β*V[next_state]
            
            # Else, if a store is chosen.
            else
                Vⱼᵢ[choice_index] = α[j] + γ*prices[j] + G[j]*gift_earned + ϵ[j] + β*V[next_state]
            end
        end

        # Get and record optimal choice and starting state.
        optimal_choice_index = argmax(Vⱼᵢ)
        State_Chosen[i, t] = state_choices[optimal_choice_index]
        Store_Chosen[i, t] = store_choices[optimal_choice_index]
        Starting_State[i, t] = starting_state

        # Update starting state for next period.
        starting_state = state_choices[optimal_choice_index]
    end
end

# Visualize simulated data.
#display(Starting_State)
#display(State_Chosen)
#display(Store_Chosen)
#display(state_space[State_Chosen])

# Store in struct for later.
Dataset = Dataset_Struct(Starting_State, State_Chosen, Store_Chosen, Observed_Prices)


# --------------------------------------------------------------------------------
# 2. Estimate model using IJC algorithm.
# --------------------------------------------------------------------------------
# Estimation parameters.
θ_draws = 30000
burn_in = 10000
Ñ = 1000

# Stored value functions and parameters.
Ṽ = zeros(size(state_space, 1))
Ṽₙ = zeros(size(state_space, 1), Ñ)
θₙ = Array{Model_Parameters_Struct}(undef, Ñ)
θ_all = Array{Model_Parameters_Struct}(undef, θ_draws)

# Other search parameters and arrays. 
r = Array{Float64}(undef, 0)                          # acceptance probabilities
draw_at_proposal = Array{Int}(undef, 0)               # track draw at each proposal for mixing
θ_placeholder = Model_Parameters_Struct([0.0, 0.0, 0.0], 0.0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0)
fill!(θₙ, θ_placeholder)

# a. Start with guess value for parameters and solve the value function.
θₙ[1] = Model_Parameters_Struct([0.0, 0.0, 0.0], 0.0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0)
θ_all[1] = θₙ[1]

global Ṽ = Get_Pseudo_Value_Function(θₙ[1], Ṽ, Ṽₙ, θₙ)
global Log_L_Y_θ_prior = Log_Sample_Likelihood(Dataset, Environment, θₙ[1], Ṽ)
global Log_K_θ_prior = Log_Prior(θₙ[1])
#global acceptance_rate = 0.33
global ρ = 0.0075 # optional: adjust variance of random walk
start_time = now()

# Main loop.
for n_draw in 2:θ_draws

    # Index to track only the Ñ most recent values.
    ñ = mod(n_draw - 1, Ñ) + 1 # loops back to 0 after surpassing Ñ

    # Calculate the pseudo value function from the previous draw.
    global Ṽ = Get_Pseudo_Value_Function(θ_all[n_draw-1], Ṽ, Ṽₙ, θₙ)
    #global Ṽ = Solve_Value_Function(θ_all[n_draw-1], Environment)

    # Calculate the sample log likelihood and log prior for the previous draw.
    global Log_L_Y_θ_prior = Log_Sample_Likelihood(Dataset, Environment, θ_all[n_draw-1], Ṽ)
    global Log_K_θ_prior = Log_Prior(θ_all[n_draw-1])

    # Variables for choosing next parameter draw.
    accepted_draw = false
    proposal_count = 0

    # Search for new parameter draw.
    while (accepted_draw == false)
        
        # Optional: track and report proposal counts.
        proposal_count += 1
        if (proposal_count >= 100) && (n_draw < burn_in);
            println("too many proposals") # warning message for failed proposals after burn-in.
            break
        end

        # b. Take new parameter draw from proposal distirbution. 
        θ_proposed = Get_Proposal(θ_all[n_draw-1], ρ)

        # Calculate the pseudo value function for this candidate draw.
        global Ṽ = Get_Pseudo_Value_Function(θ_proposed, Ṽ, Ṽₙ, θₙ)
        #global Ṽ = Solve_Value_Function(θ_proposed, Environment)

        # Calculate the sample log likelihood and log prior for this candidate draw.
        global Log_L_Y_θ_proposed = Log_Sample_Likelihood(Dataset, Environment, θ_proposed, Ṽ)
        global Log_K_θ_proposed = Log_Prior(θ_proposed)

        # Calculate and record the acceptance ratio (see notes for equation).
        rₙ = exp(Log_L_Y_θ_proposed + Log_K_θ_proposed
                - Log_L_Y_θ_prior - Log_K_θ_prior)
        push!(r, rₙ)
        push!(draw_at_proposal, n_draw)

        # c. Decide whether to accept the draw based on the value of rₙ.
        if (rₙ >= 1.0) | (rₙ > rand(Uniform(0,1)))
            accepted_draw = true
        end

        # Save the selected parameter draw.
        θₙ[ñ] = θ_proposed
        θ_all[n_draw] = θ_proposed
    end

    # Optional: update the st.dev of random walk to target 0.33 acceptance rate; See Train 2009 Ch. 12 or T1 Slide 62.
    #acceptance_rate = 1 / proposal_count
    #acceptance_rate = n_draw / size(r, 1)
    #global acceptance_rate = (0.95)*acceptance_rate + (0.05)*(1.0 / proposal_count)
    #if acceptance_rate > 0.33
    #    global ρ = ρ * 1.005
    #elseif acceptance_rate < 0.33
    #    global ρ = ρ / 1.005
    #end

    # Optional: display proposal counts.
    #println(proposal_count, ", ", round(ρ, digits=4))

    # Reset the variables for a valid candidate draw.
    accepted_draw = false
    proposal_count = 0

    # Calculate the pseudo value function for this draw.
    global Ṽ = Get_Pseudo_Value_Function(θₙ[ñ], Ṽ, Ṽₙ, θₙ)
    #global Ṽ = Solve_Value_Function(θₙ[ñ], Environment)

    # d. Update the pseudo value function with one value function iteration using the new parameter draw.
    for state_index in eachindex(Ṽ)
        Ṽₙ[state_index, ñ] = Solve_Individual_Problem(state_index, Ṽ, θₙ[ñ], Environment)
    end

    # Display progress.
    if mod(n_draw, 500) == 0; 
        println("completed $n_draw draws, acceptance rate $(n_draw/size(r,1))")
    end
end

# Display total compute time.
end_time = now()
println(canonicalize(Dates.CompoundPeriod(DateTime(end_time) - DateTime(start_time))))


# --------------------------------------------------------------------------------
# 3. Display results.
# --------------------------------------------------------------------------------
# Plots of accepted parameter draws.
draw_range = 2:θ_draws
display(plot([draw_range, [0, θ_draws], [0.0]], 
[[θ_all[n].α[1] for n in draw_range], [α[1], α[1]], [0.0]], 
labels=["accepted α₁ draws" "true value" ""], lines=[:solid :dash], legend=:topright))

display(plot([draw_range, [0, θ_draws], [0.0]], 
[[θ_all[n].α[2] for n in draw_range], [α[2], α[2]], [0.0]], 
labels=["accepted α₂ draws" "true value" ""], lines=[:solid :dash], legend=:topright))

display(plot([draw_range, [0, θ_draws], [0.0]], 
[[θ_all[n].α[3] for n in draw_range], [α[3], α[3]], [0.0]], 
labels=["accepted α₃ draws" "true value" ""], lines=[:solid :dash], legend=:topright))

display(plot([draw_range, [0, θ_draws], [0.0]], 
[[θ_all[n].γ for n in draw_range], [γ, γ], [0.0]], 
labels=["accepted γ draws" "true value" ""], lines=[:solid :dash], legend=:topright))

draw_range = 2:θ_draws
display(plot([draw_range, [0, θ_draws], [0.0]], 
[[θ_all[n].β for n in draw_range], [β, β], [0.0]], 
labels=["accepted β draws" "true value" ""], lines=[:solid :dash], legend=:bottomright))

display(plot([draw_range, [0, θ_draws], [0.0]], 
[[θ_all[n].G[1] for n in draw_range], [G[1], G[1]], [0.0]], 
labels=["accepted G₁ draws" "true value" ""], lines=[:solid :dash], legend=:bottomright))

display(plot([draw_range, [0, θ_draws], [0.0]], 
[[θ_all[n].G[2] for n in draw_range], [G[2], G[2]], [0.0]], 
labels=["accepted G₂ draws" "true value" ""], lines=[:solid :dash], legend=:bottomright))

display(plot([draw_range, [0, θ_draws], [0.0]], 
[[θ_all[n].G[3] for n in draw_range], [G[3], G[3]], [0.0]], 
labels=["accepted G₃ draws" "true value" ""], lines=[:solid :dash], legend=:bottomright))

# Plot mixing at various points. Compare to plots in T1 slide 57/58 from Walsh 2004.
display(plot(10:510, [θ_all[draw_at_proposal[n]].γ for n in 10:510], title="Mixing for γ parameter", legend=false))
display(plot(5001:5500, [θ_all[draw_at_proposal[n]].γ for n in 5001:5500], title="Mixing for γ parameter", legend=false))
display(plot(15001:15500, [θ_all[draw_at_proposal[n]].γ for n in 15001:15500], title="Mixing for γ parameter", legend=false))
display(plot(25001:25500, [θ_all[draw_at_proposal[n]].γ for n in 25001:25500], title="Mixing for γ parameter", legend=false))
display(plot(55001:55500, [θ_all[draw_at_proposal[n]].γ for n in 55001:55500], title="Mixing for γ parameter", legend=false))

# Results table.
raw_θ_all = zeros(θ_draws - burn_in, 8)

for θ_index in eachindex(raw_θ_all[:,1])
    raw_θ_all[θ_index, :] = 
    [θ_all[θ_index + burn_in].α;
     θ_all[θ_index + burn_in].γ;
     θ_all[θ_index + burn_in].β;
     θ_all[θ_index + burn_in].G]
end

# Collect key results.
true_values = [α; γ; β; G]
mean_observed = mean(raw_θ_all, dims=1)
variance_observed = var(raw_θ_all, dims=1)
stdev_observed = std(raw_θ_all, dims=1)
bottom_95_CI = sort(raw_θ_all, dims=1)[Int(round(0.025*size(raw_θ_all, 1))), :]
top_95_CI = sort(raw_θ_all, dims=1)[Int(round(0.975*size(raw_θ_all, 1))), :]

# Display key results.
bayesian_IJC_MCMC_results = [true_values mean_observed' bottom_95_CI top_95_CI]
variable_names = ["α₁", "α₂", "α₃", "γ", "G₁", "G₂", "G₃", "β"]
println("\n Bayesian MCMC with IJC Results")
display(DataFrame([variable_names bayesian_IJC_MCMC_results],
                  ["variable", "true value", "estimated mean", 
                  "95% Credible Interval Low", "95% Credible Interval High"]))


# --------------------------------------------------------------------------------
# 4. Solving the Model Numerically with Maximum Likelihood
# --------------------------------------------------------------------------------
# Obective function: minimize (negative) log likelihood.
function ℓ(θ)
    
    # Calculate value function for this parameter value.
    Ṽ = Solve_Value_Function(
    Model_Parameters_Struct(θ[1:3],θ[4],θ[5:7],zeros(3),θ[8]),
    Environment)
    
    # Calculate the log likelihood for this parameter value and value function. 
    log_likelihood = Log_Sample_Likelihood(Dataset, Environment, 
        Model_Parameters_Struct(θ[1:3],θ[4],θ[5:7],zeros(3),θ[8]), Ṽ)

    return -log_likelihood
end

# Perform numerical minimization (without derivative functions).
using Optim
result = optimize(ℓ, zeros(8), NelderMead())
θ_solution = Optim.minimizer(result)

# Calculate the numerical Hessian at the solution point.
using FiniteDiff
H(θ) = FiniteDiff.finite_difference_hessian(ℓ, θ)
H(θ_solution)

# Use the Hessian to estimate the standard errors.
V(θ) = inv(H(θ))
SE(θ) = sqrt.(diag(V(θ)))
st_dev = SE(θ_solution)

# Display results.
numerical_MLE_results = [[α;γ;G;β] θ_solution (θ_solution - 1.96*st_dev) (θ_solution + 1.96*st_dev)]
variable_names = ["α₁", "α₂", "α₃", "γ", "G₁", "G₂", "G₃", "β"]
println("\n Numerical Maximum Likelihood Results")
display(DataFrame([variable_names numerical_MLE_results],
                  ["values", "true value", "estimated mean", 
                  "95% Confidence Interval Low", "95% Condifence Interval High"]))


# --------------------------------------------------------------------------------
# 5. Benchmarking, Profiling, and Visualizing the Likelihood Function.
# --------------------------------------------------------------------------------
using BenchmarkTools
using Profile

# Timing key functions.
Ṽ = Solve_Value_Function(Model_Parameters, Environment)
@btime Log_Sample_Likelihood(Dataset, Environment, Model_Parameters, Ṽ)
@btime Solve_Individual_Problem(1, Ṽ, Model_Parameters, Environment)
@btime Solve_Value_Function(Model_Parameters, Environment)

# Profiling key functions.
function L2(n)
    for i in 1:n
        Log_Sample_Likelihood(Dataset, Environment, Model_Parameters, Ṽ)
        Solve_Value_Function(Model_Parameters, Environment)
    end
end
@profview L2(100)

# Visualize the likelihood function.
γ_range = Base._linspace(-3.00, 0.00, 20)
G_range = Base._linspace( 0.00, 5.00, 20)
β_range = Base._linspace( 0.50, 0.99, 20)

L_γ = similar(γ_range)
L_G = similar(G_range)
L_β = similar(β_range)

for n in eachindex(γ_range)
    θ1 = Model_Parameters_Struct(α, γ_range[n], [G[1], G[2], G[3]], σg1, β)
    θ2 = Model_Parameters_Struct(α, γ, [G[1], G_range[n], G[3]], σg1, β)
    θ3 = Model_Parameters_Struct(α, γ, [G[1], G[2], G[3]], σg1, β_range[n])
    L_γ[n] = Log_Sample_Likelihood(Dataset, Environment, θ1, Ṽ)
    L_G[n] = Log_Sample_Likelihood(Dataset, Environment, θ2, Ṽ)
    L_β[n] = Log_Sample_Likelihood(Dataset, Environment, θ3, Ṽ)
end

figure_1 = plot(γ_range, L_γ, label="L(Y|θ)", legend=:right)
xlabel!("γ")
ylabel!("L(Y|θ)")
title!("Likelihood for price parameter")
plot!([γ, γ], [L_γ[argmax(-L_γ)], L_γ[argmax(L_γ)]], label="true value", line=:dash)
display(figure_1)

figure_2 = plot(G_range, L_G, label="L(Y|θ)", legend=:topright)
xlabel!("G")
ylabel!("L(Y|θ)")
title!("Likelihood for reward (store 2)")
plot!([G[2], G[2]], [L_G[argmax(-L_G)], L_G[argmax(L_G)]], label="true value", line=:dash)
display(figure_2)

figure_3 = plot(β_range, L_β, label="L(Y|θ)", legend=:topright)
xlabel!("β")
ylabel!("L(Y|θ)")
title!("Likelihood for discount factor")
plot!([β, β], [L_β[argmax(-L_β)], L_β[argmax(L_β)]], label="true value", line=:dash)
display(figure_3)