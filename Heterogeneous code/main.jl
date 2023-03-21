#=
Solving the "Rewards Program" model with the "IJC algorithm" from Imai, Jain and Ching 2009.
This version uses heterogeneous individuals. 

The main code file for homogeneous individuals contains documentation on the key code sections
used here as well as the IJC algorithm.

The key differences for this heterogenous individuals implementation:
    - The parameter data struct now contains a vector of heterogeneous parameter draws
    for parameters that vary over individuals. These parameters are draw in Section 0.
    and are incorporated in Section 1. when creating the simulated dataset.
    - Various functions are modified to accomodate changes in individual parameters. Each
    now accepts an "individual index" parameter to know which individual-specific parameters
    to use when performing choice calculations.
        - Solve_Individual_Problem
        - Likelihood function
    - The IJC algorithm is modified to draw the population parameters for heterogenous 
    individuals (mean and variance of G) by Gibbs sampling, and to draw other parameters
    (individual G values, and population α, γ, and β values) by random walk Metropolis-Hastings.
    - Approximating the value function can be done in 2 ways: store value function iterations
    for each individual, or store a random individual's value at each iteration in a shared
    set of value functions. I employ the first method.
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
α   = [-0.0, -0.0, -0.0]                # store brand intercepts
γ   = -1.0                              # price coefficient
G   = [1.0, 3.0, 8.0]                   # value of gifts
σg  = diagm([0.5, 1.0, 2.0])            # covariance of gift values (heterogeneous consumers)
β   = 0.650                             # discount rate

# Environment variables.
n_stores = 3                            # number of stores
n_choices = n_stores + 1                # number of choices for consumer (stores + outside option)
s̄ = [4, 4, 6]                           # gift threshold
price_mean = [1.0, 0.75, 1.5]           # mean of observed prices 
price_stdev = diagm([0.25, 0.25, 0.25]) # standard deviation (covariance matrix) of observed prices
n_price_draws = 100                     # number of draws for price integration

# Simulated data parameters.
n_individuals = 100
n_periods = 1000

# Take draws for individual-level parameters
Gᵢ = rand(MvNormal(G, σg), n_individuals)

# Random draws of price for each store plus a vector of 0s for the outside option.
Price_Draws = [ rand(MvNormal(price_mean, price_stdev.^2), n_price_draws)'  zeros(n_price_draws) ]

# Create a vector containing all of the possible states.
state_ranges = [0:(s-1) for s in s̄]
state_space = vec([state for state in Iterators.product(state_ranges...)])

# Precalculate the state correspondence, choices and gifts (which states are accessible from given starting state).
State_Correspondence, Store_Choice, Gift_Earned = Precalculate_State_Correspondence(state_space, s̄)

# Package variables into structs.
Model_Parameters = Model_Parameters_Struct(α, γ, G, σg, Gᵢ, β)
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
for n in 1:n_individuals

    # Solve the Value Function using the true (individual) parameters.
    V = Solve_Value_Function(n, Model_Parameters, Environment)

    # Get starting state (no stamps).
    starting_state = 1

    # For each period.
    for t in 1:n_periods

        # Get prices and unobserved demand
        prices = Observed_Prices[n, t, :]
        ϵ = Unobserved_Demand[n, t, :]

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
                Vⱼᵢ[choice_index] = α[j] + γ*prices[j] + Gᵢ[j,n]*gift_earned + ϵ[j] + β*V[next_state]
            end
        end

        # Get and record optimal choice and starting state.
        optimal_choice_index = argmax(Vⱼᵢ)
        State_Chosen[n, t] = state_choices[optimal_choice_index]
        Store_Chosen[n, t] = store_choices[optimal_choice_index]
        Starting_State[n, t] = starting_state

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
θ_draws = 30000     # number of total draws
burn_in = 10000     # number of initial draws to ignore
Ñ = 1000            # number of value function approximations to save
ρ1 = 0.0075         # variance for homogeneous parameters random walk
ρ2 = 0.4000         # variance for heterogeneous parameters random walk

# Initialize and store parameters used in the value function estimation.
Ṽₙ = zeros(Ñ, size(state_space, 1), n_individuals)  # each row is a separate value function iteration
θₙ = Array{Model_Parameters_Struct}(undef, Ñ)       # vector of corresponding parameter draws
H = fill(0.01, 4)                                   # choice of kernel bandwidth, one for each type of individual parameter
Value_Function_Parameters = Value_Function_Parameters_Struct(Ṽₙ, θₙ, H)

# Stored value functions and parameters.
θ_all = Array{Model_Parameters_Struct}(undef, θ_draws)

# Other search parameters and arrays. 
r = Array{Float64}(undef, 0)                          # acceptance probabilities
draw_at_proposal = Array{Int}(undef, 0)               # track draw at each proposal for mixing
global individual_proposal_count = Array{Int}(undef, (θ_draws, n_individuals)) 
fill!(individual_proposal_count, 0)                   # track proposals for individual draws

# a. Start with guess value for parameters and solve the value function.
θ_guess = Model_Parameters_Struct([0.0, 0.0, 0.0], 0.0, [0.0, 0.0, 0.0], diagm([1.0, 1.0, 1.0]), zeros(size(Gᵢ)), 0.0)
#θ_guess = Model_Parameters_Struct([0.0, 0.0, 0.0], -1.0, [1.0, 1.0, 1.0], diagm([1.0, 1.0, 1.0]), ones(size(Gᵢ)), 0.0)
#θ_guess = Model_Parameters_Struct([0.0, 0.0, 0.0], -1.0, [4.0, 4.0, 4.0], diagm([1.0, 1.0, 1.0]), 4.0*ones(size(Gᵢ)), 0.0)
#θ_guess = Model_Parameters
fill!(θₙ, θ_guess)
fill!(θ_all, θ_guess)

start_time = now()

# Main loop: can enclose in function for profiling.
#function main(θ_draws, burn_in, θ_all, Ñ, n_individuals, n_stores, ρ1, ρ2, Dataset,
#    Environment, Value_Function_Parameters, r, draw_at_proposal, individual_proposal_count)
for n_draw in 2:θ_draws

    # Index to track only the Ñ most recent values.
    ñ = mod(n_draw - 1, Ñ) + 1 # loops back to 0 after surpassing Ñ

    # Copy parameters from previous draw which will be updated at each step.
    θ_all[n_draw] = deepcopy(θ_all[n_draw-1])

    # Step 1. Draw G through Gibbs sampling (sample G conditional on (σg, Gᵢ) from previous draw).
    Ḡ = @views vec(mean(θ_all[n_draw-1].Gᵢ, dims=2))                    # Train 2009 page 295, labeled b
    Ḡ_var = θ_all[n_draw-1].σg ./ n_individuals                         # Train 2009 page 295, labeled W
    Ḡ_draw = rand(MvNormal(Ḡ, Ḡ_var))

    # Step 2. Draw σg through Gibbs sampling (sample σg conditional on (G, Gᵢ) from previous draw).
    p1 = n_stores + n_individuals                                       # Train 2009 page 297
    S = zeros(size(Ḡ_draw, 1), size(Ḡ_draw, 1))                         # Train 2009 page 297
    for Gᵢ_ind in eachcol(θ_all[n_draw-1].Gᵢ)
        S += (Gᵢ_ind .- Ḡ_draw) * (Gᵢ_ind .- Ḡ_draw)' ./ n_individuals
    end
    p2 = (n_stores*I(n_stores) .+ n_individuals*S) / (n_stores + n_individuals)  # Train 2009 page 297
    #p2 = (n_stores*σg .+ n_individuals*S) / (n_stores + n_individuals)  # Train 2009 page 297
    σg_draw = rand(InverseWishart(p1, p2)) * n_individuals
    #σg_draw = rand(InverseWishart(p1, σg)) * n_individuals

    # Update parameter draw with draws from Step 1. and 2.
    θ_all[n_draw].G[:] = Ḡ_draw
    θ_all[n_draw].σg[:] = σg_draw

    # Step 3. Draw individual homogeneous parameters (α, γ, β).

    # Calculate the sample log likelihood and log prior for the previous draw.
    Log_L_Y_θ_previous = Log_Sample_Likelihood(Dataset, Environment, θ_all[n_draw], Value_Function_Parameters)
    Log_K_θ_previous = Log_Prior(θ_all[n_draw])

    # Variables for choosing next parameter draw.
    accepted_draw = false
    proposal_count = 0

    # Search for new parameter draw.
    while (accepted_draw == false)
        
        # Optional: track and report proposal counts.
        proposal_count += 1
        if (proposal_count >= 100) && (n_draw < burn_in);
            println("too many homogeneous proposals") # warning message for failed proposals after burn-in.
            break
        end

        # b. Take new parameter draw from proposal distirbution. 
        θ_proposed = Get_Proposal(θ_all[n_draw], ρ1)

        # Calculate the sample log likelihood and log prior for this candidate draw.
        Log_L_Y_θ_proposed = Log_Sample_Likelihood(Dataset, Environment, 
                                        θ_proposed, Value_Function_Parameters)
        Log_K_θ_proposed = Log_Prior(θ_proposed)

        # Calculate and record the acceptance ratio (see notes for equation).
        rₙ = exp(Log_L_Y_θ_proposed + Log_K_θ_proposed
                - Log_L_Y_θ_previous - Log_K_θ_previous)
        push!(r, rₙ)
        push!(draw_at_proposal, n_draw)

        # c. Decide whether to accept the draw based on the value of rₙ.
        if (rₙ >= 1.0) | (rₙ > rand(Uniform(0,1)))
            accepted_draw = true
            θ_all[n_draw] = θ_proposed
        end
    end

    # Reset the variables for a valid candidate draw.
    #println("Homogeneous parameters: $proposal_count")
    accepted_draw = false
    proposal_count = 0
   
    # Step 4. Draw heterogeneous parameters (G).
    θ_proposed_individual = deepcopy(θ_all[n_draw])

    # Draw separately for all individuals.
    Threads.@threads for individual_index in 1:n_individuals   # parallel implementation
    #for individual_index in 1:n_individuals                   # serial implementation
        
        # Calculate the sample log likelihood and log prior for the previous draw.
        Log_L_Y_θ_previous_i = Log_Sample_Likelihood(Dataset, Environment, θ_all[n_draw],
                                         Value_Function_Parameters, individual_index)
        Log_K_θ_previous_i = Log_Prior(θ_all[n_draw], individual_index)

        # Variables for choosing next parameter draw.
        accepted_draw_individual = false
            
        # Search for new parameter draw.
        while (accepted_draw_individual == false)

            # Optional: track and report proposal counts.
            individual_proposal_count[n_draw, individual_index] += 1

            if (individual_proposal_count[n_draw, individual_index] > 1000) && (n_draw < burn_in)
                println("too many heterogeneous draw, individual $individual_index")
                break
            end

            # b. Take new parameter draw from proposal distirbution.
            G_individual_proposed = Get_Proposal(θ_all[n_draw], ρ2, individual_index)
            θ_proposed_individual.Gᵢ[:, individual_index] = G_individual_proposed

            # Calculate the (individual) sample log likelihood and log prior for this candidate draw.
            Log_L_Y_θ_proposed_i = Log_Sample_Likelihood(Dataset, Environment, θ_proposed_individual,
                                         Value_Function_Parameters, individual_index)
            Log_K_θ_proposed_i = Log_Prior(θ_proposed_individual, individual_index)

            # Calculate and record the acceptance ratio (see notes for equation).
            rᵢ = exp(Log_L_Y_θ_proposed_i + Log_K_θ_proposed_i
                      - Log_L_Y_θ_previous_i - Log_K_θ_previous_i)

            # c. Decide whether to accept the draw based on the acceptance ratio.
            if (rᵢ >= 1.0) | (rᵢ > rand(Uniform(0,1)))
                accepted_draw_individual = true

                # Calculate the pseudo value function for this draw.
                Ṽ = Get_Pseudo_Value_Function(individual_index, θ_proposed_individual,
                                            Value_Function_Parameters)

                # d. Update the pseudo value function with one value function iteration.
                for state_index in eachindex(Ṽ)
                    Ṽₙ[ñ, state_index, individual_index] = 
                    Solve_Individual_Problem(individual_index, state_index,
                                            Ṽ, θ_proposed_individual, Environment)
                end
            end
        end
    end

    # Record accepted combined parameter draws. 
    θ_all[n_draw] = deepcopy(θ_proposed_individual)   
    θₙ[ñ] = deepcopy(θ_all[n_draw])

    # Display progress.
    if mod(n_draw, 100) == 0; 
        println("completed $n_draw draws, homogeneous parameter acceptance rate $(round(1/(size(r,1)/n_draw),sigdigits=2))")
        println("heterogeneous: $(round.(1 ./ (sum(individual_proposal_count,dims=1) ./ n_draw)[:,1:10], sigdigits=2)) \n")
    end
end
#end # end main function 

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
labels=["accepted α₁ draws" "true value" ""], lines=[:solid :dash], legend=:right))

display(plot([draw_range, [0, θ_draws], [0.0]], 
[[θ_all[n].α[2] for n in draw_range], [α[2], α[2]], [0.0]], 
labels=["accepted α₂ draws" "true value" ""], lines=[:solid :dash], legend=:topright))

display(plot([draw_range, [0, θ_draws], [0.0]], 
[[θ_all[n].α[3] for n in draw_range], [α[3], α[3]], [0.0]], 
labels=["accepted α₃ draws" "true value" ""], lines=[:solid :dash], legend=:bottomright))

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

display(plot([draw_range, [0, θ_draws], [0.0]], 
[[θ_all[n].Gᵢ[2,10] for n in draw_range], [Gᵢ[2,10], Gᵢ[2,10]], [0.0]], 
labels=["accepted G₂ draws for individual 10" "true value" ""], lines=[:solid :dash], legend=:bottomright))

display(plot([draw_range, [0, θ_draws], [0.0]], 
[[θ_all[n].Gᵢ[2,100] for n in draw_range], [Gᵢ[2,100], Gᵢ[2,100]], [0.0]], 
labels=["accepted G₂ draws for individual 100" "true value" ""], lines=[:solid :dash], legend=:bottomright))

display(plot([draw_range, [0, θ_draws], [0.0]], 
[[θ_all[n].σg[1,1] for n in draw_range], [σg[1,1], σg[1,1]], [0.0]], 
labels=["accepted σg₁ draws" "true value" ""], lines=[:solid :dash], legend=:bottomright))

display(plot([draw_range, [0, θ_draws], [0.0]], 
[[θ_all[n].σg[2,2] for n in draw_range], [σg[2,2], σg[2,2]], [0.0]], 
labels=["accepted σg₂ draws" "true value" ""], lines=[:solid :dash], legend=:bottomright))

display(plot([draw_range, [0, θ_draws], [0.0]], 
[[θ_all[n].σg[3,3] for n in draw_range], [σg[3,3], σg[3,3]], [0.0]], 
labels=["accepted σg₃ draws" "true value" ""], lines=[:solid :dash], legend=:bottomright))

# Plot mixing at various points. Compare to plots in T1 slide 57/58 from Walsh 2004.
#display(plot(10:510, [θ_all[draw_at_proposal[n]].γ for n in 10:510], title="Mixing for γ parameter", legend=false))
#display(plot(5001:5500, [θ_all[draw_at_proposal[n]].γ for n in 5001:5500], title="Mixing for γ parameter", legend=false))
#display(plot(15001:15500, [θ_all[draw_at_proposal[n]].γ for n in 15001:15500], title="Mixing for γ parameter", legend=false))
#display(plot(25001:25500, [θ_all[draw_at_proposal[n]].γ for n in 25001:25500], title="Mixing for γ parameter", legend=false))
#display(plot(55001:55500, [θ_all[draw_at_proposal[n]].γ for n in 55001:55500], title="Mixing for γ parameter", legend=false))

# Visualize the likelihood function.
γ_range = Base._linspace(-3.00, 0.00, 50)
G_range = Base._linspace( 0.00, 10.00, 50)
β_range = Base._linspace( 0.50, 0.99, 50)

L_γ = similar(γ_range)
L_G = similar(G_range)
L_β = similar(β_range)

# Profile a specific individual's (ip) parameters (ipar).
individual_index = 5 
parameter = 2 

for n in eachindex(γ_range)
    θ1 = Model_Parameters_Struct(α, γ_range[n], G, σg, Gᵢ, β)
    G̃ = deepcopy(Gᵢ)
    G̃[parameter, individual_index] = G_range[n]
    θ2 = Model_Parameters_Struct(α, γ, G, σg, G̃, β)
    θ3 = Model_Parameters_Struct(α, γ, G, σg, Gᵢ, β_range[n])
    L_γ[n] = Log_Sample_Likelihood(Dataset, Environment, θ1, Value_Function_Parameters)
    L_G[n] = Log_Sample_Likelihood(Dataset, Environment, θ2, Value_Function_Parameters, individual_index)
    L_β[n] = Log_Sample_Likelihood(Dataset, Environment, θ3, Value_Function_Parameters)
end

figure_1 = plot(γ_range, L_γ, label="L(Y|θ)", legend=:right)
xlabel!("γ")
ylabel!("L(Y|θ)")
title!("Likelihood for price parameter")
plot!([γ, γ], [L_γ[argmax(-L_γ)], L_γ[argmax(L_γ)]], label="true value", line=:dash)
display(figure_1)

figure_2 = plot(G_range, L_G, label="L(Y|θ)", legend=:bottomright)
xlabel!("Gᵢ")
ylabel!("L(Y|θ)")
title!("Likelihood for reward (store $individual_index) for individual $parameter")
plot!([Gᵢ[parameter,individual_index], Gᵢ[parameter,individual_index]], [L_G[argmax(-L_G)], L_G[argmax(L_G)]], label="true value", line=:dash)
display(figure_2)

figure_3 = plot(β_range, L_β, label="L(Y|θ)", legend=:topright)
xlabel!("β")
ylabel!("L(Y|θ)")
title!("Likelihood for discount factor")
plot!([β, β], [L_β[argmax(-L_β)], L_β[argmax(L_β)]], label="true value", line=:dash)
display(figure_3)

# Results table.
raw_θ_all = zeros(θ_draws - burn_in, 14)
for θ_index in eachindex(raw_θ_all[:,1])
    raw_θ_all[θ_index, :] = 
    [θ_all[θ_index + burn_in].α;
     θ_all[θ_index + burn_in].γ;
     θ_all[θ_index + burn_in].β;
     θ_all[θ_index + burn_in].G;
     diag(θ_all[θ_index + burn_in].σg);
     θ_all[θ_index + burn_in].Gᵢ[2, 10];
     θ_all[θ_index + burn_in].Gᵢ[2, 50];
     θ_all[θ_index + burn_in].Gᵢ[2, 100]]
end

# Collect key results.
true_values = [α; γ; β; G; diag(σg); 
round(Gᵢ[2,10],digits=2); round(Gᵢ[2,50],digits=2); round(Gᵢ[2,100],digits=2)]
mean_observed = mean(raw_θ_all, dims=1)
variance_observed = var(raw_θ_all, dims=1)
stdev_observed = std(raw_θ_all, dims=1)
bottom_95_CI = sort(raw_θ_all, dims=1)[Int(round(0.025*size(raw_θ_all, 1))), :]
top_95_CI = sort(raw_θ_all, dims=1)[Int(round(0.975*size(raw_θ_all, 1))), :]

# Display key results.
bayesian_IJC_MCMC_results = [true_values mean_observed' bottom_95_CI top_95_CI]
variable_names = ["α₁", "α₂", "α₃", "γ", "β", "Ḡ₁", "Ḡ₂", "Ḡ₃", "σg₁", "σg₂", "σg₃", "G₂-10", "G₂-50", "G₂-100"]
println("\n Bayesian MCMC with IJC Results")
display(DataFrame([variable_names bayesian_IJC_MCMC_results],
                  ["variable", "true value", "estimated mean", 
                  "95% Credible Interval Low", "95% Credible Interval High"]))


# --------------------------------------------------------------------------------
# 5. Benchmarking, Profiling, and Visualizing the Likelihood Function.
# --------------------------------------------------------------------------------
using BenchmarkTools
using Profile

# Timing key functions.
Ṽ = Solve_Value_Function(1, Model_Parameters, Environment)
@btime Log_Sample_Likelihood(Dataset, Environment, Model_Parameters, Value_Function_Parameters)
@btime Solve_Individual_Problem(1, 1, Ṽ, Model_Parameters, Environment)
@btime Solve_Value_Function(1, Model_Parameters, Environment)

# Profiling key functions: call function many times for sufficient sampling.
function Profile_Get_Pseudo_Value_Function(n_repetitions) 
    for repetition in 1:n_repetitions
        Get_Pseudo_Value_Function(1, Model_Parameters, Value_Function_Parameters) 
    end
end

function Profile_Log_Sample_Likelihood(n_repetitions) 
    for repetition in 1:n_repetitions
        Log_Sample_Likelihood(Dataset, Environment, Model_Parameters, Value_Function_Parameters) 
    end
end

function Profile_Solve_Individual_Problem(n_repetitions) 
    for repetition in 1:n_repetitions
        Solve_Individual_Problem(1, 1, Ṽ, Model_Parameters, Environment) 
    end
end

# Get function profiles.
@profview Profile_Get_Pseudo_Value_Function(10000)
@profview Profile_Log_Sample_Likelihood(1000)
@profview Profile_Solve_Individual_Problem(100000)

# Main IJC loop enclosed in function for profiling.
#@profview main(300, burn_in, θ_all, Ñ, n_individuals, n_stores, ρ1, ρ2, Dataset,
#Environment, Value_Function_Parameters, r, draw_at_proposal, individual_proposal_count) # main IJC loop enclosed in function for profiling
