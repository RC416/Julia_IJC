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
    - calculate the likelihood ratio with the previous draw (?? on T1 slide 54)
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
    -   Log transform the acceptance ratio (called r??? here) to avoid numerical issues with 
        very small likelihood probabilities.

        From T1 slide 54: ?? = L(?????|Y)*k(?????) / L(?????|Y)*k(?????)
        becomes:          ?? = exp(log(L(?????|Y) + log(k(?????) - log(L(?????|Y) - log(k(?????))

        Note that the magnitude of the log of the priors is small (~20) 
        compared to the sample likelihood (10???)
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
??   = [-0.0, -0.0, -0.0]            # store brand intercepts
??   = -1.0                          # price coefficient
G   = [1.0, 3.0, 8.0]               # value of gifts
??g1 = [0.0, 0.0, 0.0]               # homogeneous consumers
??   = 0.650                         # discount rate

# Environment variables.
n_stores = 3                        # number of stores
n_choices = n_stores + 1            # number of choices for consumer (stores + outside option)
s?? = [4, 4, 6]                       # gift threshold
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
state_ranges = [0:(s-1) for s in s??]
state_space = vec([state for state in Iterators.product(state_ranges...)])

# Precalculate the state correspondence, choices and gifts (which states are accessible from given starting state).
State_Correspondence, Store_Choice, Gift_Earned = Precalculate_State_Correspondence(state_space, s??)

# Package variables into structs.
Model_Parameters = Model_Parameters_Struct(??, ??, G, ??g1, ??)
Environment = Environment_Struct{n_stores}(n_choices, s??, n_price_draws, Price_Draws, state_space,
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

# Assign random draws of unobserved demand (??) for each individual, in each period, for each store (and outside option).
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
        ?? = Unobserved_Demand[i, t, :]

        # Get the available next period states and corresponding choices.
        state_choices = State_Correspondence[starting_state, :]
        store_choices = Store_Choice[starting_state, :]
        gifts_earned = Gift_Earned[starting_state, :]

        # Calculate the alternative-specific value functions.
        V?????? = zeros(n_choices)

        # Loop over each possible store choice.
        for choice_index in eachindex(store_choices)
            
            # Get the specific store choice, next state, and whether gift was earned.
            j = store_choices[choice_index]
            gift_earned = gifts_earned[choice_index]
            next_state = state_choices[choice_index]
            
            # If outside option is chosen (last index).
            if j == n_choices
                V??????[choice_index] = 0.0 + ??*prices[j] + 0.0 + ??[j] + ??*V[next_state]
            
            # Else, if a store is chosen.
            else
                V??????[choice_index] = ??[j] + ??*prices[j] + G[j]*gift_earned + ??[j] + ??*V[next_state]
            end
        end

        # Get and record optimal choice and starting state.
        optimal_choice_index = argmax(V??????)
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
??_draws = 30000
burn_in = 10000
N?? = 1000

# Stored value functions and parameters.
V?? = zeros(size(state_space, 1))
V????? = zeros(size(state_space, 1), N??)
????? = Array{Model_Parameters_Struct}(undef, N??)
??_all = Array{Model_Parameters_Struct}(undef, ??_draws)

# Other search parameters and arrays. 
r = Array{Float64}(undef, 0)                          # acceptance probabilities
draw_at_proposal = Array{Int}(undef, 0)               # track draw at each proposal for mixing
??_placeholder = Model_Parameters_Struct([0.0, 0.0, 0.0], 0.0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0)
fill!(?????, ??_placeholder)

# a. Start with guess value for parameters and solve the value function.
?????[1] = Model_Parameters_Struct([0.0, 0.0, 0.0], 0.0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0)
??_all[1] = ?????[1]

global V?? = Get_Pseudo_Value_Function(?????[1], V??, V?????, ?????)
global Log_L_Y_??_prior = Log_Sample_Likelihood(Dataset, Environment, ?????[1], V??)
global Log_K_??_prior = Log_Prior(?????[1])
#global acceptance_rate = 0.33
global ?? = 0.0075 # optional: adjust variance of random walk
start_time = now()

# Main loop.
for n_draw in 2:??_draws

    # Index to track only the N?? most recent values.
    n?? = mod(n_draw - 1, N??) + 1 # loops back to 0 after surpassing N??

    # Calculate the pseudo value function from the previous draw.
    global V?? = Get_Pseudo_Value_Function(??_all[n_draw-1], V??, V?????, ?????)
    #global V?? = Solve_Value_Function(??_all[n_draw-1], Environment)

    # Calculate the sample log likelihood and log prior for the previous draw.
    global Log_L_Y_??_prior = Log_Sample_Likelihood(Dataset, Environment, ??_all[n_draw-1], V??)
    global Log_K_??_prior = Log_Prior(??_all[n_draw-1])

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
        ??_proposed = Get_Proposal(??_all[n_draw-1], ??)

        # Calculate the pseudo value function for this candidate draw.
        global V?? = Get_Pseudo_Value_Function(??_proposed, V??, V?????, ?????)
        #global V?? = Solve_Value_Function(??_proposed, Environment)

        # Calculate the sample log likelihood and log prior for this candidate draw.
        global Log_L_Y_??_proposed = Log_Sample_Likelihood(Dataset, Environment, ??_proposed, V??)
        global Log_K_??_proposed = Log_Prior(??_proposed)

        # Calculate and record the acceptance ratio (see notes for equation).
        r??? = exp(Log_L_Y_??_proposed + Log_K_??_proposed
                - Log_L_Y_??_prior - Log_K_??_prior)
        push!(r, r???)
        push!(draw_at_proposal, n_draw)

        # c. Decide whether to accept the draw based on the value of r???.
        if (r??? >= 1.0) | (r??? > rand(Uniform(0,1)))
            accepted_draw = true
        end

        # Save the selected parameter draw.
        ?????[n??] = ??_proposed
        ??_all[n_draw] = ??_proposed
    end

    # Optional: update the st.dev of random walk to target 0.33 acceptance rate; See Train 2009 Ch. 12 or T1 Slide 62.
    #acceptance_rate = 1 / proposal_count
    #acceptance_rate = n_draw / size(r, 1)
    #global acceptance_rate = (0.95)*acceptance_rate + (0.05)*(1.0 / proposal_count)
    #if acceptance_rate > 0.33
    #    global ?? = ?? * 1.005
    #elseif acceptance_rate < 0.33
    #    global ?? = ?? / 1.005
    #end

    # Optional: display proposal counts.
    #println(proposal_count, ", ", round(??, digits=4))

    # Reset the variables for a valid candidate draw.
    accepted_draw = false
    proposal_count = 0

    # Calculate the pseudo value function for this draw.
    global V?? = Get_Pseudo_Value_Function(?????[n??], V??, V?????, ?????)
    #global V?? = Solve_Value_Function(?????[n??], Environment)

    # d. Update the pseudo value function with one value function iteration using the new parameter draw.
    for state_index in eachindex(V??)
        V?????[state_index, n??] = Solve_Individual_Problem(state_index, V??, ?????[n??], Environment)
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
draw_range = 2:??_draws
display(plot([draw_range, [0, ??_draws], [0.0]], 
[[??_all[n].??[1] for n in draw_range], [??[1], ??[1]], [0.0]], 
labels=["accepted ????? draws" "true value" ""], lines=[:solid :dash], legend=:topright))

display(plot([draw_range, [0, ??_draws], [0.0]], 
[[??_all[n].??[2] for n in draw_range], [??[2], ??[2]], [0.0]], 
labels=["accepted ????? draws" "true value" ""], lines=[:solid :dash], legend=:topright))

display(plot([draw_range, [0, ??_draws], [0.0]], 
[[??_all[n].??[3] for n in draw_range], [??[3], ??[3]], [0.0]], 
labels=["accepted ????? draws" "true value" ""], lines=[:solid :dash], legend=:topright))

display(plot([draw_range, [0, ??_draws], [0.0]], 
[[??_all[n].?? for n in draw_range], [??, ??], [0.0]], 
labels=["accepted ?? draws" "true value" ""], lines=[:solid :dash], legend=:topright))

draw_range = 2:??_draws
display(plot([draw_range, [0, ??_draws], [0.0]], 
[[??_all[n].?? for n in draw_range], [??, ??], [0.0]], 
labels=["accepted ?? draws" "true value" ""], lines=[:solid :dash], legend=:bottomright))

display(plot([draw_range, [0, ??_draws], [0.0]], 
[[??_all[n].G[1] for n in draw_range], [G[1], G[1]], [0.0]], 
labels=["accepted G??? draws" "true value" ""], lines=[:solid :dash], legend=:bottomright))

display(plot([draw_range, [0, ??_draws], [0.0]], 
[[??_all[n].G[2] for n in draw_range], [G[2], G[2]], [0.0]], 
labels=["accepted G??? draws" "true value" ""], lines=[:solid :dash], legend=:bottomright))

display(plot([draw_range, [0, ??_draws], [0.0]], 
[[??_all[n].G[3] for n in draw_range], [G[3], G[3]], [0.0]], 
labels=["accepted G??? draws" "true value" ""], lines=[:solid :dash], legend=:bottomright))

# Plot mixing at various points. Compare to plots in T1 slide 57/58 from Walsh 2004.
display(plot(10:510, [??_all[draw_at_proposal[n]].?? for n in 10:510], title="Mixing for ?? parameter", legend=false))
display(plot(5001:5500, [??_all[draw_at_proposal[n]].?? for n in 5001:5500], title="Mixing for ?? parameter", legend=false))
display(plot(15001:15500, [??_all[draw_at_proposal[n]].?? for n in 15001:15500], title="Mixing for ?? parameter", legend=false))
display(plot(25001:25500, [??_all[draw_at_proposal[n]].?? for n in 25001:25500], title="Mixing for ?? parameter", legend=false))
display(plot(55001:55500, [??_all[draw_at_proposal[n]].?? for n in 55001:55500], title="Mixing for ?? parameter", legend=false))

# Results table.
raw_??_all = zeros(??_draws - burn_in, 8)

for ??_index in eachindex(raw_??_all[:,1])
    raw_??_all[??_index, :] = 
    [??_all[??_index + burn_in].??;
     ??_all[??_index + burn_in].??;
     ??_all[??_index + burn_in].??;
     ??_all[??_index + burn_in].G]
end

# Collect key results.
true_values = [??; ??; ??; G]
mean_observed = mean(raw_??_all, dims=1)
variance_observed = var(raw_??_all, dims=1)
stdev_observed = std(raw_??_all, dims=1)
bottom_95_CI = sort(raw_??_all, dims=1)[Int(round(0.025*size(raw_??_all, 1))), :]
top_95_CI = sort(raw_??_all, dims=1)[Int(round(0.975*size(raw_??_all, 1))), :]

# Display key results.
bayesian_IJC_MCMC_results = [true_values mean_observed' bottom_95_CI top_95_CI]
variable_names = ["?????", "?????", "?????", "??", "G???", "G???", "G???", "??"]
println("\n Bayesian MCMC with IJC Results")
display(DataFrame([variable_names bayesian_IJC_MCMC_results],
                  ["variable", "true value", "estimated mean", 
                  "95% Credible Interval Low", "95% Credible Interval High"]))


# --------------------------------------------------------------------------------
# 4. Solving the Model Numerically with Maximum Likelihood
# --------------------------------------------------------------------------------
# Obective function: minimize (negative) log likelihood.
function ???(??)
    
    # Calculate value function for this parameter value.
    V?? = Solve_Value_Function(
    Model_Parameters_Struct(??[1:3],??[4],??[5:7],zeros(3),??[8]),
    Environment)
    
    # Calculate the log likelihood for this parameter value and value function. 
    log_likelihood = Log_Sample_Likelihood(Dataset, Environment, 
        Model_Parameters_Struct(??[1:3],??[4],??[5:7],zeros(3),??[8]), V??)

    return -log_likelihood
end

# Perform numerical minimization (without derivative functions).
using Optim
result = optimize(???, zeros(8), NelderMead())
??_solution = Optim.minimizer(result)

# Calculate the numerical Hessian at the solution point.
using FiniteDiff
H(??) = FiniteDiff.finite_difference_hessian(???, ??)
H(??_solution)

# Use the Hessian to estimate the standard errors.
V(??) = inv(H(??))
SE(??) = sqrt.(diag(V(??)))
st_dev = SE(??_solution)

# Display results.
numerical_MLE_results = [[??;??;G;??] ??_solution (??_solution - 1.96*st_dev) (??_solution + 1.96*st_dev)]
variable_names = ["?????", "?????", "?????", "??", "G???", "G???", "G???", "??"]
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
V?? = Solve_Value_Function(Model_Parameters, Environment)
@btime Log_Sample_Likelihood(Dataset, Environment, Model_Parameters, V??)
@btime Solve_Individual_Problem(1, V??, Model_Parameters, Environment)
@btime Solve_Value_Function(Model_Parameters, Environment)

# Profiling key functions.
function L2(n)
    for i in 1:n
        Log_Sample_Likelihood(Dataset, Environment, Model_Parameters, V??)
        Solve_Value_Function(Model_Parameters, Environment)
    end
end
@profview L2(100)

# Visualize the likelihood function.
??_range = Base._linspace(-3.00, 0.00, 20)
G_range = Base._linspace( 0.00, 5.00, 20)
??_range = Base._linspace( 0.50, 0.99, 20)

L_?? = similar(??_range)
L_G = similar(G_range)
L_?? = similar(??_range)

for n in eachindex(??_range)
    ??1 = Model_Parameters_Struct(??, ??_range[n], [G[1], G[2], G[3]], ??g1, ??)
    ??2 = Model_Parameters_Struct(??, ??, [G[1], G_range[n], G[3]], ??g1, ??)
    ??3 = Model_Parameters_Struct(??, ??, [G[1], G[2], G[3]], ??g1, ??_range[n])
    L_??[n] = Log_Sample_Likelihood(Dataset, Environment, ??1, V??)
    L_G[n] = Log_Sample_Likelihood(Dataset, Environment, ??2, V??)
    L_??[n] = Log_Sample_Likelihood(Dataset, Environment, ??3, V??)
end

figure_1 = plot(??_range, L_??, label="L(Y|??)", legend=:right)
xlabel!("??")
ylabel!("L(Y|??)")
title!("Likelihood for price parameter")
plot!([??, ??], [L_??[argmax(-L_??)], L_??[argmax(L_??)]], label="true value", line=:dash)
display(figure_1)

figure_2 = plot(G_range, L_G, label="L(Y|??)", legend=:topright)
xlabel!("G")
ylabel!("L(Y|??)")
title!("Likelihood for reward (store 2)")
plot!([G[2], G[2]], [L_G[argmax(-L_G)], L_G[argmax(L_G)]], label="true value", line=:dash)
display(figure_2)

figure_3 = plot(??_range, L_??, label="L(Y|??)", legend=:topright)
xlabel!("??")
ylabel!("L(Y|??)")
title!("Likelihood for discount factor")
plot!([??, ??], [L_??[argmax(-L_??)], L_??[argmax(L_??)]], label="true value", line=:dash)
display(figure_3)