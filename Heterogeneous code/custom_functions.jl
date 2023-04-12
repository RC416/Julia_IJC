#=
Functions used in Main Module. 
     - Function 1. solves the state transition/correspondence.
     - Functions 2. and 3. solve the Value Function. 
     - Functions 4-8. are used in the MCMC and IJC algorithm.

1. Precalculate_State_Correspondence
    - Takes a vector of all possible starting states.
    - Determines which states are accessible from a given starting state.
    - For each accessible state, determines which store was chosen and
    whether a gift was earned.

2. Solve_Individual_Problem
    - Takes a single starting state and Value Function.
    - Calculates the updated Value Function for that starting state.
    - Does so by calculating the alternative-specific Value Functions and
    the expected Value Function using the key identity from Rust 1987.

3. Solve_Value_Function
    - Applies function 2. over all possible starting states.
    - Peforms contraction mapping over Value Functions until suitable
    convergence is found.
    - Returns the solved Value Function.

4. Get_Proposal
    - Given the previously accepted parameter set, draw a new 
    proposed parameter set. 
    - This version uses a random walk with adjustable st. dev, ρ.

5. Parameter_Distance_Kernel
    - Calculates the distance between two sets of parameters.
    - This verion is a normal kernel, see T2 slide 35.

6. Log_Prior
    - Prior distribution for individual model parameters (not G, σg).
    - Overloaded with 2 versions: one for homogeneous parmeters (α,γ,β),
    and one for heterogeneous parameters (Gᵢ).
    - This uses independent normal distributions.

7. Log_Sample_Likelihood
    - For a given set of model parameters, calculate the log
    likelihood of the observed choices in the dataset.
    - The value function or pseudo value function must be called/solved
    for each individual using function 8.

8. Get_Pseudo_Value_Function
    - Takes the stored value functions and previous parameter draws.
    - Calculates the pseudo value function at a certain parameter draw
    by taking a weighted average of the stored value functions according to 
    their the distance from their corresponding parameter draws.
    - The distance between parameter sets is calculated using function 5.
=#

module custom_functions
export Precalculate_State_Correspondence, 
       Solve_Individual_Problem, Solve_Value_Function,
       Get_Proposal, Parameter_Distance_Kernel, Log_Prior, 
       Log_Sample_Likelihood, Get_Pseudo_Value_Function, 
       Model_Parameters_Struct, Environment_Struct, Dataset_Struct, Value_Function_Parameters_Struct

include("data_structs.jl")
using .data_structs

using Parameters
using Distributions
using LinearAlgebra

# --------------------------------------------------------------------------------
# 1. Precalculate the state correspondence (accessible states), choices and gifts.
# --------------------------------------------------------------------------------
function Precalculate_State_Correspondence(state_space, s̄)

    # Get key values.
    n_states = size(state_space, 1)
    n_choices = size(s̄, 1) + 1
    
    # Arrays to hold results.
    State_Correspondence = zeros(Int64, n_states, n_states)
    Store_Choice = zeros(Int64, n_states, n_states)
    Gift_Earned = zeros(Float64, n_states, n_states)

    # Go through all possible starting and ending states.
    for col in eachindex(State_Correspondence[:, 1])
        for row in eachindex(State_Correspondence[1, :])
            
            # Get the starting and ending states at these indices.
            start_state = state_space[row]
            end_state = state_space[col]

            # Check if the outside good was chosen (no change in state).
            if (start_state == end_state)
                State_Correspondence[row, col] = col
                Store_Choice[row, col] = n_choices  # assign last index
                Gift_Earned[row, col] = 0
                continue
            end

            # Otherwise, go through each index to see if one, and only one, changed by 1 unit or reset due to gift.
            store_index = -1
            gift_earned = 0

            # For each index/store.
            for j in eachindex(start_state)
                
                # Calculate how much the store's state changed.
                store_change = end_state[j] - start_state[j]

                # If the index resets due to gift earned, for the first time, flag this index.
                if (-store_change == s̄[j] - 1) && (gift_earned != 1) && (store_index == -1)
                    gift_earned = 1
                    store_index = j

                # If the state changes by exactly 1, for the first time, flag this index.
                elseif (store_change == 1) && (store_index == -1)
                    store_index = j
                
                # If the index does not change, continue to the next index.
                elseif (store_change == 0)
                    continue
                    
                # If the index changes by any other amount or more than once, stop, because this is not permitted.
                else
                    store_index = -1 
                    break
                end
            end

            # If the state changed in a permitted way, record the result.
            if store_index != -1
                State_Correspondence[row, col] = col
                Store_Choice[row, col] = store_index
                Gift_Earned[row, col] = gift_earned

            # Otherwise, record a non-permitted transition.
            else 
                State_Correspondence[row, col] = 0
                Store_Choice[row, col] = 0
                Gift_Earned[row, col] = 0
            end
        end
    end

    # Additional step: rather than returning matrices that are (n_state)x(n_states),
    # remove unnecessary columns with 0s corresponding to unavailable states. 
    # Return matrix that is (n_state)x(n_choices) since the number of choices is the same for each state.
    
    # Arrays to hold filtered results.
    State_Correspondence_Filtered = zeros(Int64, n_states, n_choices)
    Store_Choice_Filtered = zeros(Int64, n_states, n_choices)
    Gift_Earned_Filtered = zeros(Float64, n_states, n_choices)
    
    # Go through each row and get the non-zero columns.
    for starting_state in eachindex(state_space)
        
        # Get the indices of non-zero columns, corresponding to permitted state choices.
        state_choices = @views State_Correspondence[starting_state, :][State_Correspondence[starting_state, :] .!= 0]

        # Assign the values corresponding to permitted states.
        State_Correspondence_Filtered[starting_state, :] = state_choices
        Store_Choice_Filtered[starting_state, :] = Store_Choice[starting_state, state_choices]
        Gift_Earned_Filtered[starting_state, :] = Gift_Earned[starting_state, state_choices]
    end

    return State_Correspondence_Filtered, Store_Choice_Filtered, Gift_Earned_Filtered
end

# --------------------------------------------------------------------------------
# 2. Solve the individual choice problem for a given starting state.
# --------------------------------------------------------------------------------
function Solve_Individual_Problem(individual_index, starting_state_index, V, Model_Parameters, Environment)

    # Unpack parameters.
    @unpack α, γ, Gᵢ, β = Model_Parameters
    @unpack n_choices, s̄, n_price_draws, Price_Draws, 
    State_Correspondence, Store_Choice, Gift_Earned = Environment

    # Set up individual parameters.
    #Gᵢ = [G[1], G[2], G3ᵢ[individual_index]]

    # Get the set of available state/store choices and corresponding gift payouts and prices.
    state_choices = @views State_Correspondence[starting_state_index, :]
    store_choices = @views Store_Choice[starting_state_index, :]
    gifts_earned  = @views Gift_Earned[starting_state_index, :]

    # Variable to store the value function over all price draws.
    Eₚᵢ = 0.0

    for price_draw in eachrow(Price_Draws)

        # Variable to store the value of the alternative-specific value functions.
        Emaxᵢ = 0.0
        
        # Go through each allowed next state.
        for choice_index in eachindex(state_choices)

            # Get the corresponding product, state choice, and whether gift is earned.
            j = store_choices[choice_index]
            next_state = state_choices[choice_index]
            gift_earned = gifts_earned[choice_index]

            # If the choice is the outside option, use default parameter values.
            if j == n_choices
                Vⱼ = 0.0 + γ*price_draw[j] + 0.0 + β*V[next_state]
            
            # If the choice is an inside good, use product utility parameters.
            else 
                Vⱼ = α[j] + γ*price_draw[j] + Gᵢ[j, individual_index]*gift_earned + β*V[next_state]
            end

            # Tabulate the exponential value for this alternative.
            Emaxᵢ += @fastmath exp(Vⱼ)
        end

        # Take the log over each alternative for this price draw.
        Eₚᵢ += @fastmath log(Emaxᵢ)
    end

    # Return the average expected value over all price draws.
    Eₚ = Eₚᵢ / n_price_draws
    return Eₚ
end

# --------------------------------------------------------------------------------
# 3. Solve the Value Funcion using Solve_Individual_Problem.
# --------------------------------------------------------------------------------
function Solve_Value_Function(individual_index, Model_Parameters, Environment)

    # Unpack relevant parameters.
    state_space = Environment.state_space

    # Iteration parameters
    diff = Inf
    tolerance = 1e-5
    iteration_count = 0
    max_iterations = Int(1e5)

    Vⁿ⁺¹ = zeros(size(state_space, 1))
    Vⁿ = zeros(size(state_space, 1))

    while ((diff > tolerance) && (iteration_count < max_iterations))
    
        # Update the Value Function for each possible starting state.
        for state_index in eachindex(state_space)                    # single-threaded implementation
        #Threads.@threads for state_index in eachindex(state_space)  # multi-threaded implementation
            Vⁿ⁺¹[state_index] = Solve_Individual_Problem(individual_index, state_index, Vⁿ, Model_Parameters, Environment)
        end

        # Update iteration parameters.
        diff = maximum(abs.(Vⁿ⁺¹ .- Vⁿ))
        Vⁿ = deepcopy(Vⁿ⁺¹) 
        # for i in eachindex(Vⁿ⁺¹); Vⁿ[i] = Vⁿ⁺¹[i] end # fewer allocations that previous line.
        iteration_count += 1
        
        # Display progress.
        if iteration_count % 1000 == 0
            #println(diff)
        end
        if iteration_count >= max_iterations
            println("Did not converge after $iteration_count iterations. Distance is $diff")
        end
        if diff <= tolerance
            #println("Converged to distance $diff after $iteration_count iterations.")
        end
    end

    return Vⁿ⁺¹
end

# --------------------------------------------------------------------------------
# 4. Proposal distribution: get random parameter draw.
# --------------------------------------------------------------------------------
# Version 1: for drawing α, γ, β paremters (homogeneous parameters).
function Get_Proposal(Model_Parameters, ρ)

    @unpack α, γ, G, σg, Gᵢ, β = Model_Parameters

    # Random walk with adjustable st.dev of ρ. Fixed value of 0.05 is used in T2 slide 35.
    Proposed_Parameters = Model_Parameters_Struct(
        α  .+ rand(Normal(0, ρ), 3),
        γ   + rand(Normal(0, ρ)),
        deepcopy(G),
        deepcopy(σg),
        Gᵢ,
        β   + rand(Normal(0, ρ))
    )
    return Proposed_Parameters
end

# Version 2: for drawing indiviudal Gᵢ (heterogeneous parameter).
function Get_Proposal(Model_Parameters, ρ, individual_index)

    @unpack G, σg, Gᵢ = Model_Parameters
    G_ind = @views Gᵢ[:,individual_index] # Get the G values for given individual

    # Random walk with adjustable st.dev of ρ. Fixed value of 0.05 is used in T2 slide 35.
    G_ind_proposal = G_ind .+ rand(MvNormal(zeros(size(G)), σg*ρ^2))

    # Random walk with adjustable st.dev of ρ. Fixed value of 0.05 is used in T2 slide 35.
    #L = cholesky(σg)
    #G̃ᵢ = G_ind .+ (ρ * Matrix(L) * rand(Normal(0,1), size(G)))

    return G_ind_proposal
end

# --------------------------------------------------------------------------------
# 5. Get the distance between two parameter draws.
# --------------------------------------------------------------------------------
function Parameter_Distance_Kernel(θʳ, θˡ, individual_index, H)

    # Choice of bandwidth. 
    #h = 0.01 # value in example on T2 slide 35 is 0.01
    #H = fill(h, 4)
    #H = [0.01, 0.01, 0.01, 0.01]

    # Unpack individual-level model parameters.
    @unpack α, γ, Gᵢ, β = θʳ
    αʳ = α
    γʳ = γ
    Gʳ = @views Gᵢ[:, individual_index]
    βʳ = β

    @unpack α, γ, Gᵢ, β = θˡ
    αˡ = α
    γˡ = γ
    Gˡ = @views Gᵢ[:, individual_index]
    βˡ = β

    # Track total distance.
    dist = 0.0
    H_index = 1

    # Calculate distance for each variable.
    for i in eachindex(αʳ)
        dist += (αʳ[i] - αˡ[i])^2 / H[H_index]
    end
    H_index += 1

    dist += (γʳ - γˡ)^2 / H[H_index]
    H_index += 1

    for i in eachindex(Gʳ)
        dist += (Gʳ[i] - Gˡ[i])^2 / H[H_index]
    end
    H_index += 1

    dist += (βʳ - βˡ)^2 / H[H_index]
    H_index += 1

    weighted_dist = @fastmath exp(-0.5*dist)
    smallest_float = eps(0.0)
    return max(weighted_dist, smallest_float)   # return small non-zero float if weighted_dist is ≈0
end

# --------------------------------------------------------------------------------
# 6. Get the log likelihood for a given set of (individual) parameters.
# --------------------------------------------------------------------------------
# Version 1: for drawing α, γ, β paremters (homogeneous parameters).
function Log_Prior(Model_Parameters)

    @unpack α, γ, β = Model_Parameters
    Log_Prθ = 0.0

    # Get log likelihood for each parameter
    for αⱼ in α
        Log_Prθ += logpdf(Normal(0,1), αⱼ)
    end
    Log_Prθ += logpdf(Normal(-1, 1), γ)
    Log_Prθ += logpdf(Normal(0.75, 1), β)

    return Log_Prθ
end

# Version 2: for drawing indiviudal Gᵢ (heterogeneous parameter).
function Log_Prior(Model_Parameters, individual_index)

    @unpack G, σg, Gᵢ = Model_Parameters
    G_ind = @views Gᵢ[:,individual_index] # Get the G values for given individual
    Log_Prθ = 0.0

    # Get log likelihood for this individual based on population G and σg.
    Log_Prθ += logpdf(MvNormal(G, σg), G_ind)

    return Log_Prθ
end

# --------------------------------------------------------------------------------
# 7. Calculate the sample log likelihood at a certain parameter set.
# --------------------------------------------------------------------------------
# Version 1: likelihood for the entire dataset.
function Log_Sample_Likelihood(Dataset, Environment, Model_Parameters, Value_Function_Parameters)

    # Unpack dataset and model environment parameters.
    @unpack Starting_State, State_Chosen, Store_Chosen, Observed_Prices = Dataset
    @unpack α, γ, Gᵢ, β = Model_Parameters
    @unpack n_choices, s̄, state_space, State_Correspondence, 
    Store_Choice, Gift_Earned = Environment

    n_individuals, n_periods = size(Starting_State)

    # Variable to track total sample log likelihood.
    #ℓ = 0.0
    ℓ = zeros(n_individuals) # store likelihoods from each thread in a vector

    # Calculate the likelihood of each choice.
    Threads.@threads for n in 1:n_individuals
    #for n in 1:n_individuals

        # Get pseudo value function according to this individual's parameters.
        Ṽ = Get_Pseudo_Value_Function(n, Model_Parameters, Value_Function_Parameters)
        #Ṽ = Solve_Value_Function(n, Model_Parameters, Environment)

        # Loop over all choices for this individual.
        for t in 1:n_periods

            # Get starting state, store choices, chosen store, and the prices for each choice.
            starting_state_index = Starting_State[n, t]
            store_chosen = Store_Chosen[n, t]

            # Get the set of available state/store choices and corresponding gift payouts and prices.
            state_choices =  @views State_Correspondence[starting_state_index, :]
            store_choices =  @views Store_Choice[starting_state_index, :]
            gifts_earned  =  @views Gift_Earned[starting_state_index, :]
            prices =         @views Observed_Prices[n, t, :]

            # Calculate the value for each (pseudo) alternative-specific value function.
            numerator = 0.0
            denominator = 0.0

            # Go though each allowed next state.
            for choice_index in eachindex(state_choices)

                # Get the corresponding product, state choice, and whether gift is earned.
                j = store_choices[choice_index]
                next_state = state_choices[choice_index]
                gift_earned = gifts_earned[choice_index]

                # If the choice is the outside option, use default parameter values.
                if j == n_choices
                    Vⱼ = 0.0 + γ*prices[j] + 0.0 + β*Ṽ[next_state]
                # If the choice is an inside good, use product utility parameters.
                else 
                    Vⱼ = α[j] + γ*prices[j] + Gᵢ[j,n]*gift_earned + β*Ṽ[next_state]
                end

                # Collect part of the logit choice probability values: Pr(choose j) = exp(Vⱼ) / ∑ₖexp(Vₖ)
                exp_Vⱼ = @fastmath exp(Vⱼ)
                denominator += exp_Vⱼ
                if j == store_chosen
                    numerator += exp_Vⱼ
                end

            end

            # Calculate the likelihood for the individual choice.
            # Pr(choose j) = exp(Vⱼ) / ∑ₖexp(Vₖ)
            choice_likelihood = numerator / denominator

            # Add to total log likelihood value.
            ℓ[n] += @fastmath log(choice_likelihood)
        end
    end

    #return ℓ
    return sum(ℓ) # return sum of likelihoods across threads
end

# Version 2: likelihood for a single individual's observations.
function Log_Sample_Likelihood(Dataset, Environment, Model_Parameters, 
                                Value_Function_Parameters, individual_index)

    # Unpack dataset and model environment parameters.
    @unpack Starting_State, State_Chosen, Store_Chosen, Observed_Prices = Dataset
    @unpack α, γ, Gᵢ, β = Model_Parameters
    @unpack n_choices, s̄, state_space, State_Correspondence, 
    Store_Choice, Gift_Earned = Environment

    n_individuals, n_periods = size(Starting_State)

    # Variable to track individual sample log likelihood.
    ℓ = 0.0

    # Get pseudo value function according to this individual's parameters.
    Ṽ = Get_Pseudo_Value_Function(individual_index, Model_Parameters, Value_Function_Parameters)
    #Ṽ = Solve_Value_Function(individual_index, Model_Parameters, Environment)

    # Loop over all choices for this individual.
    for t in 1:n_periods

        # Get starting state, store choices, chosen store, and the prices for each choice.
        starting_state_index = Starting_State[individual_index, t]
        store_chosen = Store_Chosen[individual_index, t]

        # Get the set of available state/store choices and corresponding gift payouts and prices.
        state_choices =  @views State_Correspondence[starting_state_index, :]
        store_choices =  @views Store_Choice[starting_state_index, :]
        gifts_earned  =  @views Gift_Earned[starting_state_index, :]
        prices =         @views Observed_Prices[individual_index, t, :]

        # Calculate the value for each (pseudo) alternative-specific value function.
        numerator = 0.0
        denominator = 0.0

        # Go though each allowed next state.
        for choice_index in eachindex(state_choices)

            # Get the corresponding product, state choice, and whether gift is earned.
            j = store_choices[choice_index]
            next_state = state_choices[choice_index]
            gift_earned = gifts_earned[choice_index]

            # If the choice is the outside option, use default parameter values.
            if j == n_choices
                Vⱼ = 0.0 + γ*prices[j] + 0.0 + β*Ṽ[next_state]
            # If the choice is an inside good, use product utility parameters.
            else 
                Vⱼ = α[j] + γ*prices[j] + Gᵢ[j,individual_index]*gift_earned + β*Ṽ[next_state]
                #Vⱼ = α[j] + γ*prices[j] + [Gᵢ[1,5], 8.5, Gᵢ[3,5]][j]*gift_earned + β*Ṽ[next_state]
                #Vⱼ = α[j] + γ*prices[j] + [Gᵢ[1,5], 8.5, Gᵢ[3,5]][j]*gift_earned + β*Ṽ[next_state]
            end

            # Collect part of the logit choice probability values: Pr(choose j) = exp(Vⱼ) / ∑ₖexp(Vₖ)
            exp_Vⱼ = @fastmath exp(Vⱼ)
            denominator += exp_Vⱼ
            if j == store_chosen
                numerator += exp_Vⱼ
            end

        end

        # Calculate the likelihood for the individual choice.
        # Pr(choose j) = exp(Vⱼ) / ∑ₖexp(Vₖ)
        #choice_likelihood = numerator / denominator

        # Add to total log likelihood value.
        #ℓ += @fastmath log(choice_likelihood)
        ℓ += @fastmath log(numerator) - log(denominator)
    end

    return ℓ
end

# --------------------------------------------------------------------------------
# 8. Get the current pseudo value function at a given parameter.
# --------------------------------------------------------------------------------
function Get_Pseudo_Value_Function(individual_index, Model_Parameters, Value_Function_Parameters)

    # Unpack key variables.
    @unpack Ṽₙ, θₙ, H = Value_Function_Parameters

    # Variable set up.
    Ñ = size(Ṽₙ, 1)          # number of states
    Ṽ = zeros(size(Ṽₙ, 2))   # empty value function
    weights = zeros(Ñ)      # vector to store parameter weights

    # Calculate distance/weight for each corresponding parameter set for given individual.
    for ñ in eachindex(weights)
        weights[ñ] = Parameter_Distance_Kernel(Model_Parameters, θₙ[ñ], individual_index, H)
    end

    # Normalize weights to sum to 1.
    #weights = weights ./ sum(weights)
    sum_weights = sum(weights)
    for i in eachindex(weights)
        weights[i] = weights[i] / sum_weights
    end 

    # Calculate value function for each state as a weighted average of previous values.
    for state_index in eachindex(Ṽ)        
        Ṽ_new = 0.0
        for ñ in 1:Ñ
            Ṽ_new += Ṽₙ[ñ, state_index, individual_index] * weights[ñ]
        end
        Ṽ[state_index] = Ṽ_new
    end

    return Ṽ
end

end # end module
