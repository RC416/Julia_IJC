# Data structs used in custom_functions and main.
module data_structs
export Model_Parameters_Struct, Environment_Struct, Dataset_Struct

using Parameters

# Struct to store true values and drawn values of model parameters.
@with_kw struct Model_Parameters_Struct
    α::Vector{Float64}
    γ::Float64
    G::Vector{Float64}
    σg::Vector{Float64}
    β::Float64
end

# Struct to store key environmental values.
@with_kw struct Environment_Struct{n_stores}
    #n_stores::Int64
    n_choices::Int64
    s̄::Vector{Int64}
    n_price_draws::Int64
    Price_Draws::Matrix{Float64}
    state_space::Vector{NTuple{n_stores, Int64}}
    State_Correspondence::Matrix{Int64}
    Store_Choice::Matrix{Int64}
    Gift_Earned::Matrix{Float64}
end

# Struct to store simulated datasets.
@with_kw struct Dataset_Struct
    Starting_State::Array{Int64, 2}
    State_Chosen::Array{Float64, 2}
    Store_Chosen::Array{Float64, 2}
    Observed_Prices::Array{Float64, 3}
end

end # end module