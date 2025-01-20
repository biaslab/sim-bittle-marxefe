module NARXAgents

using JLD 
using Optim
using Distributions
using LinearAlgebra
using SpecialFunctions

export NARXAgent, update!, predictions, pol, crossentropy, mutualinfo, minimizeEFE, minimizeMSE, backshift, update_goals!, reset_buffer, get_goals, loadModel, saveModel, getAgentDict, isNARXAgent


mutable struct NARXAgent
    """
    Active inference agent based on a Nonlinear Auto-Regressive eXogenous model.

    Parameters are inferred through Bayesian filtering and controls through minimizing expected free energy.
    """

    ybuffer         ::Vector{Float64}
    ubuffer         ::Vector{Float64}
    delay_inp       ::Integer
    delay_out       ::Integer
    pol_degree      ::Integer
    order           ::Integer

    μ               ::Vector{Float64}   # Coefficients mean
    Λ               ::Matrix{Float64}   # Coefficients precision
    α               ::Float64           # Likelihood precision shape
    β               ::Float64           # Likelihood precision rate
    λ               ::Float64           # Control prior precision

    goals           ::Union{Distribution{Univariate, Continuous}, Vector}
    thorizon        ::Integer
    num_iters       ::Integer

    free_energy     ::Float64

    function NARXAgent(coefficients_mean,
                       coefficients_precision,
                       noise_shape,
                       noise_rate; 
                       goal_prior=Normal(0.0, 1.0),
                       delay_inp::Integer=1, 
                       delay_out::Integer=1, 
                       pol_degree::Integer=1,
                       time_horizon::Integer=1,
                       num_iters::Integer=10,
                       control_prior_precision::Float64=0.0,
                       )

        ybuffer = zeros(delay_out)
        ubuffer = zeros(delay_inp+1)

        order = size(pol(zeros(1 + delay_inp + delay_out), degree=pol_degree),1)
        if order != length(coefficients_mean) 
            error("Dimensionality of coefficients and model order do not match.")
        end

        free_energy = Inf

        return new(ybuffer,
                   ubuffer,
                   delay_inp,
                   delay_out,
                   pol_degree,
                   order,
                   coefficients_mean,
                   coefficients_precision,
                   noise_shape,
                   noise_rate,
                   control_prior_precision,
                   goal_prior,
                   time_horizon,
                   num_iters,
                   free_energy)
    end
end


function getAgentDict(data::Dict{String,Any})
    agentdict = Dict{String, NARXAgent}()
    set_bool = false
    for (k, v) in data
        if isNARXAgent(v)
            set_bool = true
            agentdict[k] = v
        end
    end

    # add feature to convert the dict to a Dict{String, Dict} if set_bool
    if !set_bool
        agentdict = convertDict(data)
    end

    return agentdict
end


function convertDict(data::Dict{String, Any})
    new_data = Dict{String, Dict}()
    for (key, value) in data
        new_data[key] = value
    end
    return new_data
end


pol(x; degree::Integer = 1) = cat([1.0; [x.^d for d in 1:degree]]...,dims=1)

function update!(agent::NARXAgent, y::Float64, u::Float64)

    agent.ubuffer = backshift(agent.ubuffer, u)
    ϕ = pol([agent.ybuffer; agent.ubuffer], degree=agent.pol_degree)

    μ0 = agent.μ
    Λ0 = agent.Λ
    α0 = agent.α
    β0 = agent.β

    agent.μ = inv(ϕ*ϕ' + Λ0)*(ϕ*y + Λ0*μ0)
    agent.Λ = ϕ*ϕ' + Λ0
    agent.α = α0 + 1/2
    agent.β = β0 + 1/2*(y^2 + μ0'*Λ0*μ0 - (ϕ*y + Λ0*μ0)'*inv(ϕ*ϕ' + Λ0)*(ϕ*y + Λ0*μ0))

    agent.ybuffer = backshift(agent.ybuffer, y)

    agent.free_energy = -log(marginal_likelihood(agent, (μ0, Λ0, α0, β0)))
end

function params(agent::NARXAgent)
    return agent.μ, agent.Λ, agent.α, agent.β
end

function marginal_likelihood(agent::NARXAgent, prior_params)

    μn, Λn, αn, βn = params(agent)
    μ0, Λ0, α0, β0 = prior_params

    return (det(Λn)^(-1/2)*gamma(αn)*βn^αn)/(det(Λ0)^(-1/2)*gamma(α0)*β0^α0) * (2π)^(-1/2)
end

function posterior_predictive(agent::NARXAgent, ϕ_t)
    "Posterior predictive distribution is location-scale t-distributed"

    ν_t = 2*agent.α
    m_t = dot(agent.μ, ϕ_t)
    s2_t = agent.β/agent.α*(1 + ϕ_t'*inv(agent.Λ)*ϕ_t)

    return ν_t, m_t, s2_t
end

function predictions(agent::NARXAgent, controls; time_horizon=1)
    
    m_y = zeros(time_horizon)
    v_y = zeros(time_horizon)

    ybuffer = agent.ybuffer
    ubuffer = agent.ubuffer
    
    for t in 1:time_horizon
        
        # Update control buffer
        ubuffer = backshift(ubuffer, controls[t])
        ϕ_t = pol([ybuffer; ubuffer], degree=agent.pol_degree)

        ν_t, m_t, s2_t = posterior_predictive(agent, ϕ_t)
        
        # Prediction
        m_y[t] = m_t
        v_y[t] = s2_t * ν_t/(ν_t - 2)
        
        # Update previous 
        ybuffer = backshift(ybuffer, m_y[t])
        
    end
    return m_y, v_y
end

function mutualinfo(agent::NARXAgent, ϕ)
    "Mutual information between parameters and posterior predictive (constant terms dropped)"
    return -1/2*log( agent.β/agent.α*(1 + ϕ'*inv(agent.Λ)*ϕ) )
end

function crossentropy(agent::NARXAgent, goal::Distribution{Univariate, Continuous}, m_pred, v_pred)
    "Cross-entropy between posterior predictive and goal prior (constant terms dropped)"  
    return ( v_pred + (m_pred - mean(goal))^2 ) / ( 2var(goal) )
    # return (m_pred - mean(goal))^2/(2var(goal))
end 

function EFE(agent::NARXAgent, goals, controls)
    "Expected Free Energy"

    ybuffer = agent.ybuffer
    ubuffer = agent.ubuffer
    
    J = 0
    for t in 1:agent.thorizon
        
        # Update control buffer
        ubuffer = backshift(ubuffer, controls[t])
        ϕ_t = pol([ybuffer; ubuffer], degree=agent.pol_degree)

        # Prediction
        ν_t, m_t, s2_t = posterior_predictive(agent, ϕ_t)
        
        m_y = m_t
        v_y = s2_t * ν_t/(ν_t - 2)
        
        # Accumulate EFE
        J += mutualinfo(agent, ϕ_t) + crossentropy(agent, goals[t], m_y, v_y) + agent.λ*controls[t]^2
        
        # Update previous 
        ybuffer = backshift(ybuffer, m_y)        
    end
    return J
end

function MSE(agent::NARXAgent, goals, controls)
    "Mean Squared Error between prediction and setpoint."

    ybuffer = agent.ybuffer
    ubuffer = agent.ubuffer
    
    J = 0
    for t in 1:agent.thorizon
        
        # Update control buffer
        ubuffer = backshift(ubuffer, controls[t])
        ϕ_t = pol([ybuffer; ubuffer], degree=agent.pol_degree)
        
        # Prediction
        m_y = dot(agent.μ, ϕ_t)
        
        # Accumulate objective function
        J += (mean(goals[t]) - m_y)^2 + agent.λ*controls[t]^2
        
        # Update previous 
        ybuffer = backshift(ybuffer, m_y)        
    end
    return J
end

function minimizeEFE(agent::NARXAgent, goals; u_0=nothing, time_limit=10, verbose=false, control_lims::Tuple=(-Inf,Inf))
    "Minimize EFE objective and return policy."

    if isnothing(u_0); u_0 = 1e-8*randn(agent.thorizon); end
    opts = Optim.Options(time_limit=time_limit, 
                         show_trace=verbose, 
                         allow_f_increases=true, 
                         g_tol=1e-12, 
                         show_every=10,
                         iterations=10_000)

    # Objective function
    J(u) = EFE(agent, goals, u)

    # Constrained minimization procedure
    results = optimize(J, control_lims..., u_0, Fminbox(LBFGS()), opts, autodiff=:forward)

    return Optim.minimizer(results)
end

function minimizeMSE(agent::NARXAgent, goals; u_0=nothing, time_limit=10, verbose=false, control_lims::Tuple=(-Inf,Inf))
    "Minimize MSE objective and return policy."

    if isnothing(u_0); u_0 = 1e-8*randn(agent.thorizon); end
    opts = Optim.Options(time_limit=time_limit, 
                         show_trace=verbose, 
                         allow_f_increases=true, 
                         g_tol=1e-12, 
                         show_every=10,
                         iterations=10_000)

    # Objective function
    J(u) = MSE(agent, goals, u)

    # Constrained minimization procedure
    results = optimize(J, control_lims..., u_0, Fminbox(LBFGS()), opts, autodiff=:forward)

    return Optim.minimizer(results)
end


"""
Shift elements down and element to first position

Args:  
    x (AbstractVector): Vector to shift  
    a (Number|AbstractVector): Element to add to first position  
    
Returns:  
    x (AbstractVector): Shifted vector with element added to first position
"""
function backshift(x::AbstractVector, a::Number)
    "Shift elements down and add element"

    N = size(x,1)

    # Shift operator
    S = Tridiagonal(ones(N-1), zeros(N), zeros(N-1))

    # Basis vector
    e = [1.0; zeros(N-1)]
    
    return S*x + e*a
end

function backshift(x::AbstractVector, a::AbstractVector)::AbstractVector
    "Shift elements down and add element to first position"
    circshift!(x, 1)
    x[1] = a
    return x
end

function backshift(x::Matrix, a::AbstractVector)::Matrix 
    "Shift elements down and add element to first position"
    x = [x[i, :] for i in 1:size(x,1)]
    x = backshift(x, a)
    x = mapreduce(permutedims, vcat, x)
    return x
end

function update_goals!(x::AbstractVector, g::Distribution{Univariate, Continuous})
    "Move goals forward and add a final goal"
    circshift!(x,-1)
    x[end] = g
end

"Reset buffers of a NARXAgent"
function reset_buffer(agent::NARXAgent)

    if isa(agent, NARXAgent)
        agent.ubuffer = zeros(agent.delay_inp + 1)
        agent.ybuffer = zeros(agent.delay_out)
        agent.free_energy = Inf
    end
    return agent
end

function reset_buffer(agent::Dict{String, NARXAgent})
    for (k, v) in agent

        v.ubuffer = zeros(v.delay_inp + 1)
        v.ybuffer = zeros(v.delay_out)
        v.free_energy = Inf
    end
    return agent
end

function reset_buffer(agent::Dict{String, Dict})
    for (k, v) in agent

        if v["agent"] isa NARXAgent
            v["ubuffer"] = zeros(v["agent"].delay_inp + 1)
            v["ybuffer"] = zeros(v["agent"].delay_out)
            v["agent"].free_energy = Inf
        end
    end
    return agent
end

"Save model to file"
function saveModel(
        model::NARXAgent,
        filename::String="models/model.jld",
        makedir::Bool=false
    )

    if ~endswith(filename, ".jld")
        @warn "Filename does not end with .jld. Appending .jld to filename"
        filename = "$filename.jld"
    end

    directory = join(split(filename, "/")[1:end-1])
    if ~isdir(directory) && directory != ""

        if makedir
            warn("Directory $directory does not exist. Creating directory")
            mkdir(directory)
        else
            error("Directory $directory does not exist. Model cannot be saved to $filename")
        end	
    end
    JLD.save(filename, "model", model)
    @info "Model saved to $filename"
end

"Load model from file"
function loadModel(file::String)
    isfile(file) || error("File does not exist.")

    data = JLD.load(file)

    if isNARXAgent(data)
        return data
    elseif isa(data, Dict)
        return getAgentDict(data)
    else
        error("No NARXAgent found in file, this might be a unsupported data type.")
    end
end

end
