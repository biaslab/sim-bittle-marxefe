module JlGym

using JSON
using Plots
using PyCall
using Distributions
using LinearAlgebra
gym = pyimport("gymnasium")

export random_uniform, random_uniform_sample, getactionspace, ParameterManager, check_modeldict

# deprecated
"""Parameter manager for NARXAgent"""
mutable struct ParameterManager

    # gym environment parameters
    env::PyCall.PyObject                # gym environment
    sys_ulims::Tuple{Float64, Float64}  # system limits
    
    # model parameters (NARX-EFE)
    H::Int64                            # model order
    Ly::Int64                           # number of outputs
    Lu::Int64                           # number of inputs
    α0::Float64                         # initial value of α
    β0::Float64                         # initial value of β
    λ::Float64                          # forgetting factor[lr?]
    T::Int64                            # time horizon
    M::Int64                            # number of basis functions
    μ0::Vector{Float64}                 # initial mean of the coefficients
    Λ0::Matrix{Float64}                 # initial precision of the coefficients
    
    # agent parameters (time)
    N::Int64                            # number of steps to run the environment for
    Δt::Float64                         # time step to run the environment for
    Tsteps::Vector{Float64}             # time steps
    cut_gvar::Float64                   # cut goal variance
    new_gvar::Float64                   # new goal variance
    
    # goal
    goals::Vector{Normal{Float64}}      # goal to be reached by the agent
    
    # plotting data (general storage)
    y_EFE::Vector{Float64}              # output of the environment
    z_EFE::Matrix{Float64}              # observation of the environment
    u_EFE::Vector{Float64}              # action of the environment
    reward_EFE::Vector{Float64}         # reward of the environment
    μ_EFE::Vector{Vector{Float64}}      # mean of the coefficients
    Λ_EFE::Vector{Matrix{Float64}}      # precision of the coefficients
    α_EFE::Vector{Float64}              # α storage
    β_EFE::Vector{Float64}              # β storage
    F_EFE::Vector{Float64}              # free energy storage
    pred_m::Matrix{Float64}             # prediction mean storage
    pred_v::Matrix{Float64}             # prediction variance storage

    """Constructor for ParameterManager

    Args:  
        - env_par (array): environment parameters  
        - env_kwargs (dict): environment kwargs  
        - agent_parameters (dict): agent parameters  
        - Steps (int): number of steps to run the environment for  
        - TimeStep (float): time step to run the environment for  
        - exploit (bool): whether to use the agent in exploit mode or not  
        - ExploitSteps (int): number of steps to run the environment for in exploit mode  
        - goal (array): goal to be reached by the agent
    """
    function ParameterManager(
        env_par::Vector{String},
        env_kwargs::Union{Dict{String, Float64}, Nothing},
        agent_parameters::Union{Dict{String, Real}, Nothing},
        Steps::Int64,
        TimeStep::Float64,
        exploit::Bool,
        ExploitSteps::Int64,
        goal::Vector
        )

        # setup gym environment parameters
        env_folder, env_name = env_par
        env_kwargs = env_kwargs

        # load and check model parameters
        modeldict = check_modeldict(agent_parameters, true)
        H = modeldict.H
        Ly = modeldict.Ly
        Lu = modeldict.Lu
        α0 = modeldict.α0
        β0 = modeldict.β0
        λ = modeldict.λ
        T = modeldict.T
        M = size(pol(zeros(Lu + Ly + 1), H), 1);
        
        # setup other model priors 
        # Note that these prior are determining how certain you are about the state 
        μ0 = zeros(M)
        Λ0 = diagm(ones(M))

        # setup agent parameters (time)
        N = Steps
        Δt = TimeStep
        Tsteps = 0:Δt:Δt*(N-1)
        if exploit
            if ExploitSteps<Steps
                error("ExploitSteps must be less then (total) --Steps.")
            end
            cut_gvar = Tsteps[N - ExploitSteps]
            new_gvar = 1.
        else
            cut_gvar = Tsteps[N]
            new_gvar = 1.
        end

        goals = [Normal(goal[1], goal[2]) for t in 1:T]

        # setup gym env
        if ~isempty(env_folder)
            pushfirst!(pyimport("sys")."path", "")
            pyimport(env_folder)
            env = gym.make(env_name, max_episode_steps=N, env_kwargs=env_kwargs)
        else
            env = gym.make(env_name)
        end

        # preallocate data 
        y_EFE = zeros(N)
        z_EFE = zeros(env.observation_space.shape[1], N)
        u_EFE = zeros(N + T)
        reward_EFE = zeros(N)
        μ_EFE = [μ0]
        Λ_EFE = [Λ0]
        α_EFE = [α0]
        β_EFE = [β0]
        F_EFE = zeros(N)
        pred_m = zeros(N, T)
        pred_v = zeros(N, T);

        # Might be specific for the pendulum environment
        max_torque = env.unwrapped.max_torque
        sys_ulims = (-max_torque, max_torque)

        return new(
            env,
            sys_ulims,
            H,
            Ly,
            Lu,
            α0,
            β0,
            λ,
            T,
            M,
            μ0,
            Λ0,
            N,
            Δt,
            Tsteps,
            cut_gvar,
            new_gvar,
            goals,
            y_EFE,
            z_EFE,
            u_EFE,
            reward_EFE,
            μ_EFE,
            Λ_EFE,
            α_EFE,
            β_EFE,
            F_EFE,
            pred_m,
            pred_v
        )
    end

    
end

"""Sample from a uniform distribution

Args:  
    - n (int): number of samples  
    - lb (float): lower bound  
    - ub (float): upper bound  

Returns:  
    - array: array of samples
"""
function random_uniform(n, lb, ub)
    return (ub - lb) .* rand(n) .+ lb
end

"""
Random sample from a uniform distribution depending action space shape  

Args:  
    - lb (float): lower bound of the action space  
    - ub (float): upper bound of the action space  
    - shape (int): shape of the action space  
  
Returns:  
    - action_vec (array): array of samples  
"""
function random_uniform_sample(lb, ub, shape)
    if length(lb) == 1
        return random_uniform(shape, lb, ub)
    else
        action_vec = zeros(length(lb))
        for i in 1:eachindex(lb)
            action_vec[i] = random_uniform(i, lb[i], ub[i])
        end
        return action_vec
    end
end

"""Get the actionspace

Args:  
    - env (gym environment): gym environment  
    
Returns:
    - lb (float): lower bound of the action space  
    - ub (float): upper bound of the action space  
    - shape (int): shape of the action space
"""
function getactionspace(env)
    shape = length(env.action_space.low)
    
    # check if there are different actions or if it is a single action can be taken using random_uniform.
    lb = env.action_space.low
    ub = env.action_space.high
    # run_mult = true
    if all(env.action_space.low.==env.action_space.low[1])
        if all(env.action_space.high.==env.action_space.high[1])
            lb = env.action_space.low[1]
            ub = env.action_space.high[1]
            # run_mult = false
        end
    end
    return lb, ub, shape
end

"Check if modeldict is correctly defined for NARXEFEAgent, makes a NamedTuple"
function check_modeldict(modeldict::Union{Nothing, Dict}, default::Bool=true)::NamedTuple{(:Ly, :β0, :T, :α0, :λ, :Lu, :H),Tuple{Int64,Float64,Int64,Float64,Float64,Int64,Int64}}
    model_spec = Dict("H"=>1, "Ly"=>1, "Lu"=>1, "α0"=>1.0, "β0"=>1.0, "λ"=>1e-3, "T"=>1)

    if modeldict isa Nothing

        @info "No model dictionary provided, using default values."
        modeldict = copy(model_spec)
        return (; (Symbol(k) => v for (k,v) in modeldict)...)

    end
    # Ensure correct of dict
    convert(Dict{String, Real}, modeldict)

    if isempty(modeldict)

        error("No model dictionary provided.")
        
    end

    # check if all keys are provided
    if Set(collect(keys(model_spec))) <= Set(collect(keys(modeldict)))

        @debug "All keys are provided."
        return (; (Symbol(k) => v for (k,v) in modeldict)...)

    end

    # get set difference (NOTE only the values that are not in modeldict are returned)
    not_in = setdiff(collect(keys(model_spec)), collect(keys(modeldict)))

    if ~default

        error("The following key(s) is/are not provided: $(not_in).")

    end

    for key in not_in

        value = model_spec[key]
        @warn "The following key was not provided: $key, is now overwritten with default: $value."
        modeldict[key] = value

    end

    return (; (Symbol(k) => v for (k,v) in modeldict)...)
end

"""Polynomial basis function"""
function pol(x, degree::Integer = 1) 
    return cat([1.0; [x.^d for d in 1:degree]]...,dims=1)
end


end