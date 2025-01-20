module arguments

export Arguments
"Arguments for experiment manager"
mutable struct Arguments
    algo::String
    env::String
    tensorboard_log::String
    eval_freq::Int32
    trained_agent::String
    truncate_last_trajectory::Bool
    n_timesteps::Int32
    num_threads::Int32
    log_interval::Int32
    optimization_log_path::Union{String,Nothing}
    eval_episodes::Int32
    n_eval_envs::Int32
    save_freq::Int32
    save_replay_buffer::Bool
    log_folder::String
    seed::Int32
    vec_env::String
    device::String
    n_trials::Int32
    max_total_trials::Union{Nothing, Int32}
    optimize_hyperparameters::Bool
    no_optim_plots::Union{Nothing, Bool}
    n_jobs::Int32
    sampler::String
    pruner::String
    n_startup_trials::Int32
    n_evaluations::Union{Nothing, Int32}
    storage::Nothing
    study_name::Nothing
    verbose::Int32
    gym_packages::Vector
    env_kwargs::Dict
    eval_env_kwargs::Dict
    hyperparams::Dict
    conf_file::String
    uuid::Bool
    track::Bool
    wandb_project_name::String
    wandb_entity::Nothing 
    progress::Bool
    tags::Vector

    "Set default parameters for Arguments"
    function Arguments(
        algo = "sac",
        env = "BittleBulletEnv-v0",
        tensorboard_log = "dtblog",
        eval_freq = 25000,
        trained_agent = string(""),
        truncate_last_trajectory = true,
        n_timesteps = 500000,
        num_threads = -1,
        log_interval = -1,
        optimization_log_path = string(""),
        eval_episodes = 5,
        n_eval_envs = 1,
        save_freq = -1,
        save_replay_buffer = false,
        log_folder = "logs",
        seed = -1,
        vec_env = "dummy",
        device = "auto",
        n_trials = 500,
        max_total_trials=nothing,
        optimize_hyperparameters = false,
        no_optim_plots = nothing,
        n_jobs=1,
        sampler="tpe",
        pruner = "median",
        n_startup_trials = 10,
        n_evaluations = nothing,
        storage = nothing,
        study_name = nothing,
        verbose = 1,
        gym_packages = [],
        env_kwargs = Dict(),
        eval_env_kwargs = Dict(),
        hyperparams = Dict(),
        conf_file = "DiederikScripts/hyperparams/OGsac.yml",
        uuid = false,
        track = false,
        wandb_project_name = "sb3",
        wandb_entity = nothing,
        progress = true,
        tags = [])
        return new(
            algo,
            env,
            tensorboard_log,
            eval_freq,
            trained_agent,
            truncate_last_trajectory,
            n_timesteps,
            num_threads,
            log_interval,
            optimization_log_path,
            eval_episodes,
            n_eval_envs,
            save_freq,
            save_replay_buffer,
            log_folder,
            seed,
            vec_env,
            device,
            n_trials,
            max_total_trials,
            optimize_hyperparameters,
            no_optim_plots,
            n_jobs,
            sampler,
            pruner,
            n_startup_trials,
            n_evaluations,
            storage,
            study_name,
            verbose,
            gym_packages,
            env_kwargs,
            eval_env_kwargs,
            hyperparams,
            conf_file,
            uuid,
            track,
            wandb_project_name,
            wandb_entity,
            progress,
            tags
        )
    end
end

end # module