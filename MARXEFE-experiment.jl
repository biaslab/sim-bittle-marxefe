@info "Loading Packages"

# julia packages
using Revise

using Dates
using JLD
using Dates
using Plots; default(grid=false, label="", linewidth=3,margin=20Plots.pt)
using Optim
using ForwardDiff
using Images
using PyCall
using ProgressMeter
using Distributions
using ProgressMeter
using LinearAlgebra
using DelimitedFiles

# custom packages
includet("packages/jlgym.jl"); using .JlGym
includet("packages/MARXAgents.jl"); using .MARXAgents

# python packages
gym = pyimport("gymnasium")
pybullet = pyimport("pybullet")
pushfirst!(pyimport("sys")."path", "") # add current path as search path for packages
pyimport("bittle_env");


@info "Loading environment"

# render & Images (only save images if render is true)
render = true
make_vid = false
save_every = 2                  
save_folder_image = "results/frames/"
save_folder_agent = "results/agents/"
save_folder_video = "results/videos/"

control_mode = "PD"
if render
    env = gym.make("BittleBulletEnv-v0", 
                    render_mode="human", 
                    control_mode=control_mode,
                    save_every=save_every,
                    #    render_option=render_option, # not possible right now (defaults to 1920x1060)
                    save_images_folder=save_folder_image);
else
    env = gym.make("BittleBulletEnv-v0", control_mode=control_mode);
end
max_torque = env.unwrapped._max_force
sys_ulims = (-max_torque, max_torque)
opts = Optim.Options(time_limit=20)
observation, info = env.reset();
legjoint_names = ["LFS_angle" "LFK_angle" "LBS_angle" "LBK_angle" "RFS_angle" "RFK_angle" "RBS_angle" "RBK_angle"]

@info "Environment loaded Successfully"
@info "Setting up variables"

# Time
Δt = 0.5
len_trial = 2000
tsteps = range(0, step=Δt, length=len_trial)
len_horizon = 3;
num_episodes = 1
keep_training = false
# contval = keep_training ? parse(Int8, split(split(parameterFile, "/")[end], "_")[end-1]) : 0 # continue value (prevents double names)

now = Dates.format(Dates.now(), "yy-mm-dd_HH_MM_SS")
fname_params = "results/params/agentparams-$now.jld"
fname_agents = "results/agents/agent-$now.jld"
fname_trials = "results/trials/agenttrial-$now.jld"

# Dimensionalities
Mu = 2 # includes current control uk
My = 2
Dy = 8 
Du = env.action_space.shape[1]
Dz = env.observation_space.shape[1]
Dx = My*Dy + Mu*Du

# Prior parameters
ν0 = 100.
Ω0 = 1e0*diagm(ones(Dy))
Λ0 = 1e-3*diagm(ones(Dx))
M0 = 1e-8*randn(Dx,Dy)
Υ  = 1e-1*diagm(ones(Du))

# Setpoint (desired observation)
m_star = [0., 0., 0., 1.0, 0.0, 0.0, 0.0, 0.0] # [roll, pitch, v_x, v_y, v_z, ω_x, ω_y, ω_z]
v_star = [1., 1., 1., 0.1, 1.0, 1.0, 1.0, 1.0]
goal = MvNormal(m_star, diagm(v_star))

# Start agent
if keep_training
    global agent = loadModel(parameterFile)
else
    global agent = MARXAgent(M0,Λ0,Ω0,ν0,Υ, goal, Dy=Dy, Du=Du, delay_inp=Mu, delay_out=My, time_horizon=len_horizon)
end

if make_vid && render
    mkpath(save_folder_video)
    pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI,0) # remove GUI layout from screen
    pybullet.startStateLogging(pybullet.STATE_LOGGING_VIDEO_MP4, save_folder_video*"test_$now.mp4")
end

@info "starting up training loop"
# for ep in 1:num_episodes

    _,_ = env.reset()
    reset_buffer!(agent)
    global lasti = len_trial

    # Preallocate
    # y_pln  = (zeros(len_trial, Dy,len_horizon), zeros(len_trial, Dy,Dy,len_horizon))
    global y_sim = zeros(Dy, len_trial)
    global z_sim = zeros(Dz, len_trial)
    global u_sim = zeros(Du, len_trial)
    global F_sim = zeros(len_trial)
    global preds_m = zeros(Dy, len_trial)
    global preds_S = repeat(diagm(ones(Dy)), outer=[1, 1, len_trial])
    global Ms = zeros(Dx,Dy,len_trial)
    global Λs = zeros(Dx,Dx,len_trial)
    global Ωs = zeros(Dy,Dy,len_trial)
    global νs = zeros(len_trial)

    @showprogress for k in 2:len_trial
        
        "Predict observation"
    
        x_k = [agent.ubuffer[:]; agent.ybuffer[:]]
        η_k,μ_k,Ψ_k = posterior_predictive(agent, x_k)
        preds_m[:,k] = μ_k
        preds_S[:,:,k] = inv(Ψ_k) * η_k/(η_k - 2)
        
        "Interact with environment"

        # Update system with selected control
        observation, performance, terminated, truncated, status = env.step(u_sim[:,k-1])

        # Reveal base orientation, velocity and angular velocity to agent
        z_sim[:,k] = observation
        y_sim[:,k] = observation[17:24]
                
        "Parameter estimation"

        # Update parameters
        MARXAgents.update!(agent, y_sim[:,k], u_sim[:,k-1])

        # Track parameter beliefs
        Ms[:,:,k] = agent.M
        Λs[:,:,k] = agent.Λ
        Ωs[:,:,k] = agent.Ω
        νs[k]     = agent.ν

        # Track free energy
        F_sim[k] = agent.free_energy
        
        # # Visualize objective
        # for ii in 1:Nu^2
        
        #     # Update control buffer
        #     ub = MARXAgents.backshift(agent.ubuffer, uu[:,ii])
        #     xx = [ub[:]; agent.ybuffer[:]]

        #     # Mutual info 
        #     MI[ii,k] = mutualinfo(agent, xx)
        #     CE[ii,k] = crossentropy(agent, xx)
        #     Ju[ii,k] = MI[ii,k] + CE[ii,k]

        # end

        "Action selection"
        
        # Call minimizer using constrained L-BFGS procedure
        G(u::AbstractVector) = EFE(agent, u)
        results = Optim.optimize(G, sys_ulims[1], sys_ulims[2], zeros(Du*len_horizon), Fminbox(LBFGS()), opts; autodiff=:forward)
        
        # Extract minimizing control
        policy = Optim.minimizer(results)
        u_sim[:,k] = policy[1:Du]

        # # # Planning under optimized policy
        # planned_obs = predictions(agent, reshape(policy, (Du,len_horizon)), time_horizon=len_horizon)
        # y_pln[1][k,:,:]   = planned_obs[1]
        # y_pln[2][k,:,:,:] = planned_obs[2]
        
        global lasti = k
    end
    # @info "Episode $ep finished: $(ep/num_episodes*100)% done"
    
    # Save experiment
    JLD.save(fname_agents, "agent", agent)
    JLD.save(fname_params, "M", Ms[:,:,end], "Λ", Λs[:,:,end], "Ω", Ωs[:,:,end], "ν", νs[end], "Υ", Υ)
    JLD.save(fname_trials, "states", z_sim, "actions", u_sim, "FE", F_sim, "predictions", (preds_m, preds_S))

    p1 = plot(transpose(z_sim[1:8, 1:lasti]), title="Joint Angles", xlabel="Time", ylabel="Angle (π radians)", size=(800, 400), labels=legjoint_names)
    p2 = plot(transpose(z_sim[9:16, 1:lasti]), title="Joint Velocities", xlabel="Time", ylabel="Velocity (m/s)", size=(800, 400), labels=legjoint_names)
    p4 = plot(transpose(z_sim[17:18, 1:lasti]), title="Base Orientation", xlabel="Time", ylabel="Orientation (π radians)", size=(800, 400), label=["roll" "pitch"])
    p5 = plot(transpose(z_sim[19:21, 1:lasti]), title="Base Velocity", xlabel="Time", ylabel="Velocity (m/s)", size=(800, 400), label=["v_x" "v_y" "v_z"])
    p6 = plot(transpose(z_sim[22:24, 1:lasti]), title="Base Angular Velocity", xlabel="Time", ylabel="Angular Velocity (rad/s)", size=(800, 400), label=["ω_x" "ω_y" "ω_z"])
    p7 = plot(heatmap(1:lasti, 1:4, z_sim[25:28, 1:lasti], xlims=(0, lasti+1), c=[:white, :black], title="Footplacement", xlabel="Time", ylabel="Foot", yflip=true, colorbar=false, size=(800, 200), xticks=0:10:50, yticks=1:4))
    p8 = plot(heatmap(1:lasti, 1:8, z_sim[29:36, 1:lasti], c=[:white, :black], title="Previous Action for every timestep", xlabel="Time", ylabel="Actor", yflip=true, size=(800, 400), xticks=0:10:lasti, yticks=1:8))
    
    ptot = plot(p1, p2, p4, p5, p6, p7, p8, layout=(7,1), size=(800, 2400))
    savefig(ptot, "results/experiment-$now.png")
# end

@info "Training loop finished -> closing environment"
env.close()
@info "Done!"