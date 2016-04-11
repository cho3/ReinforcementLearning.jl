#solve.jl
#a place to hold the base solver stuff

type SolverHistory
  R_tot::Array{Float64,1}
  td_err::Array{Float64,1}
  q_est::Array{Float64,1}
  w_norm::Array{Float64,1}
  sim_u::Array{Float64,1} #avg reward
  sim_v::Array{Float64,1} #var
  sim_interval::Int
end

function truncate!(stats::SolverHistory,ind::Int)
  stats.td_err = stats.td_err[1:ind]
  stats.q_est = stats.q_est[1:ind]
  stats.w_norm = stats.w_norm[1:ind]
end

function display_stats(stats::SolverHistory)
  #subplot
  subplot(511)
  plot(stats.R_tot)
  xlabel("Episode")
  ylabel("Total Reward")

  subplot(512)
  plot(stats.td_err)
  xlabel("Time Step")
  ylabel("TD Error")

  subplot(513)
  plot(stats.q_est)
  xlabel("Time Step")
  ylabel("Q Value Estimate")

  subplot(514)
  plot(stats.w_norm)
  xlabel("Time Step")
  ylabel("||w||_2")

  subplot(515)
  errorbar(stats.sim_interval*collect(1:length(stats.sim_u)),stats.sim_u,yerr=stats.sim_v)
  suptitle("Reinforcement Learning Statistics and Metrics")

end

#TODO: verbose
type Solver
  #lr::Float64 #initial learning relate
  ra::RateAdapter
  nb_episodes::Int
  nb_timesteps::Int
  discount::Float64
  simRNG::AbstractRNG
  annealer::AnnealerParam #how the learning rate adapts
  updater::UpdaterParam
  mb::Minibatcher
  er::ExperienceReplayer
  gc::GradientClipper
  verbose::Bool
  display_interval::Int
  grandiloquent::Bool #make plots?
  stats::SolverHistory
  expma_param::Float64
  sim_interval::Int
  simulator::Simulator
  best_policy::Policy
  function Solver(updater::UpdaterParam;
                    lr::Float64=0.1,
                    nb_episodes::Int=100,
                    nb_timesteps::Int=100,
                    discount::Float64=0.99,
                    simRNG::AbstractRNG=MersenneTwister(23894),
                    annealer::AnnealerParam=NullAnnealer(),
                    mb::Minibatcher=NullMinibatcher(),
                    er::ExperienceReplayer=NullExperienceReplayer(),
                    verbose::Bool=true,
                    display_interval::Int=10,
                    grandiloquent::Bool=true,
                    expma_param::Float64=0.9,
                    sim_interval::Int=100,
                    simulator::Simulator=Simulator(),
                    ra::RateAdapter=NullRateAdapter(lr),
                    gc::GradientClipper=GradientClipper())
    self = new()
    #self.lr = lr
    self.ra = ra
    self.nb_episodes = nb_episodes
    self.nb_timesteps = nb_timesteps
    self.discount = discount
    self.simRNG = simRNG
    self.annealer = annealer
    self.mb = mb
    self.er = er
    self.updater = updater
    self.verbose = verbose
    self.display_interval = display_interval
    self.grandiloquent = grandiloquent
    self.expma_param = expma_param
    self.stats = SolverHistory(zeros(nb_episodes),zeros(nb_episodes*nb_timesteps),
                  zeros(nb_episodes*nb_timesteps),zeros(nb_episodes*nb_timesteps),
                  zeros(floor(Int,nb_episodes/sim_interval)),
                  zeros(floor(Int,nb_episodes/sim_interval)),sim_interval)
    self.sim_interval = sim_interval
    self.simulator = simulator
    self.gc = gc

    return self
  end
end


#TODO: phi, a = action(policy,updater,s)?
#NOTE: there are 3 statistics that are worthwhile to keep track of:
#       total reward (per episode), td error (per time step), est. q-value (per time step)
#TODO: early stopping?
function solve(solver::Solver,bbm::BlackBoxModel,policy::Policy)

  #TODO: maintain Q-statistics from update in order to plot things
  #maintain statistics?
  solver.best_policy = deepcopy(policy)
  best_score = -Inf
  if solver.verbose
    println("Solving problem...")
    R_avg = 0.
    td_avg = 0.
    q_avg = 0.
    last_sim = 0.
  end
  ind = 1
  for ep = 1:solver.nb_episodes
    #episode setup stuff
    break_flag = false
    R_ep = 0.
    init!(policy)
    s = init(bbm)
    a = action(policy,solver.updater,s)
    phi = policy.feature_function(s) #might not be appropriate
    for t = 1:solver.nb_timesteps
      r, s_ = next(bbm,a)
      a_ = action(policy,solver.updater,s_)
      phi_ = policy.feature_function(s_) #might not be apprpopriate
      #second expression does nothing
      gamma =  ((break_flag) || (t > solver.nb_timesteps)) ? 0. : solver.discount
      #NOTE: using s,a,r,s_,a_ for maximum generality
      lr = learning_rate!(solver.ra,t,ep,phi,solver.updater)
      td, q = update!(solver.updater,solver.annealer,solver.mb,solver.er,policy.exp,solver.gc,
                      phi,a,r,phi_,a,gamma,lr)

      R_ep += r
      #update td, q
      #ind = t + (ep-1)*solver.nb_timesteps
      solver.stats.td_err[ind] = td
      solver.stats.q_est[ind] = q
      solver.stats.w_norm[ind] = norm(weights(solver.updater))
      ind += 1
      if solver.verbose
        #update moving averages
        td_avg = solver.expma_param*td_avg + (1.-solver.expma_param)*abs(td)
        q_avg = solver.expma_param*q_avg + (1.-solver.expma_param)*q
      end
      if break_flag
        break
      end

      if isterminal(bbm,a_)
        break_flag = true
        #break
      end
      #push the update frame up one time step as it were
      s = s_ #TODO: not needed?
      a = a_
      phi = phi_
    end #t

    #update R_tot
    solver.stats.R_tot[ep] = R_ep
    if solver.verbose && (mod(ep,solver.display_interval) == 0)
      #update R_avg
      R_avg = solver.expma_param*R_avg + (1.-solver.expma_param)*R_ep
      #display moving averages
      print("\r")
      print("Episode $ep, \tAvg Reward: $(round(R_avg,3)), \tAvg Abs. TD Error: $(round(td_avg,3)), \tAvg Q-value: $(round(q_avg,3)), \tLast Avg Reward: $(round(last_sim,3))")
    end

  if solver.sim_interval > 0
    if mod(ep,solver.sim_interval) == 0
      last_sim,v, et = simulate(solver.simulator,bbm,Policy(policy,solver.updater,policy.exp),"Episode $ep; ")
      sim_idx = floor(Int,ep/solver.sim_interval)
      solver.stats.sim_u[sim_idx] = last_sim
      solver.stats.sim_v[sim_idx] = v
      if last_sim > best_score
        solver.best_policy = Policy(policy,solver.updater,policy.exp)
        best_score = last_sim
      end
    end
  end

  end #ep
  truncate!(solver.stats,ind-1)
  if solver.grandiloquent
    #plot each of the major statistics
    display_stats(solver.stats)
  end

  if best_score == -Inf
    solver.best_policy = Policy(policy,solver.updater,policy.exp)
  end

  #return something--policy? stats?
  return Policy(policy,solver.updater,policy.exp)
end

##########################################################

#solver for algorithms that require multiple simulations at each time Step
#noise model stored in MCUpdater <: UpdaterParam
#todo history type for stuff

function solve(solver::Solver,bbm::BlackBoxModel)


  #init stuff
  for iter = 1:solver.nb_episodes #max_iter

      for sim_idx = 1:solver.sim.nb_sim
        #permute policy
        #run simulation
        #save statistics based on updater
      end
      #aggregate statistics
      #update updater
      if stopping_criterion()
        #store statistics
        break
      end
  end

end
