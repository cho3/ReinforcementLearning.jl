#simulator.jl
#just run a simulation of the black box model using a learned policy and collect statistics
type Simulator
  discount::Float64
  simRNG::AbstractRNG
  actRNG::AbstractRNG
  nb_sim::Int
  nb_timesteps::Int
  verbose::Bool
  display_interval::Int
  function Simulator(;discount::Float64=0.99,
                      simRNG::AbstractRNG=MersenneTwister(234234),
                      actRNG::AbstractRNG=MersenneTwister(98765432436),
                      nb_sim::Int=100,
                      nb_timesteps::Int=100,
                      verbose::Bool=true,
                      display_interval::Int=10)
    self = new()
    self.discount = discount
    self.simRNG = simRNG
    self.actRNG = actRNG
    self.nb_sim = nb_sim
    self.nb_timesteps = nb_timesteps
    self.verbose = verbose
    self.display_interval = display_interval

    return self
  end
end

#TODO: policy types should probably have the feature function built in as a matter of fact
#TODO: parallelize
#TODO: handle saving histories for visualization
function simulate(sim::Simulator,bbm::BlackBoxModel,policy::Policy)
  R_net = zeros(sim.nb_sim)
  if sim.verbose
    println("Simulating!")
  end
  for ep = 1:sim.nb_sim
    R_net[ep] = __simulate(sim,bbm,policy)
    if sim.verbose && (mod(ep,sim.display_interval) == 0)
      print("\r")
      u = mean(R_net[1:ep])
      v = std(R_net[1:ep])
      print("Simulation $(ep), Average Total Reward: $(u), 95% Confidence Interval: ($(u-1.94*v),$(u+1.94*v))")
    end
  end
  #compute relevant statistic, e.g.
  return mean(R_net)
end

#run a single simulation
function __simulate(sim::Simulator,bbm::BlackBoxModel,policy::Policy)
  R_tot = 0.
  s = init(bbm,sim.simRNG)
  a = action(policy,s) #TODO: stuff
  for t = 0:(sim.nb_timesteps-1)
    r, s_ = next(bbm,a,sim.simRNG)
    a_ = action(policy,s_)
    gamma = isterminal(bbm,a_,sim.simRNG) ? 0. : sim.discount^t
    R_tot += gamma*r
    if gamma == 0.
      break
    end
    #push the update frame up one time step as it were
    s = s_
    a = a_
  end #t
  return R_tot
end
