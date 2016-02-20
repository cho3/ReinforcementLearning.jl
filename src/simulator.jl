#simulator.jl
#just run a simulation of the black box model using a learned policy and collect statistics
type History
  S
  A
end
History() = History(0,0)

type R_Hist
  Rs::Array{Array{Real,1},1}
end
function sim_stats(hist::R_Hist,discount::Float64=0.99)
  Rs = hist.Rs
  Rs_ = zeros(length(Rs))
  for R_ep in Rs
    gamma = 1.
    R = 0.
    for r in R_ep
      R += gamma*r
      gamma *= discount
    end
    Rs[ep] = R
  end
  return mean(Rs), std(Rs)
end

type Simulator
  discount::Float64
  simRNG::AbstractRNG
  actRNG::AbstractRNG
  nb_sim::Int
  nb_timesteps::Int
  verbose::Bool
  display_interval::Int
  visualizer::Function
  hist::History
  r_hist::R_Hist
  save_rewards::Bool
  function Simulator(;discount::Float64=0.99,
                      simRNG::AbstractRNG=MersenneTwister(234234),
                      actRNG::AbstractRNG=MersenneTwister(98765432436),
                      nb_sim::Int=100,
                      nb_timesteps::Int=100,
                      verbose::Bool=true,
                      display_interval::Int=10,
                      visualizer::Function=__visualizer,
                      save_rewards::Bool=false)
    self = new()
    self.discount = discount
    self.simRNG = simRNG
    self.actRNG = actRNG
    self.nb_sim = nb_sim
    self.nb_timesteps = nb_timesteps
    self.verbose = verbose
    self.display_interval = display_interval
    self.visualizer = visualizer
    self.hist = History()
    self.save_rewards = save_rewards
    if save_rewards
      self.r_hist = R_Hist([zeros(nb_timesteps) for _ = 1:nb_sim])
    else
      self.r_hist = R_Hist(Array{Real,1}[Real[]])
    end

    return self
  end
end

#TODO: policy types should probably have the feature function built in as a matter of fact
#TODO: parallelize
#TODO: handle saving histories for visualization
function simulate(sim::Simulator,bbm::BlackBoxModel,policy::Policy,msg::AbstractString="")
  R_net = zeros(sim.nb_sim)
  if sim.verbose
    print("\rSimulating!")
  end
  for ep = 1:sim.nb_sim
    R_net[ep] = __simulate(sim,bbm,policy,ep)
    if sim.verbose && (mod(ep,sim.display_interval) == 0)
      print("\r")
      u = mean(R_net[1:ep])
      v = std(R_net[1:ep])
      print("$(msg)Simulation $(ep), Average Total Reward: $(round(u,3))+/-$(round(v,3)), 95% Confidence Interval: ($(round(u-1.94*v,3)),$(round((u+1.94*v),3))\t\t\t\t\t\t ")
    end
  end
  if sim.visualizer != __visualizer
    __viz_sim(sim,bbm,policy)
  end
  #compute relevant statistic, e.g.
  return mean(R_net), std(R_net)
end

function parallel_simulate(sim::Simulator,bbm::BlackBoxModel,policy::Policy,msg::AbstractString="")
  #each simulation instance needs its own blackboxmodel, Policy
  R_net = SharedArray(Float64,sim.nb_sim)
  #=
  R_net = collect(1:sim.nb_sim)
  if sim.verbose; print("\rSimulating Parallelly!"); end
  function p_sim(ep::Int)
    _bbm = deepcopy(bbm)
    _policy = deepcopy(policy)
    return __simulate(sim,_bbm,_policy,ep)
  end
  =#
  ##=
  @sync @parallel for ep = 1:sim.nb_sim
    #super lazy
    #println("Hello from process!")
    _bbm = deepcopy(bbm)
    _policy = deepcopy(policy)
    R_net[ep] = __simulate(sim,_bbm,_policy,ep)
  end
  # =#
  #println("Before PMAP")
  #@async R_net = pmap(p_sim,R_net)
  #println("PMAP: $R_net")
  return mean(R_net), std(R_net)
end


#run a single simulation
function __simulate(sim::Simulator,bbm::BlackBoxModel,policy::Policy,ep::Int=1)
  #println("Episode $ep")
  R_tot = 0.
  s = init(bbm,sim.simRNG)
  a = action(policy,s) #TODO: stuff
  for t = 0:(sim.nb_timesteps-1)
    r, s_ = next(bbm,a,sim.simRNG)
    #println("T=$t")
    a_ = action(policy,s_)
    gamma = sim.discount^t
    R_tot += gamma*r
    if sim.save_rewards
      sim.R_hist.Rs[ep][t] = r
    end
    if isterminal(bbm,a_,sim.simRNG)
      break
    end
    #push the update frame up one time step as it were
    s = s_
    a = a_
  end #t
  #println("Got reward $R_tot")
  return R_tot
end

function __viz_sim(sim::Simulator,bbm::BlackBoxModel,policy::Policy)

  s = init(bbm,sim.simRNG)
  a = action(policy,s) #TODO: stuff
  S = typeof(s)[s]
  A = typeof(a)[a]
  for t = 0:(sim.nb_timesteps-1)
    r, s_ = next(bbm,a,sim.simRNG)
    a_ = action(policy,s_)
    push!(S,s_)
    push!(A,a_)
    if isterminal(bbm,a_,sim.simRNG)
      r, s_ = next(bbm,a,sim.simRNG)
      a_ = action(policy,s_)
      push!(S,s_)
      push!(A,a_)
      break
    end
    #push the update frame up one time step as it were
    s = s_
    a = a_
  end #t
  sim.hist = History(S,A)
  f = figure()
  @manipulate for i = 1:length(S); withfig(f) do
    sim.visualizer(bbm.model,S[i],A[i]) end
  end
end
__visualizer{S,T}(s::Array{S,1},a::Array{T,1}) = print("Empty Visualizer")
