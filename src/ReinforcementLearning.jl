#ReinforcementLearning.jl
#desc: The author's attempt at a semi-consistent reinforcement learning package that implements a few different algorithms
#auth: Christopher Ho
#affil: Stanford University
#date: 1/11/2016

#TODO: future file demarcations
"""
ReinforcementLearning.jl
BlackBoxModel.jl
simulator.jl
solver.jl
annealer.jl
policy.jl
minibatch.jl
experiencereplay.jl
solvers/
  ForgetfulLSTD.jl
  SARSA.jl #also q-learning
  DeterministicPolicyGradient.jl
  LSPolicyIteration.jl
"""
###########################
#Dicking around for RL gm type
type BlackBoxModel
  state
  isterminal::Function
  next_state::Function
  observe::Function
  reward::Function
  init::Function
  rng::AbstractRNG
  actions::ActionSpace
  #TODO: action domain?
end

function init(bbm::BlackBoxModel,rng::AbstractRNG=MersenneTwister(4398))
  bbm.state = bbm.init(rng)
  #emit an initial observation? or is it required that you take an action first
end

function next(bbm::BlackBoxModel, action)
  bbm.state = bbm.next_state(bbm.rng,bbm.state,action)
  o = bbm.observe(bbm.rng,bbm.state,action)
  r = bbm.reward(bbm.rng,bbm.state,bbm.action)
  return r,o
end

function isterminal(bbm::BlackBoxModel,action)
  return bbm.isterminal(bbm.state,action)
end
###########################
abstract ActionSpace

#TODO: parameterize?
type DiscreteActionSpace{T} <: ActionSpace
  A::Array{T,1}
end
domain(a::DiscreteActionSpace) = a.A
length(a::DiscreteActionSpace) = length(a.A)

type ContinuousActionSpace <: ActionSpace
  #TODO
end

abstract Policy
action(::Policy,s) = error("Policy type uninstantiated")

type DiscretePolicy <: Policy
  A::DiscreteActionSpace
  feature_function::Function
  weights::Array{Float64,1} #NOTE: alternatively cold have entire updater struct
end

function action(p::DiscretePolicy,s)
  Qs = zeros(length(p.actions))
  for (i,a) in enumerate(domain(p.actions))
    Qs[i] = dot(p.weights,p.feature_function(s,a)) #where is a sensible place to put w?
  end
  return domain(p.actions)[indmax(Qs)]
end


type Simulator
  discount::Float64
  simRNG::AbstractRNG
  actRNG::AbstractRNG
  nb_sim::Int
  nb_timesteps::Int
end

#TODO: policy types should probably have the feature function built in as a matter of fact
#TODO: parallelize
#TODO: handle saving histories for visualization
function simulate(sim::Simulator,bbm::BlackBoxModel,policy::Policy)
  R_net = zeros(sim.nb_sim)
  for ep = 1:sim.nb_sim
    R_net[ep] = __simulate(sim,bbm,policy)
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
    r, s_ = next(bbm,a)
    a_ = action(policy,s_)
    gamma = isterminal(bbm,a_) ? 0. : sim.discount^t
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

###########################
abstract AnnealerParam
"""
type Annealer
  update_rate::Function
  param::AnnealerParam
end
"""
#Vanilla annealer
type NullAnnealer <: AnnealerParam end
anneal!(::AnnealerParam,dw::Array{Float64,1}) = dw


#TODO: make minibatch and experience replay modules that you can add or remove from
#     annealers as needed?
abstract ExperienceReplayer
type NullExperienceReplayer <:ExperienceReplayer end
replay!(er::NullExperienceReplayer,phi,r,phi_) = phi,r,phi_#placeholder

#TODO: parameterize this?
#TODO: or make it just a pair of feature vectors + reward
type Experience
  phi::Array{Union{Float64,Int},1}
  r::Float64
  phi_::Array{Union{Float64,Int},1}
end
remember(e::Experience) = e.phi,e.r,e.phi_
type UniformExperienceReplayer <: ExperienceReplayer
  memory::Array{Experience,1}
  nb_mem::Int
  rng::AbstractRNG
end
function replay!(er::UniformExperienceReplayer,
                  phi::::Array{Union{Float64,Int},1},
                  r::Float64,
                  phi_::Array{Union{Float64,Int},1})
  e = Experience(phi,r,phi_)
  if length(er.memory) < er.nb_mem
    push!(er.memory,e)
  else
    ind = rand(er.rng,1:er.nb_mem)
    #is it more memory efficient to delete somehow first?
    er.memory[ind] = e
  end
  ind = rand(er.rng,1:er.nb_mem)
  return remember(er.mem[ind])
end


##################################
abstract Minibatcher
type NullMinibatcher <: Minibatcher end
minibatch!(mb::NullMinibatcher,dw::Array{Float64,1}) = dw

type UniformMinibatcher <: Minibatcher
  minibatch_size::Int
  dw::Array{Float64,1}
  current_minibatch_size::Int
end
function minibatch!(mb::UniformMinibatcher,dw::Array{Float64,1})
  if mb.current_minibatch_size < mb.minibatch_size
    mb.dw += dw
    mb.current_minibatch_size += 1
    return zeros(size(dw))
  else
    dw_ = (mb.dw + dw)./mb.minibatch_size
    mb.current_minibatch_size = 0
    mb.dw = zeros(size(dw))
    return dw_
  end
end

#TODO: something that halves the learning rate every t0 time steps -- probably not that useful, but whatever
#TODO: RMSProp
#TODO: momentum
#TODO: nesterov
#TODO: minibatching/experience replay
#TODO: adagrad/adadelta

abstract UpdaterParam
#TODO: is this data structure even necessary because of multiple dispatch? ex:
"""
type Updater
  update_weights::Function
  param::UpdaterParam
end
"""
abstract ExplorationPolicy <: Policy
#NOTE: exploration policies are only called while solving, so its ok for them to have
#       a reference to the updater
#NOTE: that I know of, most function approximation reinforcement learning techniques
#       use linear stuff to get the value of something.
#       more generally, we could probably define like value(u,s,a)
type EpsilonGreedyPolicy <: ExplorationPolicy
  rng::AbstractRNG
  actions::DiscreteActionSpace #TODO
  feature_function::Function
  eps::Float64
  function EpsilonGreedyPolicy(feature_function::Function,actions::DiscreteActionSpace;
                                rng::AbstractRNG=MersenneTwister(2983461),
                                eps::Float64=0.15)
    self = new()
    self.rng = rng
    self.actions = actions
    self.feature_function = feature_function
    self.eps = eps
  end
end
function action(p::EpsilonGreedyPolicy,u::UpdaterParam,s)
  """
  Sketching things out right now, nothing here is final or working
  """
  r = rand(p.rng)
  if r < p.eps
    return rand(p.rng,1:length(p.actions))
  end
  Qs = zeros(length(p.actions))
  for (i,a) in enumerate(domain(p.actions))
    Qs[i] = dot(weights(u),p.feature_function(s,a)) #where is a sensible place to put w?
  end
  return p.actions[indmax(Qs)]
end

type SoftmaxPolicy <: ExplorationPolicy
  rng::AbstractRNG
  actions::DiscreteActionSpace #TODO
  feature_function::Function
  tau::Float64
end

import StatsBase: sample, WeightVec
#using StatsBase.sample means RNG is useless, but I'm lazy
function action(p::SoftmaxPolicy,u::UpdaterParam,s)
  """
  Sketching things out right now, nothing here is final or working
  """
  Qs = zeros(length(p.actions))
  for (i,a) in enumerate(domain(p.actions))
    Qs[i] = exp(p.tau*dot(weights(u),p.feature_function(s,a))) #where is a sensible place to put w?
  end
  return sample(p.actions,WeightVec(Qs))
end

#For continuous action spaces
#can potentially make this more generic with difference noise models
type GaussianPolicy <: ExplorationPolicy
  actions::ContinuousActionSpace

end
function action(p::GaussianPolicy,s)
  #TODO
end
##TODO: maybe have a generic solver class that has a lot of these things
##      and it has a place to hold a like "algorithm" type that is made up of an
##      updateweights function, and whatever things it needs to hold onto?
type Solver
  lr::Float64 #initial learning relate
  nb_episodes::Int
  nb_timesteps::Int
  discount::Float64
  simRNG::AbstractRNG
  annealer::AnnealerParam #how the learning rate adapts
  updater::UpdaterParam
  mb::Minibatcher
  er::ExperienceReplayer
  function Solver(updater::UpdaterParam;
                    lr::Float64=0.01,
                    nb_episodes::Int=100,
                    nb_timesteps::Int=100,
                    discount::Float64=0.99,
                    simRNG::AbstractRNG=MersenneTwister(23894),
                    annealer::AnnealerParam=NullAnnealer(),
                    mb::Minibatcher=NullMinibatcher(),
                    er::ExperienceReplayer=NullExperienceReplayer())
    self = new()
    self.lr = lr
    self.nb_episodes = nb_episodes
    self.nb_timesteps = nb_timesteps
    self.discont = discount
    self.simRNG = simRNG
    self.annealer = annealer
    self.mb = mb
    self.er = er
    self.updater = updater
  end
end


function solve(solver::Solver,bbm::BlackBoxModel,policy::Policy)

  #maintain statistics?
  for ep = 1:solver.nb_episodes
    #episode setup stuff
    s = feature_function(init(bbm,solver.simRNG))
    a = action(policy,updater,s)
    phi = policy.feature_function(s,a)
    for t = 1:solver.nb_timesteps
      r, s_ = next(bbm,a)
      a_ = action(policy,updater,s_)
      phi_ = policy.feature_function(s_,a_)
      gamma = isterminal(bbm,a_) ? 0. : solver.discount
      update!(solver.updater,solver.annealer,solver.mb,solver.er,phi,r,phi_,gamma,lr)
      if gamma == 0.
        break
      end
      #push the update frame up one time step as it were
      s = s_ #TODO: not needed?
      a = a_
      phi = phi_
    end #t

  end #ep

  #return something--policy? stats?

end

##TODO: policy representation
##TODO: basic policies
##TODO: reference NNDPG--that structure wasn't bad

#################################
type ForgetfulLSTDParam <: UpdaterParam

end


#####################################

type DeterministicPolicyGradientParam <: UpdaterParam

end
#######################################
#TODO: use enum
type SARSAParam <: UpdaterParam
  lambda::Float64 #the eligility trace parameters
  w::Array{Float64,1} #weight vector
  e::Array{Float64,1} #eligibility trace
  is_replacing_trace::Bool #i only know of two trace updates
  function SARSAParam(n::Int;
                      lambda::Float64=0.5,
                      init_method::AbstractString="unif_rand",
                      trace_type::AbstractString="replacing")
    self = new()
    if lowercase(init_method) == "unif_rand"
      self.w = rand(n)-0.5 #or something
    else
      error("No such weight initialization method: $init_method")
    end
    self.e = zeros(n)
    self.is_replacing_trace = lowercase(trace_type) == "replacing"
    return self
  end
end
weights(u::SARSAParam) = u.w

function update!(param::SARSAParam,
                  annealer::AnnealerParam,
                  mb::Minibatcher,
                  er::ExperienceReplayer,
                  phi::Array{Union{Float64,Int},1},
                  r::Union{Float64,Int},
                  phi_::Array{Union{Float64,Int},1},
                  gamma::Float64,
                  lr::Float64)
  phi,r,phi_ = replay!(er,phi,r,phi_)
  q = dot(param.w,phi) #TODO: dealing with feature functions that involve state and action?
  q_ = dot(param.w,phi_)
  del = r + discount*q_ - q #td error
  if param.is_replacing_trace
    param.e = max(phi,param.lambda.*param.e) #NOTE: assumes binary features
  else
    param.e  = phi + param.lambda.*param.e
  end
  dw = del*param.e
  param.w += lr.*anneal!(annealer,minibatch!(mb,dw))
end

#######################################
type QLearningSolver

end
