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

type Simulator
  discount::Float64
  simRNG::AbstractRNG
  actRNG::AbstractRNG
  nb_sim::Int
  nb_timesteps::Int
end

function simulate(sim::Simulator,bbm::BlackBoxModel)

end

###########################
abstract AnnealerParam
type Annealer
  update_rate::Function
  param::AnnealerParam
end
VanillaAnnealer(::AnnealerParam,dw::Array{Float64,1}) = dw

#TODO: something that halves the learning rate every t0 time steps -- probably not that useful, but whatever
#TODO: RMSProp
#TODO: momentum
#TODO: nesterov
#TODO: minibatching/experience replay
#TODO: adagrad/adadelta

abstract UpdaterParam
type Updater
  update_weights::Function
  param::UpdaterParam
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
  actRNG::AbstractRNG
  annealer::Annealer #how the learning rate adapts
  updater::Updater
end


function solve(solver::Solver,bbm::BlackBoxModel,feature_function::Function)

  #maintain statistics?
  for ep = 1:solver.nb_episodes
    #episode setup stuff
    s = feature_function(init(bbm,solver.simRNG))
    a = policy(s) #TODO: stuff
    for t = 1:solver.nb_timesteps
      r, s_ = next(bbm,a)
      s_ = feature_function(s_)
      a_ = policy(s_)
      gamma = isterminal(bbm,a_) ? 0. : solver.discount
      solver.updater.update_weights(solver.updater.param,solver.annealer,s,a,r,s_,a_,gamma,lr)
      if gamma == 0.
        break
      end
      #push the update frame up one time step as it were
      s = s_
      a = a_
    end #t

  end #ep

  #return something--policy? stats?

end

##TODO: policy representation
##TODO: basic policies
##TODO: reference NNDPG--that structure wasn't bad

#################################
type ForgetfulLSTDSolver <: Solver

end


#####################################

type DeterministicPolicyGradientSolver <: Solver

end
#######################################
type SARSAParam
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

function SARSAUpdater(param::SARSAParam,annealer::Annealer,
                      s::Array{Union{Float64,Int},1},
                      a,
                      r::Union{Float64,Int},
                      s_::Array{Union{Float64,Int},1},
                      a_,
                      gamma::Float64,
                      lr::Float64)

  q = dot(param.w,s) #TODO: dealing with feature functions that involve state and action?
  q_ = dot(param.w,s_)
  del = r + discount*q_ - q #td error
  if param.is_replacing_trace
    param.e = max(s,param.lambda.*param.e) #NOTE: assumes binary features
  else
    param.e  = s + param.lambda.*param.e
  end
  dw = del*param.e
  param.w += lr.*annealer.update_rate(annealer.param,param.e)
end

#######################################
type QLearningSolver

end
