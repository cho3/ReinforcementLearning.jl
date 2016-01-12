#ReinforcementLearning.jl
#desc: The author's attempt at a semi-consistent reinforcement learning package that implements a few different algorithms
#auth: Christopher Ho
#affil: Stanford University
#date: 1/11/2016

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
  #TODO: action domain?
end

init(bbm::BlackBoxModel)
r,o = next(bbm::BlackBoxModel,action)
isterminal(bbm::BlackBoxModel,action)


function init(bbm::BlackBoxModel,rng="")
  bbm.state = bbm.init(rng)
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

abstract Solver


##TODO: maybe have a generic solver class that has a lot of these things
##      and it has a place to hold a like "algorithm" type that is made up of an
##      updateweights function, and whatever things it needs to hold onto?
type SARSASolver <: Solver
  lr::Float64 #initial learning relate
  lambda::Float64 #eligibility trace parameter
  annealer::Function #how the learning rate adapts
  nb_episodes::Int
  nb_timesteps::Int
  discount::Float64
  simRNG::AbstractRNG
  actRNG::AbstractRNG
end


function solve(solver::SARSASolver,bbm::BlackBoxModel)

end

##TODO: policy representation
##TODO: basic policies
##TODO: reference NNDPG--that structure wasn't bad

#################################
type ForgetfulLSTDSolver <: Solver

end


#####################################

type DeterministicPolicyGradientSolver <: Solver
