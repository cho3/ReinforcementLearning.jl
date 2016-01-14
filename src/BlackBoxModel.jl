#BlackBoxModel.jl
#An implementation of a black box environmental model

type BlackBoxModel{T}
  state::T
  isterminal::Function
  next_state::Function
  observe::Function
  reward::Function
  init::Function
  rng::AbstractRNG
end

function init(bbm::BlackBoxModel,rng::AbstractRNG=MersenneTwister(4398))
  bbm.state = bbm.init(rng)
  #emit an initial observation? or is it required that you take an action first
end

function next{T}(bbm::BlackBoxModel, action::T)
  bbm.state = bbm.next_state(bbm.rng,bbm.state,action)
  o = bbm.observe(bbm.rng,bbm.state,action)
  r = bbm.reward(bbm.rng,bbm.state,bbm.action)
  return r,o
end

function isterminal{T}(bbm::BlackBoxModel,action::T)
  return bbm.isterminal(bbm.state,action)
end

"Base implementation of observe for fully observed models"
__observe{S,T}(rng::AbstractRNG,state::S,action::T) = state
