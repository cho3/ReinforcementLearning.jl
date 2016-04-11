#BlackBoxModel.jl
#An implementation of a black box environmental model

type BlackBoxModel{T}
  state::T
  isterminal::Function
  next_state::Function
  observe::Function
  reward::Function
  init::Function
  model::Model
  rng::AbstractRNG
  sasp_reward::Bool
end
function BlackBoxModel(m::Model,
                          init::Function,
                          next::Function,
                          reward::Function,
                          isterminal::Function;
                          observe::Function=__observe,
                          rng::AbstractRNG=MersenneTwister(123213),
                          sasp_reward::Bool=false)
  s = init(m,rng)
  return BlackBoxModel(s,isterminal,next,observe,reward,init,m,rng,sasp_reward)
end

function init(bbm::BlackBoxModel,rng::AbstractRNG=bbm.rng)
  bbm.state = bbm.init(bbm.model,rng)
  o = bbm.observe(rng,bbm.model,bbm.state)
  return o
  #emit an initial observation? or is it required that you take an action first
end

function next{T}(bbm::BlackBoxModel, action::T,rng::AbstractRNG=bbm.rng)
  sp = bbm.next_state(rng,bbm.model,bbm.state,action)
  if bbm.sasp_reward
    r = bbm.reward(rng,bbm.model,bbm.state,action,sp)
  else
    r = bbm.reward(rng,bbm.model,bbm.state,action)
  end
  bbm.state = sp
  o = bbm.observe(rng,bbm.model,bbm.state,action)
  return r,o
end

function isterminal{T}(bbm::BlackBoxModel,action::T,rng::AbstractRNG=bbm.rng)
  return bbm.isterminal(rng,bbm.model,bbm.state,action)
end

"Base implementation of observe for fully observed models"
__observe{S,T}(rng::AbstractRNG,m::Model,state::S,action::T) = deepcopy(state)
__observe{S}(rng::AbstractRNG,m::Model,state::S) = deepcopy(state)
