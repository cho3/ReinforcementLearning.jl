#policy.jl
#holds the exploration and frozen policies and action space stuff and whatever

abstract ActionSpace

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

abstract ExplorationPolicy <: Policy
#NOTE: exploration policies are only called while solving, so its ok for them to have
#       a reference to the updater
#NOTE: that I know of, most function approximation reinforcement learning techniques
#       use linear stuff to get the value of something.
#       more generally, we could probably define like value(u,s,a)
type EpsilonGreedyPolicy <: ExplorationPolicy
  rng::AbstractRNG
  A::DiscreteActionSpace #TODO
  feature_function::Function
  eps::Float64
  function EpsilonGreedyPolicy(feature_function::Function,actions::DiscreteActionSpace;
                                rng::AbstractRNG=MersenneTwister(2983461),
                                eps::Float64=0.15)
    self = new()
    self.rng = rng
    self.A = actions
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
    return rand(p.rng,1:length(p.A))
  end
  Qs = zeros(length(p.A))
  for (i,a) in enumerate(domain(p.A))
    Qs[i] = dot(weights(u),p.feature_function(s,a)) #where is a sensible place to put w?
  end
  return domain(p.A)[indmax(Qs)]
end

type SoftmaxPolicy <: ExplorationPolicy
  rng::AbstractRNG
  A::DiscreteActionSpace #TODO
  feature_function::Function
  tau::Float64
end

import StatsBase: sample, WeightVec
#using StatsBase.sample means RNG is useless, but I'm lazy
function action{T}(p::SoftmaxPolicy,u::UpdaterParam,s::T)
  """
  Sketching things out right now, nothing here is final or working
  """
  Qs = zeros(length(p.A))
  for (i,a) in enumerate(domain(p.A))
    Qs[i] = exp(p.tau*dot(weights(u),p.feature_function(s,a))) #where is a sensible place to put w?
  end
  return sample(domain(p.A),WeightVec(Qs))
end

#For continuous action spaces
#can potentially make this more generic with difference noise models
type GaussianPolicy <: ExplorationPolicy
  A::ContinuousActionSpace

end
function action(p::GaussianPolicy,s)
  #TODO
end

#NOTE: this miiiight not work and just pollute the namespace instead
Policy(p::Union{EpsilonGreedyPolicy,SoftmaxPolicy},u::UpdaterParam) =
                        DiscretePolicy(p.A,p.feature_function,weights(u))
