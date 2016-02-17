#policy.jl
#holds the exploration and frozen policies and action space stuff and whatever
import Base.length

type DiscreteActionSpace{T} <: ActionSpace
  A::Array{T,1}
end
domain(a::DiscreteActionSpace) = a.A
length(a::DiscreteActionSpace) = length(a.A)

type ContinuousActionSpace <: ActionSpace
  #TODO
end


action(::Policy,s) = error("Policy type uninstantiated")
range(::Policy) = error("Policy type uninstantiated")

function load_policy(fname::AbstractString,feature_function::Function)
  #TODO: enum
  file = jldopen(fname,"r")
  policy_type = read(file,"policy")
  if policy_type == "SoftmaxPolicy"
    policy = load_softmaxpolicy(file,feature_function)
  elseif policy_type == "DiscretePolicy"
    policy = load_discretepolicy(file,feature_function)
  elseif policy_type == "EpsilonGreedyPolicy"
    policy = load_epsilongreedypolicy(file,feature_function)
  else
    error("Loading unsupported for type: `$policy_type`")
  end
  close(file)
  return policy
end

###################################################################
#DiscretePolicy
type DiscretePolicy <: Policy
  A::DiscreteActionSpace
  feature_function::Function
  weights::RealVector #NOTE: alternatively cold have entire updater struct
  exp::ActionFeatureExpander
end
range(p::DiscretePolicy) = p.A
length(p::Policy,bbm::BlackBoxModel) =
  length(expand(p.exp,p.feature_function(bbm.state),domain(range(p))[1]))

function action{T}(p::DiscretePolicy,s::T)
  Qs = zeros(length(p.A))
  for (i,a) in enumerate(domain(p.A))
    Qs[i] = dot(p.weights,expand2(p.exp,p.feature_function(s),a)) #where is a sensible place to put w?
  end
  return domain(p.A)[indmax(Qs)]
end
value{S,T}(p::DiscretePolicy,s::S,a::T) = dot(p.weights,expand(p.exp,p.feature_function(s),a))
values{T}(p::DiscretePolicy,s::T) = [value(s,a) for a in p.A]
value{T}(p::DiscretePolicy,s::T) = maximum(values(p,s))

function save(p::DiscretePolicy,file) #no JldFile type available ...:(
  write(file,"weights",p.weights)
  write(file,"expander",p.exp)
  write(file,"actions",p.A)
end

function load_discretepolicy(file,feature_function::Function)
  #file = jldopen(fname,"r")
  weights = read(file,"weights")
  expander = read(file,"expander")
  A = read(file,"actions")
  #close(file)
  return DiscretePolicy(actions,feature_function,weights,expander)
end
######################################################################
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
  exp::ActionFeatureExpander
  function EpsilonGreedyPolicy(feature_function::Function,actions::DiscreteActionSpace;
                                rng::AbstractRNG=MersenneTwister(2983461),
                                eps::Float64=0.05,
                                exp::FeatureExpander=NullFeatureExpander(actions))
    self = new()
    self.rng = rng
    self.A = actions
    self.feature_function = feature_function
    self.eps = eps
    self.exp = exp
    return self
  end
end
range(p::EpsilonGreedyPolicy) = p.A
function action{T}(p::EpsilonGreedyPolicy,u::UpdaterParam,s::T)

  r = rand(p.rng)
  if r < p.eps
    return domain(p.A)[rand(p.rng,1:length(p.A))]
  end
  Qs = zeros(length(p.A))
  for (i,a) in enumerate(domain(p.A))
    Qs[i] = dot(weights(u),expand2(p.exp,p.feature_function(s),a)) #where is a sensible place to put w?
  end
  return domain(p.A)[indmax(Qs)]
end

function save(p::EpsilonGreedyPolicy,file) #no JldFile type available ...:(
  write(file,"weights",p.weights)
  write(file,"expander",p.exp)
  write(file,"actions",p.A)
  write(file,"eps",p.eps)
  write(file,"rng",p.rng)
end

function load_epsilongreedypolicy(file,feature_function::Function)
  #file = jldopen(fname,"r")
  weights = read(file,"weights")
  expander = read(file,"expander")
  A = read(file,"actions")
  rng = read(file,"rng")
  eps= read(file,"eps")
  #close(file)
  return EpsilonGreedyPolicy(rng,actions,feature_function,weights,eps,expander)
end
#########################################################################
type SoftmaxPolicy <: ExplorationPolicy
  rng::AbstractRNG
  A::DiscreteActionSpace #TODO
  feature_function::Function
  tau::Float64
  exp::ActionFeatureExpander
end
range(p::SoftmaxPolicy) = p.A

#using StatsBase.sample means RNG is useless, but I'm lazy
function action{T}(p::SoftmaxPolicy,u::UpdaterParam,s::T)
  Qs = zeros(length(p.A))
  for (i,a) in enumerate(domain(p.A))
    Qs[i] = exp(p.tau*dot(weights(u),expand2(p.exp,p.feature_function(s),a))) #where is a sensible place to put w?
  end
  return sample(domain(p.A),WeightVec(Qs))
end

function save(p::SoftmaxPolicy,file) #no JldFile type available ...:(
  write(file,"weights",p.weights)
  write(file,"expander",p.exp)
  write(file,"actions",p.A)
  write(file,"tau",p.tau)
  write(file,"rng",p.rng)
end

function load_softmaxpolicy(file,feature_function::Function)
  #file = jldopen(fname,"r")
  weights = read(file,"weights")
  expander = read(file,"expander")
  A = read(file,"actions")
  rng = read(file,"rng")
  tau = read(file,"tau")
  #close(file)
  return SoftmaxPolicy(rng,actions,feature_function,weights,tau,expander)
end
#########################################################################
#For continuous action spaces
#can potentially make this more generic with difference noise models
type GaussianPolicy <: ExplorationPolicy
  A::ContinuousActionSpace

end
function action{T}(p::GaussianPolicy,u::UpdaterParam,exp::FeatureExpander,s::T)
  #TODO
end

#NOTE: this miiiight not work and just pollute the namespace instead
Policy(p::Union{EpsilonGreedyPolicy,SoftmaxPolicy},u::UpdaterParam,exp::FeatureExpander) =
                        DiscretePolicy(range(p),p.feature_function,weights(u),exp)
