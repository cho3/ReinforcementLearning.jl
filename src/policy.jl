#policy.jl
#holds the exploration and frozen policies and action space stuff and whatever
import Base.length

type DiscreteActionSpace{T} <: ActionSpace
  A::Array{T,1}
end
domain(a::DiscreteActionSpace) = a.A
length(a::DiscreteActionSpace) = length(a.A)

type ContinuousActionSpace <: ActionSpace
  ub::RealVector
  lb::RealVector
  function ContinuousActionSpace(;n::Int=1,lb::Union{RealVector,Real}=zeros(n),ub::Union{RealVector,Real}=ones(n))
    if length(ub) != 1 || length(lb) != 1
      assert(length(ub) == length(lb))
    end
    if length(ub) == 1
      ub = ub*ones(max(length(lb),n))
    end
    if length(lb) == 1
      lb = lb*ones(max(length(ub),n))
    end
    assert(!(false in (lb .< ub)))
    self = new()

    self.ub = ub
    self.lb = lb

    return self
  end
end
length(a::ContinuousActionSpace) = length(a.ub)
function bound(a::ContinuousActionSpace,x::RealVector)
  assert(length(x) == length(a))
  return max(min(x,a.ub),a.lb)
end
bound(a::ContinuousActionSpace,x::Real,i::Int) = max(min(x,a.ub[i]),a.lb[i])

type NullPolicy <: Policy end

action(::Policy,s) = error("Policy type uninstantiated")
range(::Policy) = error("Policy type uninstantiated")
init!(p::Policy) = init!(p.exp)
behavior(::Policy) = 0 #or an actual policy

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

update!(p::Policy,w) = begin p.weights += w end
set!(p::Policy,w) = begin p.weights = w end
squash(p::Policy) = vec(p.weights)
desquash(p::Policy,w::Union{RealMatrix,RealVector}) = reshape(w,size(p.weights))
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
  length(expand2(p.exp,p.feature_function(init(bbm)),domain(range(p))[1]))

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
  behavior::Policy
  function EpsilonGreedyPolicy(feature_function::Function,actions::DiscreteActionSpace;
                                rng::AbstractRNG=MersenneTwister(2983461),
                                eps::Float64=0.05,
                                exp::FeatureExpander=NullFeatureExpander(actions),
                                behavior::Policy=NullPolicy())
    self = new()
    self.rng = rng
    self.A = actions
    self.feature_function = feature_function
    self.eps = eps
    self.exp = exp
    self.behavior = behavior
    return self
  end
end
behavior(p::EpsilonGreedyPolicy) = typeof(p.behavior) != NullPolicy ? p.behavior : 0
range(p::EpsilonGreedyPolicy) = p.A
function action{T}(p::EpsilonGreedyPolicy,u::UpdaterParam,s::T)
  #breaking this out because expand might have memory things....
  f = expand(p.exp,p.feature_function(s))
  r = rand(p.rng)
  if r < p.eps
    return domain(p.A)[rand(p.rng,1:length(p.A))]
  end
  b = behavior(p)
  if b != 0
    return action(b,s)
  end
  Qs = zeros(length(p.A))
  for (i,a) in enumerate(domain(p.A))
    #println(size(weights(u)))
    phi = expand(p.exp,f,a)
    #println(size(f))
    #println(size(weights(u)))
    Qs[i] = dot(weights(u),phi)
    #Qs[i] = dot(weights(u),expand2(p.exp,p.feature_function(s),a)) #where is a sensible place to put w?
  end
  return domain(p.A)[indmax(Qs)]
end

action{T}(p::EpsilonGreedyPolicy,u::ActorCritic,s::T) = error("Actor Critic does not work with epsilon greedy exploration")

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

action{T}(p::SoftmaxPolicy,u::ActorCritic,s::T) = action(p,u,s)

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
type GaussianExplorationPolicy <: ExplorationPolicy
  A::ContinuousActionSpace

end
function action{T}(p::GaussianExplorationPolicy,u::UpdaterParam,exp::FeatureExpander,s::T)
  #TODO
end


#########################################################################
###Continuous Stuff/Actor Critic Stuff####

action{T}(p::Policy,u::ActorCritic,s::T) = p.action_type(action(u.policy,u,s,true))
########################################################################

type LinearPolicy <: Policy
  A::ContinuousActionSpace
  #weights::RealMatrix
  feature_function::Function
  exp::FeatureExpander
  action_type::DataType
  rng::AbstractRNG
  function LinearPolicy(n::Int,
                        A::ContinuousActionSpace,
                        ff::Function,
                        action_type::DataType;
                        exp::FeatureExpander=TrueNullFeatureExpander(),
                        rng::AbstractRNG=MersenneTwister(999))
    self = new()

    self.A = A
    self.feature_function = ff
    self.exp = exp
    self.action_type = action_type
    #self.weights = zeros(n,length(A))
    self.rng = rng
    return self
  end
end
# TODO make sure it can work with state only expansions-- ayy expand() vs expand2()

#Infinite hacking
action{T}(p::LinearPolicy,u::ActorCritic,s::T;is_explore::Bool=false,precalculated::Bool=false) =
    action(p,weights(u),s,is_explore=is_explore,precalculated=precalculated)
function action{T}(p::LinearPolicy,w::RealMatrix,s::T;is_explore::Bool=false,precalculated::Bool=false)
  if precalculated
    phi = s
  else
    phi = expand(p.exp,p.feature_function(s))
  end
  a_vec = vec(transpose(w)*phi)
  if is_explore
    return a_vec
  end
  a_vec = bound(p.A,a_vec)
  return p.action_type(a_vec)
end

#weights(p::LinearPolicy) = vec(p.weights)

function ident(x::RealMatrix)
  X = zeros(length(x),size(x,2))
  l = size(x,1)
  offset = 0
  for j = 1: size(x,2)
    X[1+offset:l+offset,j] = X[:,j]
    offset += l
  end
  return X
end
#alternatively: blkdiag


gradient(p::LinearPolicy,w::RealMatrix,phi::RealVector,i::Int) = phi#expand(p.exp,p.feature_function(s))
jacobian(p::LinearPolicy,w::RealMatrix,phi::RealVector) = blkdiag([sparse(gradient(p,w,phi,1)) for i = 1:length(p.A)]...)#repmat(gradient(p,w,phi,1),1,length(p.A))

function loggradient(p::LinearPolicy,weights::RealMatrix,phi::RealVector,i::Int)
  #linear gradient for each index, divided by the action
  #phi = expand(p.exp,p.feature_function(s))
  a = dot(weights[:,i],phi)
  return phi./bound(p.A,a,i)
end

function loggradient(p::LinearPolicy,weights::RealMatrix,phi::RealVector)
  _J = [loggradient(p,weights,phi,i) for i = 1:length(p.A)]
  J = zeros(length(_J[1]),length(_J))
  for i = 1:length(p.A)
    J[:,i] = _J[i]
  end
  return J
end


#########################################################################

type SigmoidPolicy <: Policy
    A::ContinuousActionSpace
    #weights::RealMatrix
    feature_function::Function
    exp::FeatureExpander
    action_type::DataType
    rng::AbstractRNG
    function SigmoidPolicy(n::Int,
                          A::ContinuousActionSpace,
                          ff::Function,
                          action_type::DataType;
                          exp::FeatureExpander=TrueNullFeatureExpander(),
                          rng::AbstractRNG=MersenneTwister(999))
      self = new()

      self.A = A
      self.feature_function = ff
      self.exp = exp
      self.action_type = action_type
      #self.weights = zeros(n,length(A))
      self.rng = rng
      return self
    end
end

sigmoid(x::Real) = 1./(1+exp(-x))
sigmoid(x::AbstractArray) = [sigmoid(y) for y in x]

action{T}(p::SigmoidPolicy,u::ActorCritic,s::T;is_explore::Bool=false,precalculated::Bool=false) =
    action(p,weights(u),s,is_explore=is_explore,precalculated=precalculated)
function action{T}(p::SigmoidPolicy,w::RealMatrix,s::T;is_explore::Bool=false,precalculated::Bool=false)
  if precalculated
    phi = s
  else
    phi = expand(p.exp,p.feature_function(s))
  end
  a_vec = sigmoid(vec(transpose(w)*phi)) #[0,1]
  a_vec = a_vec.*(p.A.ub-p.A.lb)+p.A.lb #[lb,ub]
  if is_explore
    return a_vec
  end
  return p.action_type(a_vec)
end

#weights(p::SigmoidPolicy) = vec(p.weights)

function gradient(p::SigmoidPolicy,weights::RealMatrix,phi::RealVector,i::Int)
  #phi = expand(p.exp,p.feature_function(s))
  z = dot(weights[:,i],phi)
  sz = sigmoid(z)
  return sz.*(1-sz).*phi
end

function jacobian(p::SigmoidPolicy,weights::RealMatrix,phi::RealVector)
  #phi = expand(p.exp,p.feature_function(s))
  _J = [gradient(p,weights,phi,i) for i = 1:length(p.A)]
  J = zeros(length(_J[1]),length(_J))
  for i = 1:length(p.A)
    J[:,i] = _J[i]
  end
  return J
end

function loggradient(p::SigmoidPolicy,weights::RealMatrix,phi::RealVector,i::Int)
  #linear gradient for each index, divided by the action
  #phi = expand(p.exp,p.feature_function(s))
  phi = expand(p.exp,p.feature_function(s))
  z = dot(weights[:,i],phi)
  sz = sigmoid(z)
  return (1-sz).*phi
end

function loggradient(p::SigmoidPolicy,weights::RealMatrix,phi::RealVector)
  _J = [loggradient(p,weights,phi,i) for i = 1:length(p.A)]
  J = zeros(length(_J[1]),length(_J))
  for i = 1:length(p.A)
    J[:,i] = _J[i]
  end
  return J
end

#########################################################################
type GaussianPolicy <: Policy
  A::ContinuousActionSpace
  #weights::RealMatrix
  feature_function::Function
  exp::FeatureExpander
  action_type::DataType
  sigma::Union{Array{Float64,1},Float64} #if < 0 then we calculate it
  rng::AbstractRNG
  function GaussianPolicy(n::Int,
                        A::ContinuousActionSpace,
                        ff::Function,
                        action_type::DataType;
                        exp::FeatureExpander=TrueNullFeatureExpander(),
                        sigma::Float64=1.,
                        rng::AbstractRNG=MersenneTwister(999))
    self = new()

    self.A = A
    self.feature_function = ff
    self.exp = exp
    self.action_type = action_type
    #TODO logic for sigma stuff
    if sigma < 0.
      self.weights = zeros(n,2*length(A)) #sigma is parameterized
      self.sigma = -1.
    else
      self.weights = zeros(n,2*length(A))
      self.sigma = sigma
    end

    self.rng = rng
    return self
  end
end

is_sigma_param(p::GaussianPolicy) = p.sigma[1] < 0

action{T}(p::GaussianPolicy,u::ActorCritic,s::T;is_explore::Bool=false,precalculated::Bool=false) =
    action(p,weights(u),s,is_explore=is_explore,precalculated=precalculated)
function action{T}(p::GaussianPolicy,w::RealMatrix,s::T;is_explore::Bool=false,precalculated::Bool=false)
  if precalculated
    phi = s
  else
    phi = expand(p.exp,p.feature_function(s))
  end
  sigma = p.sigma
  mu_indices = collect(1:length(p.A))
  if is_sigma_param(p)
    assert(size(w,2) == length(p.A)*2)
    #calculate sigma
    sigma = abc
    #update mu_indices
    mu_indices = collect(1:2:length(p.A)*2)
  end
  mu = vec(transpose(w[:,mu_indices])*phi)
  Z = randn(p.rng,length(mu_indices)).*sigma
  if is_explore
    return a_vec
  end
  return p.action_type(bound(p.A,mu+Z))
end

function loggradient{S}(p::GaussianPolicy,phi::RealVector,a::S,i::Int)
  #calculate mu sigma, phi TODO
  phi = 0.
  mu = 0.
  sigma = 0.
  grad_u = phi*(a[i]-mu)/(sigma^2)
  if !is_sigma_param(p)
    return grad_u
  end
  grad_v = phi*((((a[i]-mu)/sigma)^2)-1.)
  return hcat(grad_u,grad_v)
end
#########################################################################

type FiniteStateController <: Policy
  A::DiscreteActionSpace #or potentially a regressor at each node, but whatever
  psi::Array{RealMatrix,1} #for each internal state: prob of each action/softmax matrix for each action given a feature vector
  eta::Array{RealMatrix,1} #for each internal state: probability of shifting to another state given an observation
  feature_function::Function
  exp::FeatureExpander
  current_state::Int #idx of current state
  rng::AbstractRNG
end

update!(fsc::FiniteStateController,w) = begin fsc.psi += w[1]; fsc.eta += w[2] end
set!(fsc::FiniteStateController,w) = begin fsc.psi = w[1]; fsc.eta = w[2] end
init!(fsc::FiniteStateController) = begin fsc.current_state=1 end

function action{T}(fsc::FiniteStateController,o::T)
  phi = expand(fsc.exp,fsc.feature_function(o))
  #transition internal state based on observation
  #r = rand(fsc.rng)
  w = WeightVec(exp(eta[fsc.current_state]*phi))
  fsc.current_state = sample(fsc.rng,w)

  #emit action according to softmax probability
  v = WeightVec(exp(psi[fsc.current_state]*phi))
  adx = sample(fsc.rng,v)
  return domain(fsc.A)[adx]
end


function squash(fsc::FiniteStateController)
  u = zeros(sum([length(p) for p in fsc.psi])+sum([length(p) for p in fsc.eta]))
  offset = 0
  for p in fsc.psi
    v = vec(p)
    d = length(v)
    u[1+offset:d+offset] = v
    offset += d
  end
  for p in fsc.eta
    v = vec(p)
    d = length(v)
    u[1+offset:d+offset] = v
    offset += d
  end
  return u
end

function desquash(fsc::FiniteStateController,phi::RealVector)
  psi = [zeros(1,1) for _ in fsc.psi]
  eta = [zeros(1,1) for _ in fsc.eta]
  offset = 0
  for (i,p) in enumerate(fsc.psi)
    psi[i] = reshape(phi[1+offset:length(p)+offset],size(p))
  end
  for (i,p) in enumerate(fsc.eta)
    eta[i] = reshape(phi[1+offset:length(p)+offset],size(p))
  end
  return psi, eta
end

function loggradient{S}(fsc::FiniteStateController,phi::RealVector,a::S,i::Int)

end

function gradient{S}(fsc::FiniteStateController,phi::RealVector,a::S,i::Int)

end
##########################################################################
#NOTE: this miiiight not work and just pollute the namespace instead
Policy(p::Union{EpsilonGreedyPolicy,SoftmaxPolicy},u::UpdaterParam,exp::FeatureExpander) =
                        DiscretePolicy(range(p),p.feature_function,weights(u),deepcopy(exp))

#TODO add expander
type ContinuousPolicy <: Policy
  p::Policy
  w::RealMatrix
end
update!(p::ContinuousPolicy,w::RealMatrix) = begin p.w += w end
set!(p::ContinuousPolicy,w::RealMatrix) = begin p.w = w end
squash(p::ContinuousPolicy) = vec(p.w)
desquash(p::ContinuousPolicy,w::RealVector) = reshape(w,size(p.w))
Policy(p::Union{SigmoidPolicy,LinearPolicy,GaussianPolicy},u::ActorCritic,exp::FeatureExpander) =
  ContinuousPolicy(p,weights(u))

action{T}(p::ContinuousPolicy,s::T) = action(p.p,p.w,s)
