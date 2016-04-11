#__solvers.jl
#a spot to hold different reinforcement learning solvers for now before it gets
# broken up into individual files

#TODO: use enum
function init_weights(nb_feat::Int,method::AbstractString="unif_rand";
                      rng::AbstractRNG=MersenneTwister(193))
  if lowercase(method) == "unif_rand"
    return 2*(rand(rng,nb_feat)-0.5)/sqrt(nb_feat)
  elseif lowercase(method) == "zero"
    return zeros(nb_feat)
  elseif lowercase(method) == "one"
    return ones(nb_feat)
  elseif lowercase(method) == "optimistic"
    return 10*ones(nb_feat)
  elseif lowercase(method) == "rand"
    return rand(rng,nb_feat)
  elseif lowercase(method) == "half"
    return 0.5*ones(nb_feat)
  else
    error("No such initialization method: $method")
  end
end

#####################################################################
#TODO: reference my other implementation that actually works
include("ForgetfulLSTD.jl")
#####################################

#Deterministic Policy Gradient
type COPDACQ <: ActorCritic
  v::RealVector
  th::RealVector
  w::RealVector
  lr_th::Float64
  lr_v::Float64
  lr_w::Float64
  natural_gradient::Bool
  sigma::Union{Float64,Array{Float64,1}} #a hack to add gaussian exploration
  policy::Policy
  function COPDACQ(n::Int,policy::Policy; #nb feat, nb action
                    init_method::AbstractString="unif_rand",
                    lr_v::Float64=1.,
                    lr_w::Float64=1.,
                    lr_th::Float64=0.1,
                    natural_gradient::Bool=false,
                    sigma::Float64=1.)
    self = new()

    m = length(policy.A) #TODO def domain(::Policy)
    self.v = init_weights(n,init_method)
    self.w = init_weights(n*m,init_method)
    self.th = init_weights(n*m,init_method)
    #policy.weights = self.th
    self.policy = policy
    self.lr_th = lr_th
    self.lr_w = lr_w
    self.lr_v = lr_v
    if length(sigma) != 1
      assert(length(sigma) == length(policy.A))
    end
    self.sigma = sigma
    self.natural_gradient = natural_gradient
    return self
  end
end

#another hack Union{LinearPolicy,SigmoidPolicy,GaussianPolicy}

function action{T}(p::LinearPolicy,u::COPDACQ,s::T)
  #println("HI")
  Z = randn(length(p.A)).*u.sigma #TODO::RNG
  return p.action_type(bound(p.A,action(u.policy,u::ActorCritic,s,is_explore=true)+Z))
end
#have to do this because ambiguity reasons idk
function action{T}(p::SigmoidPolicy,u::COPDACQ,s::T)
  #println("HI")
  Z = randn(length(p.A)).*u.sigma #TODO::RNG
  return p.action_type(bound(p.A,action(u.policy,u::ActorCritic,s,is_explore=true)+Z))
end
weights(p::COPDACQ) = reshape(p.th,length(p.v),length(p.policy.A))

function pad!(p::COPDACQ,nb_new_feat::Int,nb_feat::Int)
  if nb_new_feat <= 0
    return
  end

  pad!(p.w,nb_new_feat,nb_feat)
  pad!(p.v,nb_new_feat,nb_feat)
  pad!(p.th,nb_new_feat,nb_feat)
end

function update!{T}(param::COPDACQ,
                  annealer::AnnealerParam,
                  mb::Minibatcher,
                  er::ExperienceReplayer,
                  exp::FeatureExpander,
                  gc::GradientClipper,
                  phi::RealVector,
                  a::T,
                  r::Real,
                  phi_::RealVector,
                  a_::T,
                  gamma::Float64,
                  lr::Float64)
  phi,a,r,phi_,a_ = replay!(er,phi,a,r,phi_,a_)
  #expand the state representation
  f = expand(exp,phi)
  f_ = expand(exp,phi_)
  #expand with respect to actions
  #NOTE: this might be a singleton array
  v = dot(param.v,f)
  v_ = dot(param.v,f_)

  J = jacobian(param.policy,weights(param),f)
  mu = action(param.policy,param,f,precalculated=true)

  advantage = transpose(vec(a)-vec(mu))*transpose(J)*param.w
  advantage = advantage[1]

  q = v + advantage #q_ = v_ since we are assumed to follow the actor Policy

  del = r + gamma*v_ - q

  if !param.natural_gradient
    dth = clip!(gc,J*(transpose(J)*param.w),1)
  else
    dth = clip!(gc,param.w,1)
  end
  #NOTE this should be length n*mx1, get reshaped in policy
    # and then the gradient should be zero for terms that relate to other actions
  param.th = vec(param.th + anneal!(annealer,minibatch!(mb,dth,1),param.lr_th*lr,1))

  f_sa = J*(vec(a)-vec(mu))
  dw = clip!(gc,del*f_sa,2)
  param.w = vec(param.w + anneal!(annealer,minibatch!(mb,dw,2),param.lr_w*lr,2))

  dv = clip!(gc,del*f,3)
  param.v = vec(param.v + anneal!(annealer,minibatch!(mb,dv,3),param.lr_v*lr,3))

  #param.policy.weights = param.th

  nb_new_feat = update!(exp,f,del)
  pad!(param,nb_new_feat,length(f))
  return del, q
end


############################################################################


type COPDACGQ <: ActorCritic
  v::RealVector
  th::RealVector
  w::RealVector
  u::RealVector
  lr_th::Float64
  lr_v::Float64
  lr_w::Float64
  lr_u::Float64
  policy::Policy
  sigma::Union{Float64,Array{Float64,1}} #a hack to add gaussian exploration
  natural_gradient::Bool
  function COPDACGQ(n::Int,policy::Policy; #nb feat, nb action
                    init_method::AbstractString="unif_rand",
                    lr_v::Float64=1.,
                    lr_w::Float64=1.,
                    lr_th::Float64=0.1,
                    lr_u::Float64=1e-4,
                    sigma::Float64=1.,
                    natural_gradient::Bool=false)
    self = new()

    m = length(policy.A) #should define domain(::Policy), but whatever
    self.v = init_weights(n,init_method)
    self.w = init_weights(n*m,init_method)
    self.th = init_weights(n*m,init_method)
    self.u = spzeros(n,1)
    self.policy = policy
    self.lr_th = lr_th
    self.lr_w = lr_w
    self.lr_v = lr_v
    self.lr_u = lr_u
    if length(sigma) != 1
      assert(length(sigma) == length(policy.A))
    end
    self.sigma = sigma
    self.natural_gradient = natural_gradient
    return self
  end
end

function action{T}(p::LinearPolicy,u::COPDACGQ,s::T)
  #println("HI")
  Z = randn(length(p.A)).*u.sigma #TODO::RNG
  return p.action_type(bound(p.A,action(u.policy,u::ActorCritic,s,is_explore=true)+Z))
end

function action{T}(p::SigmoidPolicy,u::COPDACGQ,s::T)
  #println("HI")
  Z = randn(length(p.A)).*u.sigma #TODO::RNG
  return p.action_type(bound(p.A,action(u.policy,u::ActorCritic,s,is_explore=true)+Z))
end
weights(p::COPDACGQ) = reshape(p.th,length(p.v),length(p.policy.A))

function pad!(p::COPDACGQ,nb_new_feat::Int,nb_feat::Int)
  if nb_new_feat <= 0
    return
  end

  pad!(p.w,nb_new_feat,nb_feat)
  pad!(p.v,nb_new_feat,nb_feat)
  pad!(p.u,nb_new_feat,nb_feat)
  pad!(p.th,nb_new_feat,nb_feat)
end

function update!{T}(param::COPDACGQ,
                  annealer::AnnealerParam,
                  mb::Minibatcher,
                  er::ExperienceReplayer,
                  exp::FeatureExpander,
                  gc::GradientClipper,
                  phi::RealVector,
                  a::T,
                  r::Real,
                  phi_::RealVector,
                  a_::T,
                  gamma::Float64,
                  lr::Float64)
  phi,a,r,phi_,a_ = replay!(er,phi,a,r,phi_,a_)
  #expand the state representation
  f = expand(exp,phi)
  f_ = expand(exp,phi_)
  #expand with respect to actions
  #NOTE: this might be a singleton array
  v = dot(param.v,f)
  v_ = dot(param.v,f_)

  J = jacobian(param.policy,weights(param),f)
  mu = action(param.policy,param,f,precalculated=true)

  advantage = transpose(vec(a)-vec(mu))*transpose(J)*param.w
  advantage = advantage[1]

  q = v + advantage #q_ = v_ since we are assumed to follow the actor Policy

  del = r + gamma*v_ - q

  if !param.natural_gradient
    dth = ckip!(gc,J*(transpose(J)*param.w),1)
  else
    dth = clip!(gc,param.w,1)
  end
  param.th = vec(param.th + anneal!(annealer,minibatch!(mb,dth,1),param.lr_th*lr,1))

  f_sa = J*(vec(a)-vec(mu))
  dw = clip!(gc,del*f_sa,2) #- gamma*f_sa*dot(f_sa,param.u)
  #I think the MSPBE correction here is just zero since a_ = mu_
  param.w = vec(param.w + anneal!(annealer,minibatch!(mb,dw,2),param.lr_w*lr,2))

  dv = clip!(gc,del*f - gamma*f_*dot(f_sa,param.u),3)
  param.v = vec(param.v + anneal!(annealer,minibatch!(mb,dv,3),param.lr_v*lr,3))

  du = clip!(gc,(del-dot(f_sa,param.u))*f_sa,4)
  param.u = vec(param.u + anneal!(annealer,minibatch!(mb,du,4),param.lr_u*lr,4))

  #param.policy.weights = param.th

  nb_new_feat = update!(exp,f,del)
  pad!(param,nb_new_feat,length(f))
  return del, q
end


#####################################
#TODO: actor critic models--will do survey and see if I can subset down to some atomic
#     classes
type StochasticActorCritic <: ActorCritic
  r_avg::Float64
  lambda::Float64
  v::RealVector
  u::RealMatrix
  e_u::RealVector
  e_v::RealVector
  policy::Policy
  lr_v::Float64 #relative to core learning rate
  lr_u::Float64
  lr_r::Float64
  gradient_scaling::Bool
  is_replacing_trace::Bool
  function StochasticActorCritic(n::Int,policy::Policy;
                            lr_v::Float64=1.,
                            lr_u::Float64=0.01,
                            lr_r::Float64=0.02,
                            gradient_scaling::Bool=false,
                            lambda::Float64=0.95,
                            criticless::Bool=false,
                            init_method::AbstractString="unif_rand",
                            is_replacing_trace::Bool=true)
    if criticless
      lr_v = 0.
    end
    self = new()

    self.r_avg = 0.
    self.lambda=lambda
    self.v = init_weights(n,init_method)
    self.u = init_weights(n,init_method)
    self.e_u = spzeros(n,1)
    self.e_v = spzeros(n,1)
    self.policy = policy
    self.lr_v = lr_v
    self.lr_u = lr_u
    self.lr_r = lr_r
    self.gradient_scaling = gradient_scaling
    self.is_replacing_trace = is_replacing_trace

    return self
  end
end
weights(u::StochasticActorCritic) = u.u

#TODO figure out how to modify pad!
function pad!(p::StochasticActorCritic,nb_new_feat::Int,nb_feat::Int)
  if nb_new_feat <= 0
    return
  end
  pad!(p.v,nb_new_feat,nb_feat)
  pad!(p.u,nb_new_feat,nb_feat)
  pad!(p.e_u,nb_new_feat,nb_feat,p.lambda) #since if this was a new feature, then the trace was active
  pad!(pe._v,nb_new_feat,nb_feat)
end

#NOTE: single step GQ for now (no eligbility traces)
function update!{T}(param::StochasticActorCritic,
                  annealer::AnnealerParam,
                  mb::Minibatcher,
                  er::ExperienceReplayer,
                  exp::FeatureExpander,
                  gc::GradientClipper,
                  phi::RealVector,
                  a::T,
                  r::Real,
                  phi_::RealVector,
                  a_::T,
                  gamma::Float64,
                  lr::Float64)
  phi,a,r,phi_,a_ = replay!(er,phi,a,r,phi_,a_)
  #expand the state representation
  f = expand(exp,phi)
  f_ = expand(exp,phi_)
  #expand with respect to actions
  #NOTE: this might be a singleton array
  v = dot(param.v,f)
  v_ = dot(param.v,f_)
  del = r - param.r_avg + gamma*v_ - v
  param.r_avg = param.r_avg + lr*param.lr_r*del
  # end step
  if param.is_replacing_trace
    param.e_v = vec(max(phi,param.e_v*gamma*param.lambda)) #NOTE: assumes binary features
  else
    param.e_v = vec(phi + param.e_v*gamma*param.lambda)
  end
  dv = clip!(gc,del*param.e_v,1)
  param.v = param.v + anneal!(annealer,minibatch!(mb,dv,1),param.lr_v*lr,1) #MINIBATCH + ANNEALING TODO
  param.e_u = gamma*param.lambda*param.e_u + loggradient(param.policy,f)#"gradient term--TODO"
  du = clip!(gc,del*param.e_u,2)
  lr_u = lr*param.lr_u
  if param.gradient_scaling
    lr_u = l*= sig2 # FROM GRADIENT TERM TODO
  end
  param.u = param.u + anneal!(annealer,minibatch!(mb,du,2),lr_u,2)
  param.policy.weights = param.u
  nb_new_feat = update!(exp,f,del)
  pad!(param,nb_new_feat,length(f))
  return del, q
end

#######################################################################

#Incremental Natural Actor Critic (INAC)
#TODO
#TODO make some kind of policy distribution type for which we can define REINFORCE update




#######################################
#TODO: use enum
include("SARSA.jl")

#######################################
include("QLearning.jl")
######################################
include("GQLearning.jl")

######################################
include("TrueOnlineTD.jl")

######################################
#TODO: special annealer type for this solver?
include("DoubleQLearning.jl")

###############################################
#LSPolicyIteration
include("LSPI.jl")

#####################################################################
#Model Predictive Control
#=
  NOTE: do something like: save result from last time step to warmstart solution
    for current time step
=#

###########################
include("MPC.jl")
#####################################################################
### FUTURE STUFF ###
# Natural Actor critic
# Deterministic Policy gradient
# Stochastic Actor critic
# Finite Different Policy gradient
# Natural Policy gradient
# Likelihood weighed policy gradient
# Cross-Entropy Policy Optimization


#=
## For Monte Carlo Simulation based solvers (i.e. not updating based on SARSA):
function solve(updater,bbm)
  for iter = 1:updater.max_iter
    #how to do individual simulations, but maintain whatver important stats we need?
    #also, keeping track of permutations on policies
    #verbose stuff
    if stopping_crit(updater.solver)
      break
    end
  end
end
=#

#TODO some kind of permuation/noise model

####################################################################

#Finite difference policy gradient
type FiniteDifferencePolicyGradient <: MCUpdater
  policy::Policy #actual representation arbitrary--get squashed weights out via weights()
  eps::Float64 #fixed with or sigma for
  is_gaussian::Bool #else unif_rand
  tol::Float64
  abs_tol::Bool #else rel_tol
end


type ReinforcePolicyGradient <: MCUpdater
  policy::Policy
  tol::Float64
  abs_tol::Bool #else rel_tol
end


type NaturalPolicyGradient <: MCUpdater
  policy::Policy
  tol::Float64
  abs_tol::Bool #else rel_tol
end
#################################################################

type CrossEntropy <: MCUpdater
  policy::Policy
#  proposal_distr::distribution
  nb_elite_samples::Int
end
#=
function sufficient_statistics(updater::CrossEntropy,history::MCHistory)

end
=#
