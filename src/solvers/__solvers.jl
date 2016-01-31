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
  else
    error("No such initialization method: $method")
  end
end

#####################################################################
#TODO: reference my other implementation that actually works
type ForgetfulLSTDParam <: UpdaterParam
  alpha::Float64
  beta::Float64
  lambda::Float64
  k::Int
  e::RealVector
  th::RealVector
  d::RealVector
  A::RealMatrix
  #TODO: constructor
  function ForgetfulLSTDParam(nb_feat::Int;
                              alpha::Float64=0.01/sqrt(nb_feat),
                              lambda::Float64=0.95,
                              k::Int=1,
                              init_method::AbstractString="zero")
    #This is actual ForgetfulLSTD
    self = new()
    self.alpha = alpha
    self.beta = alpha
    self.lambda = lambda
    self.th = init_weights(nb_feat,init_method) #TODO: init_method...
    self.e = spzeros(nb_feat,1)#zeros(nb_feat)
    self.d = deepcopy(self.th)./alpha
    self.A = speye(nb_feat)/alpha

    return self
  end
end
weights(u::ForgetfulLSTDParam) = u.th

function update!{T}(param::ForgetfulLSTDParam,
                  annealer::AnnealerParam,
                  mb::Minibatcher,
                  er::ExperienceReplayer,
                  exp::FeatureExpander,
                  phi::RealVector,
                  a::T,
                  r::Real,
                  phi_::RealVector,
                  a_::T,
                  gamma::Float64,
                  lr::Float64)

  param.e = vec(param.e -param.beta*phi*dot(phi,param.e) + phi)
  param.A = param.A -(param.beta*phi)*(phi'*param.A) + param.e*transpose(phi-gamma*phi_)
  param.d = vec(param.d -param.beta*phi*dot(phi,param.d) + param.e*r)
  param.e = vec(gamma*param.lambda*param.e)
  #TODO: figure out how to apply minibatching, per parameter learning rates, and/or experience replay
  for i = 1:param.k
    param.th += param.alpha*(param.d-param.A*param.th)
  end
  return 0., 0. #not sure what the estim is off the top of my head
end
#####################################

type DeterministicPolicyGradientParam <: UpdaterParam

end

#####################################
#TODO: actor critic models--will do survey and see if I can subset down to some atomic
#     classes
type ActorCriticParam <: UpdaterParam

end

#######################################
#TODO: use enum
type SARSAParam <: UpdaterParam
  lambda::Float64 #the eligility trace parameters
  w::RealVector #weight vector
  e::RealVector #eligibility trace
  is_replacing_trace::Bool #i only know of two trace updates
  function SARSAParam(n::Int;
                      lambda::Float64=0.5,
                      init_method::AbstractString="unif_rand",
                      trace_type::AbstractString="replacing")
    self = new()
    self.w = init_weights(n,init_method) #or something
    self.e = spzeros(n,1)
    self.is_replacing_trace = lowercase(trace_type) == "replacing"
    self.lambda = lambda
    return self
  end
end
weights(u::SARSAParam) = u.w
function pad!(p::SARSAParam,nb_new_feat::Int,nb_feat::Int)
  if nb_new_feat <= 0
    return
  end
  pad!(p.w,nb_new_feat,nb_feat)
  pad!(p.e,nb_new_feat,nb_feat)
end

function update!{T}(param::SARSAParam,
                  annealer::AnnealerParam,
                  mb::Minibatcher,
                  er::ExperienceReplayer,
                  exp::FeatureExpander,
                  phi::RealVector,
                  a::T,
                  r::Real,
                  phi_::RealVector,
                  a_::T,
                  gamma::Float64,
                  lr::Float64)
  phi,a,r,phi_,a_ = replay!(er,phi,a,r,phi_,a_)

  f = expand(exp,phi)
  f_ = expand(exp,phi_)
  phi = expand(exp::ActionFeatureExpander,f,a)
  phi_ = expand(exp::ActionFeatureExpander,f_,a_)
  #NOTE: this might be a singleton array
  q = dot(param.w,phi) #TODO: dealing with feature functions that involve state and action?
  q_ = dot(param.w,phi_)
  del = r + gamma*q_ - q #td error
  if param.is_replacing_trace
    param.e = vec(max(phi,param.e)) #NOTE: assumes binary features
  else
    param.e = vec(phi + param.e)
  end
  dw = vec(del*param.e)
  param.w = vec(param.w + anneal!(annealer,minibatch!(mb,dw),lr))
  param.e = gamma*param.lambda*param.e
  nb_new_feat = update!(exp,f,del)
  pad!(param,nb_new_feat,length(f))
  return del, q
end

#######################################
type QParam <: UpdaterParam
  w::RealVector
  e::RealVector
  lambda::Float64
  A::DiscreteActionSpace
  #feature_function::Function
  is_replacing_trace::Bool
  function QParam(n::Int,A::DiscreteActionSpace;
                  lambda::Float64=0.95,
                  init_method::AbstractString="unif_rand",
                  trace_type::AbstractString="replacing")
    self = new()
    self.w = init_weights(n,init_method)
    self.e = spzeros(n,1)
    self.lambda = lambda
    self.is_replacing_trace = lowercase(trace_type) == "replacing"
    self.A = A
    #self.feature_function = ff

    return self
  end
end
weights(p::QParam) = p.w
function pad!(p::QParam,nb_new_feat::Int,nb_feat::Int)
  if nb_new_feat <= 0
    return
  end
  pad!(p.w,nb_new_feat,nb_feat)
  pad!(p.e,nb_new_feat,nb_feat)
end

function update!{T}(param::QParam,
                  annealer::AnnealerParam,
                  mb::Minibatcher,
                  er::ExperienceReplayer,
                  exp::FeatureExpander,
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
  phi = expand(exp::ActionFeatureExpander,f,a)
  #phi_ = expand(exp::ActionFeatureExpander,f_,a_)
  #NOTE: this might be a singleton array
  q = dot(param.w,phi) #TODO: dealing with feature functions that involve state and action?
  q_ = maximum([dot(param.w,expand(exp::ActionFeatureExpander,f_,_a)) for _a in domain(param.A)])
  del = r + gamma*q_ - q #td error
  if param.is_replacing_trace
    param.e = vec(max(phi,param.e)) #NOTE: assumes binary features
  else
    param.e = vec(phi + param.e)
  end
  dw = vec(del*param.e)
  param.w = vec(param.w + anneal!(annealer,minibatch!(mb,dw),lr))
  param.e = gamma*param.lambda*param.e
  nb_new_feat = update!(exp,f,del)
  pad!(param,nb_new_feat,length(f))
  return del, q
end

######################################
type GQParam <: UpdaterParam
  w::RealVector
  th::RealVector
  #e::RealVector
  b::Float64 #learning rate for gradient corrector
  #lambda::Float64
  #rho::Float64
  A::DiscreteActionSpace
  #feature_function::Function
  is_replacing_trace::Bool
  function GQParam(n::Int,A::DiscreteActionSpace;
                  #lambda::Float64=0.95,
                  b::Float64=1e-7,
                  init_method::AbstractString="unif_rand"
                  #trace_type::AbstractString="replacing"
                  )
    self = new()
    self.w = spzeros(n,1)#init_weights(n,"zero")
    self.th = init_weights(n,init_method)
    self.b = b
    #self.e = spzeros(n,1)
    #self.lambda = lambda
    #self.is_replacing_trace = lowercase(trace_type) == "replacing"
    self.A = A
    #self.feature_function = ff

    return self
  end
end
weights(p::GQParam) = p.th
function pad!(p::GQParam,nb_new_feat::Int,nb_feat::Int)
  if nb_new_feat <= 0
    return
  end
  pad!(p.w,nb_new_feat,nb_feat)
  pad!(p.th,nb_new_feat,nb_feat)
end

#NOTE: single step GQ for now (no eligbility traces)
function update!{T}(param::GQParam,
                  annealer::AnnealerParam,
                  mb::Minibatcher,
                  er::ExperienceReplayer,
                  exp::FeatureExpander,
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
  phi = expand(exp::ActionFeatureExpander,f,a)
  #phi_ = expand(exp::ActionFeatureExpander,f_,a_)
  #NOTE: this might be a singleton array
  q = dot(param.th,phi) #TODO: dealing with feature functions that involve state and action?
  #extracting out the best next action--semi-redundant step
  phi_s = [expand(exp::ActionFeatureExpander,f_,_a) for _a in domain(param.A)]
  qs_ = [dot(param.th,v) for v in phi_s]
  q_ind_ = indmax(qs_)
  q_ = qs_[q_ind_]
  phi_ = phi_s[q_ind_]
  # end step
  del = r + gamma*q_ - q #td error
  dth = vec(del*phi-gamma*dot(param.w,phi)*phi_)
  param.th = vec(param.th + anneal!(annealer,minibatch!(mb,dth),lr))
  param.w = vec(param.w + param.b*(del-dot(param.w,phi))*phi)
  nb_new_feat = update!(exp,f,del)
  pad!(param,nb_new_feat,length(f))
  return del, q
end

######################################
type TrueOnlineTDParam <: UpdaterParam
  lambda::Float64 #the eligility trace parameters
  w::RealVector #weight vector
  e::RealVector #eligibility trace
  q_old::Float64 #correction term
  is_replacing_trace::Bool #i only know of two trace updates
  function TrueOnlineTDParam(n::Int;
                      lambda::Float64=0.5,
                      init_method::AbstractString="unif_rand",
                      trace_type::AbstractString="replacing")
    self = new()
    self.w = init_weights(n,init_method) #or something
    self.e = zeros(n)
    self.is_replacing_trace = lowercase(trace_type) == "replacing"
    self.lambda = lambda
    self.q_old = 0.
    return self
  end
end
weights(u::TrueOnlineTDParam) = u.w

function pad!(p::TrueOnlineTDParam,nb_new_feat::Int,nb_feat::Int)
  if nb_new_feat <= 0
    return
  end
  pad!(p.w,nb_new_feat,nb_feat)
  pad!(p.e,nb_new_feat,nb_feat)
end

function update!{T}(param::TrueOnlineTDParam,
                  annealer::AnnealerParam,
                  mb::Minibatcher,
                  er::ExperienceReplayer,
                  exp::FeatureExpander,
                  phi::RealVector,
                  a::T,
                  r::Real,
                  phi_::RealVector,
                  a_::T,
                  gamma::Float64,
                  lr::Float64)
  phi,a,r,phi_,a_ = replay!(er,phi,a,r,phi_,a_)

  f = expand(exp,phi)
  f_ = expand(exp,phi_)
  phi = expand(exp::ActionFeatureExpander,f,a)
  phi_ = expand(exp::ActionFeatureExpander,f_,a_)
  #NOTE: this might be a singleton array
  q = dot(param.w,phi) #TODO: dealing with feature functions that involve state and action?
  q_ = dot(param.w,phi_)
  del = r + gamma*q_ - q #td error
  param.e = vec(gamma*param.lambda*param.e + phi - lr*gamma*param.lambda*dot(param.e,phi)*phi)
  dw = vec((del+q-param.q_old)*param.e - (q - param.q_old)*phi)
  param.w += anneal!(annealer,minibatch!(mb,dw),lr)
  param.q_old = q_
  nb_new_feat = update!(exp,f,del)
  pad!(param,nb_new_feat,length(f))
  return del, q
end

######################################
#TODO: special annealer type for this solver?
type DoubleAnnealer <: AnnealerParam
  A::AnnealerParam
  B::AnnealerParam
end
DoubleAnnealer(an::AnnealerParam) = DoubleAnnealer(an,deepcopy(an))

type DoubleMinibatcher <: Minibatcher
  A::Minibatcher
  B::Minibatcher
end
DoubleMinibatcher(mb::Minibatcher) = DoubleMinibatcher(mb,deepcopy(mb))

type DoubleQParam <: UpdaterParam
  wA::RealVector
  wB::RealVector
  updatingA::Bool
  rng::AbstractRNG
  is_deterministic_switch::Bool
  feature_function::Function
end

function pad!(p::DoubleQParam,nb_new_feat::Int,nb_feat::Int)
  if nb_new_feat <= 0
    return
  end
  pad!(p.wA,nb_new_feat,nb_feat)
  pad!(p.wB,nb_new_feat,nb_feat)
end

function update!{T}(param::DoubleQParam,
                    annealer::DoubleAnnealer,
                    mb::DoubleMinibatcher,
                    er::ExperienceReplayer,
                    exp::FeatureExpander,
                    phi::RealVector,
                    a::T,
                    r::Union{Float64,Int},
                    phi_::RealVector,
                    a_::T,
                    gamma::Float64,
                    lr::Float64)
  phi,a,r,phi_,a_ = replay!(er,phi,a,r,phi_,a_)

  f = expand(exp,phi)
  f_ = expand(exp,phi_)
  phi = expand(exp::ActionFeatureExpander,f,a)
  phi_ = expand(exp::ActionFeatureExpander,f_,a_)

  updateAflag = false
  if param.is_deterministic_switch
    param.updatingA = !param.updatingA
    updateAflag = param.updatingA
  else
    updateAflag = rand(param.rng,Bool)
  end
  if updateAflag
    QB_ = dot(param.wB,phi_,a_) #TODO Max over feature function
    QA = dot(param.wA,phi,a)
    del = r + discount*QB_ - QA
    dw = del*phi
    param.wA += anneal!(annealer.B,minibatch!(mb.B,dw),lr)
    nb_new_feat = update!(exp,f,del)
    pad!(param,nb_new_feat,length(f))
    return del, QA
  else
    QA_ = dot(param.wA,phi_,a_)
    QB = dot(param.wB,phi,a)
    del = r + discount*QA_ - QB
    dw = del*phi
    param.wB += anneal!(annealer.B,minibatch!(mb.B,dw),lr)
    nb_new_feat = update!(exp,f,del)
    pad!(param,nb_new_feat,length(f))
    return del, QB
  end
end

###############################################
#LSPolicyIteration


type LSPIParam <: UpdaterParam
  w::RealVector
  d::Int
  D::Int
  B::RealMatrix
  b::RealVector
  del::Float64
  discount::Float64
  tol::Float64
  done_flag::Bool
  function LSPIParam(nb_feat::Int, D::Int;
                      init_method::AbstractString="unif_rand",
                      del::Float64=0.01,
                      discount::Float64=0.99,
                      tol::Float64=0.001)
    self = new()
    self.w = init_weights(nb_feat,init_method)
    self.d = 0
    self.D = D
    self.del = del
    self.discount = discount
    self.tol = tol
    self.done_flag = false
    self.B = eye(nb_feat)/del
    self.b = zeros(nb_feat)
    return self
  end
end
weights(p::LSPIParam) = p.w
isdone(p::LSPIParam) = p.done_flag


##TODO: make this an online algorithm that accumulates the values, and resets after
#     D interval (updates w, resets B,b every D timesteps)
function update!{T}(param::LSPIParam,
                    annealer::AnnealerParam,
                    mb::Minibatcher,
                    er::ExperienceReplayer,
                    exp::FeatureExpander,
                    phi::RealVector,
                    a::T,
                    r::Union{Float64,Int},
                    phi_::RealVector,
                    a_::T,
                    gamma::Float64,
                    lr::Float64)
  phi,a,r,phi_,a_ = replay!(er,phi,a,r,phi_,a_)

  phi = expand(exp::ActionFeatureExpander,phi,a)
  phi_ = expand(exp::ActionFeatureExpander,phi_,a_)

  #print("\rUpdating Policy...")

  param.B -= vec(param.B*phi)*(transpose(phi-gamma*phi_)*param.B)
  param.b = vec(param.b+phi*r)

  if param.d < param.D
    param.d += 1
  else
    w = vec(param.B*param.b)
    if norm(w - param.w) < param.tol
      param.done_flag = true
    end
    param.w = w
    param.d = 0
    param.B = eye(length(phi))/param.del
    param.b = zeros(length(phi))
  end

  return 0.,0.

end
