#__solvers.jl
#a spot to hold different reinforcement learning solvers for now before it gets
# broken up into individual files

#TODO: reference my other implementation that actually works
type ForgetfulLSTDParam <: UpdaterParam
  alpha::Float64
  beta::Float64
  lambda::Float64
  k::Int
  th::RealVector
  d::RealVector
  A::RealMatrix
  #TODO: constructor
  function ForgetfulLSTDParam(nb_feat::Int;
                              alpha::Float64=0.01/sqrt(n),
                              lambda::Float64=0.95,
                              k::Int=1,
                              init_method::AbstractString="")
    #This is actual ForgetfulLSTD
    self = new()
    self.alpha = alpha
    self.beta = alpha
    self.lambda = lambda
    self.th = zeros(n) #TODO: init_method...
    self.d = deepcopy(self.th)./alpha
    self.A = eye(n)/alpha

    return self
  end
end
weights(u::ForgetfulLSTDParam) = u.th

function update!{T}(param::ForgetfulLSTDParam,
                  annealer::AnnealerParam,
                  mb::Minibatcher,
                  er::ExperienceReplayer,
                  phi::RealVector,
                  a::T,
                  r::Real,
                  phi_::RealVector,
                  a_::T,
                  gamma::Float64,
                  lr::Float64)

  param.e += vec(-param.beta*phi*dot(phi,param.e) + phi)
  param.A += -(param.beta*phi)*(phi'*param.A) + param.e*transpose(phi-gamma*phi_)
  param.d += vec(-param.beta*phi*dot(phi,param.d) + param.e*r)
  param.e = vec(gamma*param.lambda*param.e)
  #TODO: figure out how to apply minibatching, per parameter learning rates, and/or experience replay
  for i = 1:param.k
    param.th += lr*(param.d-param.A*param.th)
  end
  return 0., 0. #not sure what the estim is off the top of my head
end
#####################################

type DeterministicPolicyGradientParam <: UpdaterParam

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

function update!{T}(param::SARSAParam,
                  annealer::AnnealerParam,
                  mb::Minibatcher,
                  er::ExperienceReplayer,
                  phi::RealVector,
                  a::T,
                  r::Real,
                  phi_::RealVector,
                  a_::T,
                  gamma::Float64,
                  lr::Float64)
  phi,a,r,phi_,a_ = replay!(er,phi,a,r,phi_,a_)
  #NOTE: this might be a singleton array
  q = dot(param.w,phi) #TODO: dealing with feature functions that involve state and action?
  q_ = dot(param.w,phi_)
  del = r + gamma*q_ - q #td error
  if param.is_replacing_trace
    param.e = vec(max(phi,param.e)) #NOTE: assumes binary features
  else
    param.e  = vec(phi + param.e)
  end
  dw = vec(del*param.e)
  param.w += lr.*anneal!(annealer,minibatch!(mb,dw))
  param.e *= gamma*param.lambda
  return del, q
end

#######################################
type QParam <: UpdaterParam

end


######################################
type GQParam <: UpdaterParam

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
  wA::Array{Float64,1}
  wB::Array{Float64,1}
  updatingA::Bool
  rng::AbstractRNG
  is_deterministic_switch::Bool
end

function update!{T}(param::DoubleQParam,
                    annealer::DoubleAnnealer,
                    mb::DoubleMinibatcher,
                    er::ExperienceReplayer,
                    phi::RealVector,
                    a::T,
                    r::Union{Float64,Int},
                    phi_::RealVector,
                    a_::T,
                    gamma::Float64,
                    lr::Float64)
  phi,a,r,phi_,a_ = replay!(er,phi,a,r,phi_,a_)

  updateAflag = false
  if param.is_deterministic_switch
    param.updatingA = !param.updatingA
    updateAflag = param.updatingA
  else
    updateAflag = rand(param.rng,Bool)
  end
  if updateAflag
    QB_ = dot(param.wB,phi_)
    QA = dot(param.wA,phi)
    del = r + discount*QB_ - QA
    dw = del*phi
    param.wA += lr*anneal!(annealer.B,minibatch!(mb.B,dw))
    return del, QA
  else
    QA_ = dot(param.wA,phi_)
    QB = dot(param.wB,phi)
    del = r + discount*QA_ - QB
    dw = del*phi
    param.wB += lr*anneal!(annealer.B,minibatch!(mb.B,dw))
    return del, QB
  end
end
