#__solvers.jl
#a spot to hold different reinforcement learning solvers for now before it gets
# broken up into individual files
abstract UpdaterParam

#TODO: reference my other implementation that actually works
type ForgetfulLSTDParam <: UpdaterParam
  alpha::Float64
  beta::Float64,
  lambda::Float64
  k::Int
  th::Array{Float64,1}
  d::Array{Float64,1}
  A::Array{Float64,2}
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
  end
end
weights(u::ForgetfulLSTDParam) = u.th

function update!{T}(param::ForgetfulLSTDParam,
                  annealer::AnnealerParam,
                  mb::Minibatcher,
                  er::ExperienceReplayer,
                  phi::Array{Union{Float64,Int},1},
                  a::T,
                  r::Union{Float64,Int},
                  phi_::Array{Union{Float64,Int},1},
                  a_::T,
                  gamma::Float64,
                  lr::Float64)

  param.e = e - param.beta*phi*dot(phi,param.e) + phi
  param.A = param.A - param.beta*phi*(phi'*param.A) + param.e*transpose(phi-gamma*phi_)
  param.d = d - param.beta*phi*dot(phi,param.d) + param.e*r
  param.e = gamma*param.lambda*e
  #TODO: figure out how to apply minibatching, per parameter learning rates, and/or experience replay
  for i = 1:param.k
    param.th += lr*(param.d-param.A*param.th)
  end
end
#####################################

type DeterministicPolicyGradientParam <: UpdaterParam

end
#######################################
#TODO: use enum
type SARSAParam <: UpdaterParam
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
weights(u::SARSAParam) = u.w

function update!{T}(param::SARSAParam,
                  annealer::AnnealerParam,
                  mb::Minibatcher,
                  er::ExperienceReplayer,
                  phi::Array{Union{Float64,Int},1},
                  a::T,
                  r::Union{Float64,Int},
                  phi_::Array{Union{Float64,Int},1},
                  a_::T,
                  gamma::Float64,
                  lr::Float64)
  phi,r,phi_ = replay!(er,phi,r,phi_)
  #NOTE: this might be a singleton array
  q = dot(param.w,phi) #TODO: dealing with feature functions that involve state and action?
  q_ = dot(param.w,phi_)
  del = r + discount*q_ - q #td error
  if param.is_replacing_trace
    param.e = max(phi,param.lambda.*param.e) #NOTE: assumes binary features
  else
    param.e  = phi + param.lambda.*param.e
  end
  dw = del*param.e
  param.w += lr.*anneal!(annealer,minibatch!(mb,dw))
end

#######################################
type QParam <: UpdaterParam

end
######################################
type DoubleQParam <: UpdaterParam

end
