#ForgetfulLSTD.jl
#see paper: a deeper look at planning as learning from replay

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
                  gc::GradientClipper,
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
