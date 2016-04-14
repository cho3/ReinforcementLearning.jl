#GQLearning.jl
#See paper by Maei, Sutton 2010

type GQParam <: UpdaterParam
  w::RealVector
  th::RealVector
  e::RealVector
  b::Float64 #learning rate for gradient corrector
  lambda::Float64
  #rho::Float64
  A::DiscreteActionSpace
  #feature_function::Function
  is_replacing_trace::Bool
  function GQParam(n::Int,A::DiscreteActionSpace;
                  lambda::Float64=0.95,
                  b::Float64=1e-4,
                  init_method::AbstractString="unif_rand",
                  trace_type::AbstractString="replacing")
    self = new()
    self.w = init_weights(n,"zero")#spzeros(n,1)#init_weights(n,"zero")
    self.th = init_weights(n,init_method)
    self.b = b
    self.e = init_weights(n,"zero")# spzeros(n,1)
    self.lambda = lambda
    self.is_replacing_trace = lowercase(trace_type) == "replacing"
    self.A = A
    #self.feature_function = ff

    return self
  end
end
weights(p::GQParam) = p.th
function pad!(p::GQParam,nb_new_feat::Union{Int,Array{Tuple{Int,Int},1}},nb_feat::Int)
  if length(nb_new_feat) <= 0
    return
  elseif typeof(nb_new_feat) == Int
    if nb_new_feat <= 0
      return
    end
  end
  pad!(p.w,nb_new_feat,nb_feat)
  pad!(p.th,nb_new_feat,nb_feat)
  pad!(p.e,nb_new_feat,nb_feat,p.lambda) #since if this was a new feature, then the trace was active
end

#NOTE: single step GQ for now (no eligbility traces)
function update!{T}(param::GQParam,
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
  f, f_ = expand3(exp,phi,phi_)
  #f_ = expand(exp,phi_)
  #expand with respect to actions
  #phi = expand(exp::ActionFeatureExpander,f,a)
  phi = expand(exp,f,a)
  #phi_ = expand(exp::ActionFeatureExpander,f_,a_)
  #NOTE: this might be a singleton array
  #println(size(phi))
  #println(size(param.th))
  q = dot(param.th,vec(phi)) #TODO: dealing with feature functions that involve state and action?
  #extracting out the best next action--semi-redundant step
  phi_s = [expand(exp::ActionFeatureExpander,f_,_a) for _a in domain(param.A)]
  qs_ = [dot(param.th,vec(v)) for v in phi_s]
  q_ind_ = indmax(qs_)
  q_ = qs_[q_ind_]
  phi_ = phi_s[q_ind_]
  # end step
  if param.is_replacing_trace
    param.e = vec(max(phi,param.e)) #NOTE: assumes binary features
  else
    param.e = vec(phi + param.e)
  end
  del = r + gamma*q_ - q #td error
  #dth = vec(del*phi-gamma*dot(param.w,phi)*phi_)
  dth = clip!(gc,vec(del*param.e-gamma*(1-param.lambda)*dot(param.w,param.e)*phi_),1)
  param.th = vec(param.th + anneal!(annealer,minibatch!(mb,dth,1),lr,1))
  dw = clip!(gc,(del*param.e-dot(param.w,phi)*phi),2)
  param.w = vec(param.w + anneal!(annealer,minibatch!(mb,dw,2),lr*param.b,2))
  param.e = gamma*param.lambda*param.e
  nb_new_feat = update!(exp,f,del)
  pad!(param,nb_new_feat,length(f))
  return del, q
end
