#SARSA.jl
#see: Reinforcement Learning an introduction by Sutton & Barto

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
    self.e = init_weights(n,"zero")#spzeros(n,1)
    self.is_replacing_trace = lowercase(trace_type) == "replacing"
    self.lambda = lambda
    return self
  end
end
weights(u::SARSAParam) = u.w
function pad!(p::SARSAParam,nb_new_feat::Union{Int,Array{Tuple{Int,Int},1}},nb_feat::Int)
  if length(nb_new_feat) <= 0
    return
  elseif typeof(nb_new_feat) == Int
    if nb_new_feat <= 0
      return
    end
  end
  pad!(p.w,nb_new_feat,nb_feat)
  pad!(p.e,nb_new_feat,nb_feat)
end

function update!{T}(param::SARSAParam,
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

  f,f_ = expand3(exp,phi,phi_)
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
  dw = clip!(gc,vec(del*param.e),1)
  param.w = vec(param.w + anneal!(annealer,minibatch!(mb,dw,1),lr,1))
  param.e = gamma*param.lambda*param.e
  nb_new_feat = update!(exp,phi,del)
  pad!(param,nb_new_feat,length(f))
  return del, q
end
