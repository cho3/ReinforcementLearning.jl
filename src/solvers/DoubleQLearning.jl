#DoubleQLearning.jl
#see something

#=
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
=#

type DoubleQParam <: UpdaterParam
  wA::RealVector
  wB::RealVector
  updatingA::Bool
  rng::AbstractRNG
  is_deterministic_switch::Bool
  feature_function::Function
end

function pad!(p::DoubleQParam,nb_new_feat::Union{Int,Array{Tuple{Int,Int},1}},nb_feat::Int)
  if length(nb_new_feat) <= 0
    return
  elseif typeof(nb_new_feat) == Int
    if nb_new_feat <= 0
      return
    end
  end
  pad!(p.wA,nb_new_feat,nb_feat)
  pad!(p.wB,nb_new_feat,nb_feat)
end

function update!{T}(param::DoubleQParam,
                    annealer::AnnealerParam,
                    mb::Minibatcher,
                    er::ExperienceReplayer,
                    exp::FeatureExpander,
                    gc::GradientClipper,
                    phi::RealVector,
                    a::T,
                    r::Union{Float64,Int},
                    phi_::RealVector,
                    a_::T,
                    gamma::Float64,
                    lr::Float64)
  phi,a,r,phi_,a_ = replay!(er,phi,a,r,phi_,a_)

  f,f_ = expand3(exp,phi,phi_)
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
    dw = clip!(gc,del*phi,1)
    param.wA += anneal!(annealer.B,minibatch!(mb.B,dw,1),lr,1)
    nb_new_feat = update!(exp,phi,del)
    pad!(param,nb_new_feat,length(f))
    return del, QA
  else
    QA_ = dot(param.wA,phi_,a_)
    QB = dot(param.wB,phi,a)
    del = r + discount*QA_ - QB
    dw = clip!(gc,del*phi,2)
    param.wB += anneal!(annealer.B,minibatch!(mb.B,dw,2),lr,2)
    nb_new_feat = update!(exp,phi,del)
    pad!(param,nb_new_feat,length(f))
    return del, QB
  end
end
