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
                    gc::GradientClipper,
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
