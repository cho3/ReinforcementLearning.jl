type NullMinibatcher <: Minibatcher end
Minibatcher() = NullMinibatcher()
minibatch!(mb::NullMinibatcher,dw::Union{RealVector,RealMatrix},idx::Int) = dw

"This is uniform because you could presumably do other weird averaging things"
type UniformMinibatcher <: Minibatcher
  minibatch_size::Dict{Int,Int}
  dw::Dict{Int,Union{RealVector,RealMatrix}}
  current_minibatch_size::Dict{Int,Int}
end
function minibatch!(mb::UniformMinibatcher,dw::Union{RealVector,RealMatrix},idx::Int)
  if mb.current_minibatch_size[idx] < mb.minibatch_size[idx]
    mb.dw[idx] += dw
    mb.current_minibatch_size[idx] += 1
    return zeros(size(dw))
  else
    dw_ = (mb.dw[idx] + dw)./mb.minibatch_size[idx]
    mb.current_minibatch_size[idx][idx] = 0
    mb.dw[idx] = zeros(size(dw))
    return dw_
  end
end
