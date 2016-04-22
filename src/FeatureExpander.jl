
#iFDD

init!(exp::FeatureExpander) = 0
expand{T}(exp::FeatureExpander,phi::RealVector,a::T) = phi
expand(exp::FeatureExpander,phi::RealVector) = phi
update!(::FeatureExpander,phi::RealVector,del::Float64) = 0
expand3(exp::FeatureExpander,phi::RealVector,phi_::RealVector) = expand(exp,phi),expand(exp,phi_)
length(::FeatureExpander) = 0

type TrueNullFeatureExpander <: FeatureExpander
end

type NullFeatureExpander{T} <: ActionFeatureExpander
  A_indices::Dict{T,Int}
  #=
  function NullFeatureExpander{T}(A::DiscreteActionSpace{T})
    self = new()
    self.A_indices = Dict{T,Int}([a=>i for (i,a) in enumerate(domain(A))])
    return self
  end
  =#
end
NullFeatureExpander{T}(A::DiscreteActionSpace{T}) = NullFeatureExpander(Dict{T,Int}([a=>i for (i,a) in enumerate(domain(A))]))

#NOTE: this one assumes binary sparse features
function expand{T}(exp::ActionFeatureExpander,phi::Union{SparseMatrixCSC{Float64,Int},SparseMatrixCSC{Int,Int}},a::T)
  active_indices = find(phi)
  nb_feat = length(phi)
  active_indices += nb_feat*(exp.A_indices[a]-1)
  return sparsevec(active_indices,ones(length(active_indices)),nb_feat*length(exp.A_indices))
end

#NOTE: this one assumes dense features which might not be binary
function expand{T}(exp::ActionFeatureExpander,phi::Union{Array{Float64,1},Array{Int,1}},a::T)
  _phi = zeros(length(phi)*length(exp.A_indices))
  _phi[1+length(phi)*(exp.A_indices[a]-1):length(phi)*exp.A_indices[a]] = phi
  return _phi
end
expand2{T}(exp::ActionFeatureExpander,phi::RealVector,a::T) = expand(exp,expand(exp,phi),a)

update!(::NullFeatureExpander,phi::RealVector,del::Float64) = 0

shift!(::FeatureExpander,shift_idx::Int...) = 0

##########################################################################
type iFDDExpander{T} <: ActionFeatureExpander
  A_indices::Dict{T,Int}
  learned_features::Array{Tuple{Int,Int},1}
  learned_feature_set::Set{Tuple{Int,Int}}
  err_dict::Dict{Tuple{Int,Int},Float64}
  app_dict::Dict{Tuple{Int,Int},Int}
  xi::Float64 #[0.1,0.2,0.5] cutoff criterion
  #shift::Array{Int,1}
  #div::Array{Int,1}
  #other stuff
end
length(exp::iFDDExpander) = length(exp.learned_features)
#iFDDExpander(a,b,c,d,e,f) = iFDDExpander(a,b,c,d,e,f,Int[],Int[])
#iFDDExpander(a,b,c,d,e,f,div) = iFDDExpander(a,b,c,d,e,f,div,zeros(Int,length(div)))

#this is the third bin function....
#=
function bin(idx::Int,div::Array{Int,1})
  """
  div = [first divider idx, second divider idx...]
  idx = some idx--does it fall into the first bin, second bin etc...
  """
  val = findfirst((div-idx) .> 0)
  return val
end
=#

#shift(exp::iFDDExpander,idx::Int) = bin(idx,exp.div) == 0 ? 0 : exp.shift[bin(idx,exp.div)]

#=
function shift!(exp::iFDDExpander,shift_idx::Int...)
  for (i,idx) in enumerate(shift_idx)
    exp.shift[i] += idx
    exp.div[i] += idx
    #findfirst(exp.div .> i)
    #if _ == 0: 0
  end
end
=#

function iFDDExpander{T}(A::DiscreteActionSpace{T};xi::Float64=0.5)
  #self = new()
  A_indices = Dict{T,Int}([a=>i for (i,a) in enumerate(domain(A))])
  learned_features = Tuple{Int,Int}[]
  learned_feature_set = Set{Tuple{Int,Int}}()
  err_dict = Dict{Tuple{Int,Int},Float64}()
  app_dict = Dict{Tuple{Int,Int},Int}()
  return iFDDExpander(A_indices,learned_features,learned_feature_set,err_dict,app_dict,xi)
end
function iFDDExpander(A::ContinuousActionSpace;xi::Float64=0.5)
  #self = new()
  A_indices = Dict{Float64,Int}()
  learned_features = Tuple{Int,Int}[]
  learned_feature_set = Set{Tuple{Int,Int}}()
  err_dict = Dict{Tuple{Int,Int},Float64}()
  app_dict = Dict{Tuple{Int,Int},Int}()
  return iFDDExpander(A_indices,learned_features,learned_feature_set,err_dict,app_dict,xi)
end


function expand(expander::iFDDExpander,phi::RealVector)
  _phi = spzeros(length(expander.learned_features),1)
  active_indices = Set(find(phi))
  offset = length(phi) #+ sum(expander.shift) #???
  phi = vcat(phi,_phi)
  #NOTE: order matters for learned features!
  for (k,(i,j)) in enumerate(expander.learned_features)
    #i += shift(expander,i) #???
    #j += shift(expander,j) #???
    if (i in active_indices) && (j in active_indices)
      #(phi_copy[i] > 0) && (phi_copy[j] > 0)
      #NOTE: i think the issue is here--how to do sorted power set?
      phi[offset+k] = 1.
      push!(active_indices,offset+k)
      delete!(active_indices,i)
      delete!(active_indices,j)
      #remove i and j from active indices?
      phi[i] = 0
      phi[j] = 0
      #i think tihs works?
    end
  end
  return phi
end

#TODO: figure out a way to make sure the state representation doesnt explode
#NOTE: what needs to happen is i deactivate features that are subfeatures to an existing feature
function update!(expander::iFDDExpander,phi::RealVector,del::Float64)
  active_indices = find(phi)
  nb_new_feat = 0
  for i in active_indices
    for j in active_indices
      #enforce ordering on (i,j) pairs
      #i -= shift(expander,i)
      #j -= shift(expander,j)
      #TODO: maintain set of learned pairs for collision detection
      if (i >= j) || ((i,j,) in expander.learned_feature_set)
        continue
      end
      err = get(expander.err_dict,(i,j),0.) + del
      expander.err_dict[(i,j)] = err
      count = get(expander.app_dict,(i,j),0) + 1
      expander.app_dict[(i,j)] = count
      if abs(err)/sqrt(count) > expander.xi
        push!(expander.learned_features,(i,j))
        push!(expander.learned_feature_set,(i,j))
        #delete from err_dict, app_dict?
        delete!(expander.err_dict,(i,j))
        delete!(expander.app_dict,(i,j))
        nb_new_feat += 1
      end
    end #j
  end #i
  return nb_new_feat
end
#######################################################################
type iFDDProperExpander{T} <: ActionFeatureExpander
  A_indices::Dict{T,Int}
  learned_features::Dict{Set{Int},Int}
  index_feature_map::Dict{Int,Set{Int}}
  err_dict::Dict{Set{Int},Float64}
  app_dict::Dict{Set{Int},Int}
  xi::Float64 #[0.1,0.2,0.5] cutoff criterion
  #other stuff

end
length(exp::iFDDProperExpander) = length(exp.learned_features)

function shift!(exp::iFDDProperExpander,shift_idx::Int...)
  # gotta shift everything fml qq
end

function iFDDProperExpander{T}(A::DiscreteActionSpace{T};xi::Float64=0.5)
  #self = new()
  A_indices = Dict{T,Int}([a=>i for (i,a) in enumerate(domain(A))])
  learned_features = Dict{Set{Int},Int}()
  index_feature_map = Dict{Int,Set{Int}}()
  err_dict = Dict{Set{Int},Float64}()
  app_dict = Dict{Set{Int},Int}()
  return iFDDProperExpander(A_indices,learned_features,
                      index_feature_map,err_dict,app_dict,xi)
end

to_features(exp::iFDDProperExpander,i::Int) =
            i in keys(exp.index_feature_map) ? exp.index_feature_map[i]: Set([i])

#NOTE probably issue with old features not being zerod out? alternatively,
# could be issue with how new features are eing indexed
function expand(expander::iFDDProperExpander,phi::RealVector)
  _phi = spzeros(length(expander.learned_features),1)
  #NOTE: issue is with this
  active_indices = Set(find(phi))
  feat_set = sortedpowerset(active_indices)
  #=
  if length(expander.learned_features) > 0
    println(active_indices)
    println(feat_set)
  end
  =#
  offset = length(phi)
  phi = vcat(phi,_phi)
  #println(length(phi))
  #phi = spzeros(length(expander.learned_features)+offset,1)
  #NOTE: order matters for learned features!
  for _f in feat_set
    f = Set(_f)
    if (f in keys(expander.learned_features)) && (length(setdiff(f,active_indices)) == 0)
      active_indices = setdiff(active_indices,f)
      #phi[expander.learned_features[f]+offset] = 1.
      phi[expander.learned_features[f]] = 1.
      for idx in f
        assert(idx <= offset)
        phi[idx] = 0
      end
    end
  end

  return phi
end

#TODO: figure out a way to make sure the state representation doesnt explode
#NOTE: what needs to happen is i deactivate features that are subfeatures to an existing feature
function update!(expander::iFDDProperExpander,phi::RealVector,del::Float64)
  #phi = expand(expander,phi) #inefficient T.T
  active_indices = find(phi)
  #println(active_indices)
  nb_new_feat = 0
  for i in active_indices
    for j in active_indices
      #enforce ordering on (i,j) pairs
      if i >= j
        continue
      end
      f = union(to_features(expander,i),to_features(expander,j))
      #=
      if length(f) > 2
        print(to_features(expander,i))
        print(to_features(expander,j))
        println(f)
      end
      =#
      #println(f)
      #TODO: maintain set of learned pairs for collision detection
      if f in keys(expander.learned_features)
        continue
      end
      err = get(expander.err_dict,f,0.) + del
      expander.err_dict[f] = err
      count = get(expander.app_dict,f,0) + 1
      expander.app_dict[f] = count
      if abs(err)/sqrt(count) > expander.xi
        #ind = length(expander.learned_features) + 1
        ind = length(phi) + nb_new_feat + 1
        expander.learned_features[f] = ind
        expander.index_feature_map[ind] = f
        #delete from err_dict, app_dict?
        delete!(expander.err_dict,f)
        delete!(expander.app_dict,f)
        nb_new_feat += 1
      end
    end #j
  end #i
  return nb_new_feat
end

#pad!(x::SparseMatrixCSC,nb_new_feat::Int) = x.m += nb_new_feat
#pad!{T}(x::Array{T,1},nb_new_feat::Int) = append!(x,zeros(T,nb_new_feat))

function pad!(x::SparseMatrixCSC,nb_new_feat::Int,interval_length::Int,val::Float64=0.)
  if length(x) == 0 && nb_new_feat == 0
    return x
  end
  if interval_length == 0
    return zeros(nb_new_feat)
  end
  assert(mod(length(x),interval_length)== 0)
  intervals = length(x)/interval_length
  x.m += nb_new_feat*intervals
  #go through each element, greater than interval, incrememnt accordingly
  #can probably do in one pass\
  for (i,ind) in enumerate(x.rowval)
    x.rowval[i] += floor(Integer,ind/interval_length)*nb_new_feat
  end
  if val != 0.
    new_inds = vec([i+convert(Int,j*interval_length+(j-1)*nb_new_feat) for i=1:nb_new_feat, j=1:intervals])
    x[new_inds] = val
  end
  return x
end

#TODO I have no idea how to do this
function pad!(x::SparseMatrixCSC,new_feat::Array{Tuple{Int,Int},1},interval_length::Int,val::Float64=0.)
  if length(x) == 0 && length(new_feat) == 0
    return x
  end
  if interval_length == 0
    return spzeros(sum([j for (i,j) in new_feat]))
  end
  assert(mod(length(x),interval_length)== 0)
  intervals = length(x)/interval_length
  #TODO Fix
  x.m += nb_new_feat*intervals
  #go through each element, greater than interval, incrememnt accordingly
  #can probably do in one pass\
  for (loc, nb_new_feat) in new_feat
    for (i,ind) in enumerate(x.rowval) #index of nonzero elements
      x.rowval[i] += floor(Integer,ind/interval_length)*nb_new_feat
      #the above pushes everything beyond the frame

    end
  end
  #=
  if val != 0.
    new_inds = vec([i+convert(Int,j*interval_length+(j-1)*nb_new_feat) for i=1:nb_new_feat, j=1:intervals])
    x[new_inds] = val
  end
  =#
  return x
end

#TODO pad! for Array{T,2}
#TODO Switch to tuple representation since order matters
function pad!{T}(x::Array{T,1},nb_new_feat::Int,interval_length::Int,val::T=zero(T))
  if length(x) == 0 && nb_new_feat == 0
    return x
  end
  if interval_length == 0
    return zeros(nb_new_feat)
  end
  assert(mod(length(x),interval_length)== 0)
  intervals = length(x)/interval_length
  #star from the bottom
  for i = intervals:-1:1
    for j = 1:nb_new_feat
      insert!(x,convert(Int,i*interval_length+1),val)
    end
  end
  return x
end

function pad!{T}(x::Array{T,1},new_feat::Array{Tuple{Int,Int},1},interval_length::Int,val::T=zero(T))
  if length(x) == 0 && length(new_feat) == 0
    return x
  end
  if interval_length == 0
    return zeros(sum([j for (i,j) in new_feat]))
  end
  assert(mod(length(x),interval_length)== 0)
  intervals = length(x)/interval_length
  #star from the bottom
  for i = intervals:-1:1
    for (loc,nb_new_feat) in reverse(new_feat)
      for j = 1:nb_new_feat
        insert!(x,convert(Int,(i-1)*interval_length+loc+1),val)
      end
    end
  end
  return x
end

function pad!{T}(x::Array{T,2},new_feat::Array{Tuple{Int,Int},1},interval_length::Int,val::T=zero(T))
  if length(x) == 0 && length(new_feat) == 0
    return x
  end
  if interval_length == 0
    return zeros(sum([j for (i,j) in new_feat]))
  end
  assert(mod(size(x,1),interval_length)== 0)
  intervals = size(x,1)/interval_length
  len = size(x,2)
  #star from the bottom
  for i = intervals:-1:1
    for (loc,nb_new_feat) in reverse(new_feat)
      x = [x[1:(i-1)*interval_length+loc,:];ones(nb_new_feat,len)*val;x[(i-1)*interval_length+loc+1:end,:]]
    end
  end
  return x
end

####################################################################
#WORDS

type TimeCatExpander <: ActionFeatureExpander
  #k::Int #nb of time slices to queue
  memory::Tuple{RealVector,RealVector} #f_{t-1}, PHI_{t-1}
  last_vec::Tuple{RealVector,RealVector,RealVector} #to make it work with updating: f_{t}, f_{t-1},PHI_{t-1}
  last_vec_::Tuple{RealVector,RealVector,RealVector}
  inframe_exp::FeatureExpander #in-frame expander (state space representation)
  crossframe_exp::FeatureExpander #cross frame expander (histories)
  action_exp::FeatureExpander #action feature expander?
end
TimeCatExpander(n,exp1,exp2,exp3) = TimeCatExpander((spzeros(n,1),spzeros(0,1),),
                                                (spzeros(n,1),spzeros(n,1),spzeros(0,1),),
                                                (spzeros(n,1),spzeros(n,1),spzeros(0,1),),
                                                exp1,exp2,exp3)

function expand3(exp::TimeCatExpander,phi::RealVector,phi_::RealVector)
  #f_ = expand(exp.inframe_exp,phi_)
  #f, PHI = exp.memory
  #f_ = expand(exp.crossframe_exp,f_,f,PHI)
  f_ = expand(exp.crossframe_exp,exp.last_vec_...) #TODO don't use expand fml
  f = expand(exp.crossframe_exp,exp.last_vec...) #this was calculated when it was expanded in the policy
  return f, f_
end

function expand(exp::TimeCatExpander,phi::RealVector)
  f_ = expand(exp.inframe_exp,phi)
  #PHI = vcat(f,exp.memory)
  f, PHI = exp.memory
  phi_ = expand(exp.crossframe_exp,f_,f,PHI)
  PHI_ = phi_[length(PHI)+length(f)+length(f_)+1:end] #assumes expand(crossframe_exp,phi) adds to the end
  exp.memory = (f_,PHI_)
  exp.last_vec = exp.last_vec_
  exp.last_vec_ = (f_,f,PHI)
  assert(countnz(phi_) > 0)
  return phi_
end

expand{T}(exp::TimeCatExpander,phi::RealVector,a::T) = expand(exp.action_exp,phi,a)


##this is why we're making init!
function init!(exp::TimeCatExpander)
  #fill memory with zeros
  #TODO fix
  mem = exp.memory
  last = exp.last_vec
  exp.memory = (spzeros(size(mem[1])...),spzeros(size(mem[2])...))
  exp.last_vec = (spzeros(size(last[1])...),spzeros(size(last[1])...),spzeros(size(last[3])...),)
  exp.last_vec_ = (spzeros(size(last[1])...),spzeros(size(last[1])...),spzeros(size(last[3])...),)
end

function update!(exp::TimeCatExpander,phi::RealVector,del::Float64)
  #TODO how to partition out the individual expanders
  #TODO how to keep track of each expander's indices fml
  """
  In frame expansion should be fine
  --just need to remember to pad the memory and last vector if need be
  The issue is with the cross frame expander:
    the cross frame expanded vector is:
    [phi_, phi, PHI, PHI_]
  when in-frame gets a new feature, then cross frame indices increment by 2
    one for phi_, one for phi
  when cross-frame gets a new feature, indices increase by 1?
    for PHI getting one bigger

  Probably what needs to happen is change the signature of update! in solvers
    so operate on original feature vectors--inefficient but more flexible
  """
  #f = expand(exp.inframe_exp,phi)
  #PHI = exp.last_vec
  f_,f,PHI = exp.last_vec
  PHI_ = expand(exp.crossframe_exp,f_,f,PHI)
  PHI_ = PHI_[1+length(f_)+length(f)+length(PHI):end]
  new_feat1 = update!(exp.inframe_exp,f_,del)
  new_feat2 = update!(exp.crossframe_exp,del,f_,f,PHI,PHI_)

  ret = [(length(f_),new_feat1),(length(f_)*2,new_feat1),
                    (length(f_)+length(f)+length(PHI),new_feat2),
                    (length(f_)+length(f)+length(PHI)+length(PHI_),new_feat2)]
  #println(ret)
  g_,g,GAM = exp.last_vec_
  #Pad memory!
  exp.last_vec = (pad!(f_,new_feat1,length(f_)),pad!(f,new_feat1,length(f)),pad!(PHI,new_feat2,length(PHI)),)
  #println("last_vec_")
  #println(exp.last_vec_)
  exp.last_vec_ = (pad!(g_,new_feat1,length(g_)),pad!(g,new_feat1,length(g)),pad!(GAM,new_feat2,length(GAM)),)
  #println(exp.last_vec_)
  exp.memory = (pad!(exp.memory[1],new_feat1,length(exp.memory[1])),pad!(exp.memory[2],new_feat2,length(exp.memory[2])),)
  return ret
  #TODO: dictionary representation--where to insert other than end

  #TODO shift frame
  #TODO how to correctly pad the things

end


 ################################################################
 type iFDDCrossExpander{T} <: ActionFeatureExpander
   A_indices::Dict{T,Int}
   learned_features::Array{Tuple{Tuple{Int,Int},Tuple{Int,Int}},1}
   learned_feature_set::Set{Tuple{Tuple{Int,Int},Tuple{Int,Int}}}
   err_dict::Dict{Tuple{Tuple{Int,Int},Tuple{Int,Int}},Float64}
   app_dict::Dict{Tuple{Tuple{Int,Int},Tuple{Int,Int}},Int}
   xi::Float64 #[0.1,0.2,0.5] cutoff criterion
   #shift::Array{Int,1}
   #div::Array{Int,1}
   #other stuff
 end
length(exp::iFDDCrossExpander) = length(exp.learned_features)

function iFDDCrossExpander{T}(A::DiscreteActionSpace{T};xi::Float64=0.5)
  #self = new()
  A_indices = Dict{T,Int}([a=>i for (i,a) in enumerate(domain(A))])
  learned_features = Tuple{Tuple{Int,Int},Tuple{Int,Int}}[]
  learned_feature_set = Set{Tuple{Tuple{Int,Int},Tuple{Int,Int}}}()
  err_dict = Dict{Tuple{Tuple{Int,Int},Tuple{Int,Int}},Float64}()
  app_dict = Dict{Tuple{Tuple{Int,Int},Tuple{Int,Int}},Int}()
  return iFDDCrossExpander(A_indices,learned_features,learned_feature_set,err_dict,app_dict,xi)
end
function iFDDCrossExpander(A::ContinuousActionSpace;xi::Float64=0.5)
  #self = new()
  A_indices = Dict{Float64,Int}()
  learned_features = Tuple{Tuple{Int,Int},Tuple{Int,Int}}[]
  learned_feature_set = Set{Tuple{Tuple{Int,Int},Tuple{Int,Int}}}()
  err_dict = Dict{Tuple{Tuple{Int,Int},Tuple{Int,Int}},Float64}()
  app_dict = Dict{Tuple{Tuple{Int,Int},Tuple{Int,Int}},Int}()
  return iFDDCrossExpander(A_indices,learned_features,learned_feature_set,err_dict,app_dict,xi)
end

#NOTE probably modifyign copies of stuff here
function expand(expander::iFDDCrossExpander,phi::RealVector...)
  _phi = spzeros(length(expander.learned_features),1)
  active_indices = [Set(find(f)) for f in phi]
  push!(active_indices,Set{Int}()) #this one is for PHI_
  #NOTE PHI_ not in phi...-->must add to end(somehow)
  #new_active_indices = Set{Int}()
  phi = RealVector[deepcopy(f) for f in phi]
  push!(phi,_phi)
  #TODO add _phi to phi
  offset = 0#length(phi) + sum(expander.shift) #???
  #phi = vcat(phi,_phi)
  #NOTE: order matters for learned features!
  for (k,((i,idx),(j,jdx))) in enumerate(expander.learned_features)
    #i += shift(expander,i) #???
    #j += shift(expander,j) #???
    if (idx in active_indices[i]) && (jdx in active_indices[j])
      #(phi_copy[i] > 0) && (phi_copy[j] > 0)
      #NOTE: i think the issue is here--how to do sorted power set?
      _phi[offset+k] = 1.
      push!(active_indices[4],offset+k)
      delete!(active_indices[i],idx)
      delete!(active_indices[j],jdx)
      #remove i and j from active indices?
      phi[i][idx] = 0
      phi[j][jdx] = 0
      #i think tihs works?
    end
  end

  f = phi[1]
  for _f in phi[2:end]
    f = vcat(f,_f)
  end
  #println(f)
  #phi = vcat(f,_phi)

  assert(countnz(phi) > 0)
  return vec(f)
end

#TODO: figure out a way to make sure the state representation doesnt explode
#NOTE: what needs to happen is i deactivate features that are subfeatures to an existing feature
function update!(expander::iFDDCrossExpander,del::Float64,phi::RealVector...)
  _active_indices = [find(f) for f in phi]
  #TODO get active_indices into form: [](i,idx)]
  #println("eyy")
  # idx'th index of the i'th feature vector
  active_indices = [(0,0) for _ in 1:sum([length(f) for f in _active_indices])]
  idz = 1
  for (i,f) in enumerate(phi)
    for ind in _active_indices[i]
      active_indices[idz] = (i,ind)
      idz += 1
    end
  end
  nb_new_feat = 0
  for (k,(i,idx)) in enumerate(active_indices)
    for (l,(j,jdx)) in enumerate(active_indices)
      #enforce ordering on (i,j) pairs
      #i -= shift(expander,i)
      #j -= shift(expander,j)
      #TODO: maintain set of learned pairs for collision detection
      if (k >= l) || (((i,idx),(j,jdx),) in expander.learned_feature_set)
        continue
      end
      err = get(expander.err_dict,((i,idx),(j,jdx),),0.) + del
      expander.err_dict[((i,idx),(j,jdx),)] = err
      count = get(expander.app_dict,((i,idx),(j,jdx),),0) + 1
      expander.app_dict[((i,idx),(j,jdx),)] = count
      if abs(err)/sqrt(count) > expander.xi
        push!(expander.learned_features,((i,idx),(j,jdx),))
        push!(expander.learned_feature_set,((i,idx),(j,jdx),))
        #delete from err_dict, app_dict?
        delete!(expander.err_dict,((i,idx),(j,jdx),))
        delete!(expander.app_dict,((i,idx),(j,jdx),))
        nb_new_feat += 1
      end
    end #j
  end #i
  return nb_new_feat
end

function label(expander::iFDDCrossExpander,labels::Array{AbstractString,1}=AbstractString[])
  #labels: name for each feature index
  labels = Dict{Int,AbstractString}([i=>s for (i,s) in enumerate(labels)])
  names = ["" for _ = 1:length(expander.learned_features)]
  names_fn = [()->"" for _ = 1:length(expander.learned_features)]
  for (k,((i,idx),(j,jdx))) in enumerate(expander.learned_features)
    str = ""
    """
    1: label w/ t
    2: label w/ t-1
    3: names[idx] w/ t-1
    4: names[idx] w/ t
    """
    if i < 3
      s1 = get(labels,idx,str(idx))
    end
    if j < 3
      s2 = get(labels,jdx,str(jdx))
    end

    if i in [1;4]
      if j in [1;4]
        names_fn[k] = (t)->string(s1,t,s2,t)
      else
        names_fn[k] = (t)->string(s1,t,s2,t-1) #t-1
      end
    else
      if j in [1;4]
        names_fn[k] = (t)->string(s1,t-1,s2,t)
      else
        names_fn[k] = (t)->string(s1,t-1,s2,t-1) #t-1
      end
    end

    s = ["",""]
    t = [0;0]
    f = [()->"",()->""]
    for (y,(x,xdx)) in enumerate(zip([i,j],[idx,jdk]))
      if x == 1
        f[y] = (t)->string(get(labels,xdx,str(xdx)),t)

      elseif x == 2
        f[y] = (t)->string(get(labels,xdx,str(xdx)),t-1)
      elseif x == 3
        f[y] = (t)->names_fn[xdx](t-1)
      else
        f[y] = (t)->names_fn[xdx](t)
      end
      s[y] = ""
      t[y] = 0
    end
    str_fn[k] = (t)->string(f[1](t),f[2](t))

    names[k] = str_fn[k](0)
  end

  return names

end
