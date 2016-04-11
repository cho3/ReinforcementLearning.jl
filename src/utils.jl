#utils.jl
#Mostly just to hold a function that tests if your your generative function is
#   consistent with your distribution function

function test_equals(x)
  y = deepcopy(x)
  try
    assert(x == y)
    assert(y == x)
  catch
    error("== does not work for $(typeof(x)). Make sure you import Base.== and write a function for your type")
  end
end

function test_hash(x)
  y = deepcopy(x)
  d1 = Dict{typeof(x),Int}(x=>1)
  d2 = Dict{typeof(y),Int}(y=>1)
  try
    assert(get(d1,x,0) == 1)
    assert(get(d1,y,0) == 1)
    assert(get(d2,x,0) == 1)
    assert(get(d2,y,0) == 1)
  catch
    error("hash does not work for $(typeof(x)). Make sure you import Base.hash and write a function for your type")
  end
end

function test(generator::Function,distribution::Function,input;
  nb_samples::Int=1000,
  lambda::Float64=1., #coef to use for the power divergence test
  p_reject::Float64=0.05,
  rng::AbstractRNG=MersenneTwister(83453495))

  #assume some function signature: next(rng,s,a), transition(s,a)
  s = generator(rng,input...)
  #after first sample, do some basic tests to make sure == and hash works properly
  test_equals(s)
  test_hash(s)
  #Generate samples
  counts_dict = Dict{typeof(s),Int}(s=>1)
  for i = 2:nb_samples
    s = generator(rng,input...)
    counts_dict[s] = get(counts_dict,s,0) + 1
  end

  #calculate distribution
  d = distribution(input...)
  counts = Int[]
  probs = Float64[]
  for s in support(d) #TODO: require support to be defined for this distribution type
    push!(counts,get(counts_dict,s,0))
    push!(probs,pdf(d,s)) #TODO: require pdf to be defined
  end
  #get statistics
  #NOTE: API for PowerDivergenceTest is kinda wonky--dont feel like figuring it out attempt
  #println(counts_dict)
  #println(counts)
  #println(probs)
  if length(probs) == 1
    if (sum(probs) != 1.) || (length(counts) != 1)
      error("There is an issue with something")
    end
    return
  end
  stats = ChisqTest(counts,probs)
  #stats = PowerDivergenceTest(counts,probs,lambda)

  #test statistics
  if pvalue(stats) < p_reject
    println("Error: Generator generates from the same distribution than provided with probability $(pvalue(stats)), less than threshold of $p_reject")
    println("\tPlease check your code")
    error("Inconsistent generator and distribution")
  end

end

#TODO: make some modifications for this to work for continuous state AND actions
function generate_tilecoder(nb_tiles::Int,
                            nb_intervals::Union{Int,Array{Int,1}},
                            A::DiscreteActionSpace,
                            lb::RealVector,
                            ub::RealVector)

  #TODO: some thing sensible to hold things like ub and lb?
  assert(length(ub) == length(lb))
  assert(!(false in (lb .< ub)))
  #test to make sure hashing works for each action in domain(A)
  #TODO:
  A_indices = [a=>i for (i,a) in enumerate(domain(A))]

  #precompute offsets
  if typeof(nb_intervals) == Int
    nb_intervals = ones(Int,size(lb))*nb_intervals
  end

  interval_size = (ub-lb)./nb_intervals

  C = sort(rand(nb_tiles)*nb_tiles) #placeholder
  offsets = [c*interval_size[i]/nb_tiles for i=1:length(ub), c in C]

  __nb_feat = prod(nb_intervals)*nb_tiles
  nb_feat = __nb_feat*length(A)

  interval_sizes = (ub - lb)./(nb_intervals-1) #the -1 makes it different

  tile_size = prod(nb_intervals)
  dim_offsets = [1; cumprod(nb_intervals)]
#Union{Array{Float64,1},Array{Int,1}}

  function feature_function(state::RealVector)
    active_indices = zeros(Int,nb_tiles)
    for i = 1:nb_tiles
      state_ = state + offsets[:,i]

      fs = round(Int,floor(state_-lb)./interval_sizes)
      fs = max(min(fs,nb_intervals-1),0)

      ft = 0 #TODO

      ft_ = sum([dim_offsets[i]*fs[i] for i = 1:length(state_)])
      ft = round(Int,ft_ + (i-1)*tile_size + 1)
      assert(ft,0,>)
      assert(ft,__nb_feat + 1,<)
      active_indices[i] = ft
    end
    return sparsevec(active_indices,ones(length(active_indices)),__nb_feat)
  end

  function feature_function(state::RealVector,action::eltype(domain(A)))
    active_indices = find(feature_function(state))
    active_indices += __nb_feat*(A_indices[action] - 1)

    return sparsevec(active_indices,ones(nb_tiles),nb_feat)
  end

  return feature_function
  #and then i guess you cast the state input with some function you write to map
  #state to an array
end

function bin(x::Real,lb::Real,ub::Real,nb_bins::Int)
  if x <= lb
    return 1
  elseif x >= ub
    return nb_bins
  end
  inc = (ub-lb)/nb_bins
  i = floor(Int,(x-lb)/inc) + 1
  i = max(min(i,nb_bins),1)

  return i

end

function generate_radial_basis{T}(exemplars::Array{T,1},sigma::Union{Real,Array{Real,1}},dist::Function)
  if length(sigma) == 1
    sigma = sigma*ones(length(exemplars))
  elseif length(sigma) != length(exemplars)
    error("Inconsistent exemplar and sigma vector length")
  end

  function feature_function(s::T)
    phi = ones(length(exemplars)+1)
    for i = 1:length(exemplars)
      phi[i] = exp(-dist(s,exemplars[i])./(2*sigma[i]))
    end
    return phi
  end
  return feature_function
end
#NOTE: from StatsBase
function sample(rng::AbstractRNG,wv::WeightVec)
    t = rand(rng) * sum(wv)
    w = values(wv)
    n = length(w)
    i = 1
    cw = w[1]
    while cw < t && i < n
        i += 1
        @inbounds cw += w[i]
    end
    return i
end

sample(rng::AbstractRNG,a::AbstractArray, wv::WeightVec) = a[sample(rng,wv)]
#NOTE: From rosettacode
#NOTE: powerset has size 2^N
function powerset{T}(x::Union{Vector{T},Set{T}})
  result = Vector{T}[[]] #orig code
  for elem in x,j in eachindex(result)
    push!(result,[result[j];elem]) #orig
  end
  return result
end

sortedpowerset{T}(x::Union{Vector{T},Set{T}}) = sort(powerset(x),by=length,rev=true)
