#utils.jl
#Mostly just to hold a function that tests if your your generative function is
#   consistent with your distribution function

using HypothesisTests

function test_equals(x)
  y = deepcopy(x)
  try
    assert(x == y)
    assert(y == x)
  catch
    error("== does not work for $(typeof(x)). Make sure you import Base.== and write a function for your type")
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
    stats = PowerDivergenceTest(counts,probs,lambda)

    #test statistics
    if pvalue(stats) < p_reject
      println("Error: Generator generates from a different distribution than provided with probability $(pvalue(stats)), less than threshold of $p_reject")
      println("\tPlease check your code")
      error("Inconsistent generator and distribution")
    end

  end
