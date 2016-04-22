push!(LOAD_PATH,joinpath("..","src"))
using ReinforcementLearning
using PyPlot
using Interact
import Iterators.product

typealias State Tuple{Float64,Float64}
typealias Action Float64

type InvertedPendulumModel <: Model
    g::Float64
    m::Float64
    l::Float64
    M::Float64
    alpha::Float64
    dt::Float64
    function InvertedPendulumModel(;
                                    g::Float64=9.81,
                                    m::Float64=2.,
                                    M::Float64=8.,
                                    l::Float64=0.5,
                                    dt::Float64=0.1)
        self = new()
        self.g = g
        self.m = m
        self.l = l
        self.M = M
        self.m = m
        self.alpha = 1/(m+M)
        self.dt = dt
        return self
    end
end


#init2(m::GridWorldModel,rng::AbstractRNG) = (rand(rng,1:m.W),rand(rng,1:m.H))
init(m::InvertedPendulumModel,rng::AbstractRNG) = ((rand(rng)-0.5)*0.1,(rand(rng)-0.5)*0.1)

isend(rng::AbstractRNG,m::InvertedPendulumModel,s::State,a::Action) = abs(s[1]) >= pi/2

reward(rng::AbstractRNG,m::InvertedPendulumModel,s::State,a::Action) = abs(s[1]) < pi/2 ? 0.: -1.

function dwdt(m::InvertedPendulumModel,th::Float64,w::Float64,u::Float64)
    num = m.g*sin(th)-m.alpha*m.m*m.l*(w^2)*sin(2*th)*0.5 - m.alpha*cos(th)*u
    den = (4/3)*m.l - m.alpha*m.l*(cos(th)^2)
    return num/den
end


function rk45(m::InvertedPendulumModel,s::State,a::Action)
    k1 = dwdt(m,s[1],s[2],a)
    #something...
end

function euler(m::InvertedPendulumModel,s::State,a::Action)
    alph = dwdt(m,s[1],s[2],a)
    w_ = s[2] + alph*m.dt
    th_ = s[1] + s[2]*m.dt + 0.5*alph*m.dt^2
    if th_ > pi
        th_ -= 2*pi
    elseif th_ < -pi
        th_ += 2*pi
    end
    return (th_,w_)
end

function next(rng::AbstractRNG,m::InvertedPendulumModel,s::State,a::Action)
    a_offset = 20*(rand(rng)-0.5)
    a_ = a + a_offset

    return euler(m,s,a_)
    #something..
end

nb_th_bins = 20
nb_w_bins = 20

exemplars = collect(product([-pi/4;0.;pi/4],[-1.;0;1.]))

dist(a::State,b::State) = norm([a[1]-b[1];a[2]-b[2]],2)


function generate_joint_featurefunction(m::InvertedPendulumModel)
    nb_feat = nb_th_bins*nb_w_bins
  function feature_function(s::State)
        active_indices = [ReinforcementLearning.bin(s[1],-pi,pi,nb_th_bins)+nb_th_bins*(ReinforcementLearning.bin(s[2],-1,1,nb_w_bins)-1)]
    phi = sparsevec(active_indices,ones(length(active_indices)),nb_feat)
    return phi
  end
  return feature_function
end


function generate_disjoint_featurefunction(m::InvertedPendulumModel)
    nb_feat = nb_th_bins + nb_w_bins
    function feature_function(s::State)
        active_indices = [ReinforcementLearning.bin(s[1],-pi,pi,nb_th_bins);ReinforcementLearning.bin(s[2],-1,1,nb_w_bins)]
        return sparsevec(active_indices,ones(2),nb_feat)
    end
end

function visualize(m::InvertedPendulumModel,s::State,a::Action)
    #NOTE: th = 0 is upright
    th = s[1] + pi/2.
    #base grid
    w = 1.5*m.l
    fill([-w,w,w,-w],[-w,-w,w,w],color="#FFFFFF",edgecolor="#000000")
    #draw cart
    dx = 0.05*sign(a)
    h = 0.1
    l = 0.125
    fill([-l+dx;l+dx;l+dx;-l+dx],[-h;-h;h;h],color="#FF0000")
    #draw pole
    u = m.l*cos(th) #+ dx
    v = m.l*sin(th)
    arrow(dx,0,u,v,width=m.l/10,head_width=0.,head_length=0.,color="#00FF00")
    #add cart direction (force)
    if abs(dx) > 0
        arrow(dx,0.,dx,0.,width=h,head_width=1.75*h,head_length=abs(dx),color="#0000FF")
    end
    #add pole velocity
    du = -s[2]*m.l*sin(th)/5.
    dv = s[2]*m.l*cos(th)/5.
    arrow(u,v,du,dv,width=m.l/10,head_width=m.l/5,head_length=m.l/5,color="#FF00FF")
end

function visualize(m::InvertedPendulumModel,S::Array{State,1},A::Array{Action,1})
  assert(length(S) == length(A))
  f = figure()
  @manipulate for i = 1:length(S); withfig(f) do
    visualize(m,S[i],A[i]) end
  end
end

_A = Action[-50;0;50]
A = DiscreteActionSpace(_A)

m = InvertedPendulumModel()

bbm = BlackBoxModel(m,init,next,reward,isend)
