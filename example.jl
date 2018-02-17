using Boltzmann
using MNIST

include("reporter.jl")

# we need to reshape weights/samples for visualization purposes
function reshape_mnist(samples; c=10, r=10, h=28, w=28)
  f = zeros(r*h,c*w)
  for i=1:r, j=1:c
    f[(i-1)*h+1:i*h,(j-1)*w+1:j*w] = reshape(samples[:,(i-1)*c+j],h,w)
  end
  w_min = minimum(samples)
  w_max = maximum(samples)
  scale = x -> (x-w_min)/(w_max-w_min)
  map!(scale,f,f)
  colorview(Gray,f)
end

# hyperparameters
nh = 100
sigma = 0.001
n_epochs = 200
lr = 1e-5
batch_size = 100
randomize = true

X, _ = traindata()
X = 2(X  ./ (maximum(X) - minimum(X))) - 1

# using +- 1 binary units (Ising spins)
rbm = IsingRBM(28*28, nh; sigma=sigma, X=X)

# declaring wanted plots
weights = Dict(
  :ys => [:W],
  :transforms => [x->x[:]],
  :title => "Weights",
  :seriestype => :histogram,
  :leg => false,
  :nbins => 100
)

PL = Dict(
  :ys => [(:rbm, :X)],
  :transforms => [Boltzmann.pseudo_likelihood],
  :title => "Pseudolikelihood",
  :incremental => true,
  :leg => false
)
  
features = Dict(
  :ys => [:W],
  :transforms => [W -> reshape_mnist(W')],
  :title => "Features",
  :ticks => nothing
)

reconstructions = Dict(
  :ys => [(:rbm, :X)],
  :transforms => [(rbm, X) -> reshape_mnist(generate(rbm, X[:,1:100], n_gibbs=1))],
  :title => "Reconstructions",
  :ticks => nothing
)
  
# getting the reporter
vr = VisualReporter(rbm, 600, [weights, PL, features, reconstructions], init=Dict(:X => X))

fit(rbm, X;
  n_epochs=n_epochs,
  lr=lr,
  batch_size=batch_size,
  randomize=randomize,
  scorer=Boltzmann.pseudo_likelihood,
  reporter=vr, init=Dict(:X => X)
)

mp4(vr.anim, "mnist_log.mp4", fps=2)
gif(vr.anim, "mnist_log.gif")
