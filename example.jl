using Boltzmann
using MNIST

include("reporter.jl")
include("../mf-rbm/mf.jl")

X, y = testdata()  # test data is smaller, no need to downsample
X = 2(X  ./ (maximum(X) - minimum(X))) - 1

rbm = IsingRBM(28*28, 100; sigma=0.001, X=X)

vr = default_reporter(rbm, 100, X)
fit(rbm, X, n_epochs=500, lr=1e-6, batch_size=20, eps=1e-8, max_iter=1000, randomize=true, reporter=vr, init=Dict(:X => X))
#sampler=tap_gradient,
mp4(vr.anim, "log.mp4", fps=2)
