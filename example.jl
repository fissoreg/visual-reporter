using Boltzmann
using MNIST

include("reporter.jl")

X, y = testdata()  # test data is smaller, no need to downsample
X = X .* 2 ./ (maximum(X) - minimum(X)) - 1

rbm = IsingRBM(28*28, 100; TrainData=X)

vr = default_reporter(rbm, 100, X)
fit(rbm, X, n_epochs=50, lr=0.00001, batch_size=20, n_gibbs=5, randomize=true, reporter=vr)
