using Boltzmann
using MNIST

nh = 100
sigma = 0.001
n_epochs = 2
lr = 1e-6
batch_size = 20
eps = 1e-8
max_iter = 1000
randomize = true
name = "rbm"

include("reporter.jl")
include("../mf-rbm/mf.jl")

X, y = traindata()  # test data is smaller, no need to downsample
X = 2(X  ./ (maximum(X) - minimum(X))) - 1

rbm = IsingRBM(28*28, nh; sigma=sigma, X=X)

start = time()

vr = default_reporter(rbm, 100, X)
fit(rbm, X;
    n_epochs=n_epochs,
    lr=lr,
    batch_size=batch_size,
    eps=eps,
    max_iter=max_iter,
    randomize=randomize,
    scorer=Boltzmann.pseudo_likelihood,
    reporter=vr, init=Dict(:X => X)
   )

t = time() - start
println("Elapsed: $(t/60/60)")

#sampler=tap_gradient,

filename = string(name, "_", nh,"_",sigma,"_",n_epochs,"_",lr,"_",batch_size,"_",eps,"_",max_iter,"_",randomize)
dir = string("log/",filename)

try
  mkdir("log")
  mkdir(string("log/",filename))
end

include("log.jl")

mp4(vr.anim, string(dir,"/",filename,".mp4"), fps=2)
gif(vr.anim, string(dir,"/",filename,".gif"))

writedlm(string(dir,"/W.dat"),rbm.W)
writedlm(string(dir,"/vbias.dat"),rbm.vbias)
writedlm(string(dir,"/hbias.dat"), rbm.hbias)
