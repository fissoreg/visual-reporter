using Boltzmann
using Plots, Images
using Stats

struct VisualReporter <: Boltzmann.BatchReporter
  every::Int
  exec::Function
end

function samplesToImg(samples; c=10, r=10)
  f = zeros(r*28,c*28)
  for i=1:r, j=1:c
    f[(i-1)*28+1:i*28,(j-1)*28+1:j*28] = reshape(samples[:,(i-1)*c+j],28,28)
  end
  w_min = minimum(samples)
  w_max = maximum(samples)
  λ = x -> (x-w_min)/(w_max-w_min)
  map!(λ,f,f)
  colorview(Gray,f)
end

function saveSamples(samples;path="samples.jpg")
  img = samplesToImg(samples)
  println(path)
  Images.save(path,img)
  #plot(img)
end

p2 = plot([mean,var,kurtosis], 1, rand(100))

function exec(rbm::AbstractRBM, epoch::Int, score, ctx::Dict{Any,Any})
  U,s,V = svd(rbm.W)
  
  p1 = plot(samplesToImg(ctx[:persistent_chain]))
  push!(uk, kurtosis(U))
  push!(p2, 1, mean(U))
  push!(p2, 2, var(U))
  push!(p2, 3, kurtosis(U))
  p3 = histogram(U[:], nbins=100, yscale=:log10)
  p4 = histogram(V[:], nbins=100, yscale=:log10)

  #gui(p1)
  println(epoch)
  plot(p1, p2, p3, p4)
  gui()
end

X = randn(784, 2000)    # 2000 observations (examples) #  with 100 variables (features) each
X = (X + abs(minimum(X))) / (maximum(X) - minimum(X)) # scale X to [0..1]
rbm = GRBM(784, 50)     # define Gaussian RBM with 100 visible (input) 
                        #  and 50 hidden (output) variables
fit(rbm, X, reporter=VisualReporter(1,exec))        
