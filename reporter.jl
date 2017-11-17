using Boltzmann
using Plots, Images
using Stats
using Boltzmann
using MNIST_utils

# TODO:
#  - define constructor (initializing plots in it)
#  - implement preprocessing
#  - eval() should probably be removed, using a Dict
#  - how to pass Plots arguments to plot()?

struct VisualReporter <: Boltzmann.BatchReporter
  every::Int
  pre::Dict{Symbol,Any}
  plots::Array{Dict{Symbol,Any},1}
end

pre = Dict(
  :in => :W,
  :preprocessor => svd,
  :out => (:U, :s, :V)
)

p1 = Dict(
  :ys => [:X],
  :transforms = [x -> samplesToImg(generate(x[:,1:100], n_gibbs=15))],
  :title = "Sampling"
)

p2 = Dict(
  :ys => [:U, :V],
  :transforms => [mean, mean],
  :labels => ["U", "V"],
  :title => "Means",
  :incremental => true
)

function report(reporter::VisualReporter, rbm::AbstractRBM, epoch::Int, current_batch::Int, scorer::Function, X::Boltzmann.Mat, ctx::Dict{Any,Any})
  
  # aliases (make :W, :v and :h symbols available)
  W = rbm.W
  v = rbm.vbias
  h = rbm.hbias

  for plot in reporter.plots
    if haskey(plot, :incremental) && plot[:incremental]
      for i=1:length(plot[:ys])
        push!(plot[:plot], i, plot[:transforms][i](eval(plot[:ys][i])))
      end
    else
        plot[:plot] = plot(hcat(map((f,x) -> f(eval(x)), zip(plot[:transforms], plot[:ys]))))
    end
  end
end

X = randn(784, 2000)    # 2000 observations (examples) #  with 100 variables (features) each
X = (X + abs(minimum(X))) / (maximum(X) - minimum(X)) # scale X to [0..1]
rbm = GRBM(784, 50)     # define Gaussian RBM with 100 visible (input) 
                        #  and 50 hidden (output) variables
fit(rbm, X, reporter=VisualReporter(1,exec))        
