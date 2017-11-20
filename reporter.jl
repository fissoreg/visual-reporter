using Boltzmann
using Plots, Images
using Stats
using Boltzmann
using MNIST_utils

# TODO:
#  - define constructor (initializing plots in it) - Done!
#  - implement preprocessing - Settling...
#  - eval() should probably be removed, using a Dict
#  - how to pass Plots arguments to plot()? - SOLVED: splat Dict after semicolon, arg names being symbols

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

function get_args(rbm::AbstractRBM, args::Dict{Symbol,Any}...)
  merge(Dict(:W => rbm.W, :vbias => rbm.vbias, :hbias => rbm.hbias), args...)
end

function y_values(args, ys)
  [args[k] for k in ys]
end

function apply_transforms(plot, ys)
  hcat(map((f,x) -> f(x), zip(plot[:transforms], ys)))
end

function plot_final(plot, args)
  ys = y_values(plot[:ys])
  transformed = apply_transforms(plot, ys)
  plot[:plot] = plot(apply_transform(plot, ys))
end

function init_plot_incremental(plot, args)
  plot[:plot] = plot(0, apply_transforms(plot, ys))
end

function update_plot_incremental(plot, args)
  for i=1:length(plot[:ys])
    # NOTE: see if the following could be a special case of apply_transforms()
    push!(plot[:plot], i, plot[:transforms][i](plot[:ys][i]))
  end
end

function VisualReporter(rbm::AbstractRBM, every::Int, pre::Dict{Symbol,Any}, plots::Array{Dict{Symbol,Any},1}, init::Dict{Symbol,Any})
  args = get_args(rbm, init)
  new_args = preprocessing(pre, args)
  args = merge(args, new_args)
   
  for p in plots
    if haskey(plot, :incremental) && plot[:incremental]
      init_plot_incremental(plot, args)
    else
      plot_final(plot, args)
    end
  end
end

function report(reporter::VisualReporter, rbm::AbstractRBM, epoch::Int, current_batch::Int, scorer::Function, X::Boltzmann.Mat, ctx::Dict{Any,Any})
  
  # aliases (make :W, :v and :h symbols available)
  W = rbm.W
  v = rbm.vbias
  h = rbm.hbias

  for plot in reporter.plots
  end
end

X = randn(784, 2000)    # 2000 observations (examples) #  with 100 variables (features) each
X = (X + abs(minimum(X))) / (maximum(X) - minimum(X)) # scale X to [0..1]
rbm = GRBM(784, 50)     # define Gaussian RBM with 100 visible (input) 
                        #  and 50 hidden (output) variables
fit(rbm, X, reporter=VisualReporter(1,exec))        
