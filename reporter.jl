#module VisualReporter

using Boltzmann
using Plots, Images
using Stats

# importing function to extend
import Boltzmann.report

struct VisualReporter <: Boltzmann.BatchReporter
  every::Int
  pre::Dict{Symbol,Any}
  plots::Array{Dict{Symbol,Any},1}
end

function preprocessing(pre::Dict{Symbol,Any}, args::Dict)
  out = pre[:preprocessor](y_values(args, pre[:in])...)
  Dict(zip(pre[:out], out))
end

function get_args(rbm::AbstractRBM, args::Dict... ; pre=Dict())
  full_args = merge(Dict(:W => rbm.W, :vbias => rbm.vbias, :hbias => rbm.hbias), args...)
  new_args = isempty(pre) ? pre : preprocessing(pre, full_args)
  merge(new_args, full_args)
end

function get_plot_args(plot)
  # NOTE: the array of Symbols should be saved somewhere...
  filter((k,v) -> !(k in [:ys, :transforms, :incremental, :plot]), plot)
end

function y_values(args, ys)
  [args[k] for k in ys]
end

function apply_transforms(plot, ys)
  t = [f(x) for (f,x) in zip(plot[:transforms], ys)]
  # WARNING: there's probably a more robust way...
  if size(t)[1] == 1
    t[1]
  else
    hcat(t)
  end
end

function get_plot_data(p, args)
  ys = y_values(args, p[:ys])
  apply_transforms(p, ys)
end

function plot_final!(p, args)
  p[:plot] = plot(get_plot_data(p, args); get_plot_args(p)...)
end

function init_plot_incremental!(p, args)
  p[:plot] = plot(0, get_plot_data(p, args); get_plot_args(p)...)
end

function update_plot_incremental!(p, args)
  for i=1:length(p[:ys])
    # NOTE: see if the following could be a special case of apply_transforms()
    push!(p[:plot], i, p[:transforms][i](p[:ys][i]))
  end
end

function VisualReporter(rbm::AbstractRBM, every::Int, pre::Dict{Symbol,Any}, plots::Array{Dict{Symbol,Any},1}; init=Dict())
  println("IN")
  args = get_args(rbm, init, pre=pre) 
  for p in plots
    if haskey(p, :incremental) && p[:incremental]
      init_plot_incremental!(p, args)
    else
      plot_final!(p, args)
    end
  end
  plot(map(p -> p[:plot], plots)...)
  gui()
  println("Init done.")
  
  VisualReporter(every, pre, plots)
end

function report(reporter::VisualReporter, rbm::AbstractRBM, epoch::Int, current_batch::Int, scorer::Function, X::Boltzmann.Mat, ctx::Dict{Any,Any})
  println("Reporting")
  #for plot in reporter.plots
  #end
end

using MNIST_utils

X = randn(784, 2000)    # 2000 observations (examples) #  with 100 variables (features) each
X = (X + abs(minimum(X))) / (maximum(X) - minimum(X)) # scale X to [0..1]
rbm = GRBM(784, 50)     # define Gaussian RBM with 100 visible (input) 
                        #  and 50 hidden (output) variables

pre = Dict(
  :in => [:W],
  :preprocessor => svd,
  :out => [:U, :s, :V]
)

p1 = Dict(
  :ys => [:X],
  :transforms => [x -> samplesToImg(generate(rbm, x[:,1:100], n_gibbs=15))],
  :title => "Sampling"
)

p2 = Dict(
  :ys => [:U, :V],
  :transforms => [mean, mean],
  :labels => ["U", "V"],
  :title => "Means",
  :incremental => true
)

vr = VisualReporter(rbm, 10, pre, [p1, p2], init=Dict(:X => X))
fit(rbm, X, reporter=vr) 
