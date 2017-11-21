#module VisualReporter

using Boltzmann
using Plots, Images
using Stats

# importing function to extend
import Boltzmann.report

# TODO:
#  - define constructor (initializing plots in it) - Done!
#  - implement preprocessing - Done!
#  - eval() should probably be removed, using a Dict - Done!
#  - how to pass Plots arguments to plot()? - SOLVED: splat Dict after semicolon, arg names being symbols

struct VisualReporter <: Boltzmann.BatchReporter
  every::Int
  pre::Dict{Symbol,Any}
  plots::Array{Dict{Symbol,Any},1}
end

function get_args(rbm::AbstractRBM, args::Dict{Symbol,Any}...; pre=Dict())
  new_args = isempty(pre) ? pre : preprocessing(pre, args)
  merge(Dict(:W => rbm.W, :vbias => rbm.vbias, :hbias => rbm.hbias), new_args, args...)
end

function get_plot_args(plot)
  # NOTE: the array of Symbols should be saved somewhere...
  filter((k,v) -> !(k in [:ys, :transforms, :incremental, :plot]), plot)
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
  plot[:plot] = plot(apply_transform(plot, ys), get_plot_args(plot))
end

function init_plot_incremental(plot, args)
  plot[:plot] = plot(0, apply_transforms(plot, ys), get_plot_args(plot))
end

function update_plot_incremental(plot, args)
  for i=1:length(plot[:ys])
    # NOTE: see if the following could be a special case of apply_transforms()
    push!(plot[:plot], i, plot[:transforms][i](plot[:ys][i]))
  end
end

function preprocessing(pre::Dict{Symbol,Any}, args::Dict{Symbol,Any})
  out = pre[:preprocessing](y_values(args, pre[:in])...)
  Dict(zip(pre[:out], out))
end

function VisualReporter(rbm::AbstractRBM, every::Int, pre::Dict{Symbol,Any}, plots::Array{Dict{Symbol,Any},1}, init::Dict{Symbol,Any})
  args = get_args(rbm, init, pre=pre)
   
  for p in plots
    if haskey(plot, :incremental) && plot[:incremental]
      init_plot_incremental(plot, args)
    else
      plot_final(plot, args)
    end
  end
end

function report(reporter::VisualReporter, rbm::AbstractRBM, epoch::Int, current_batch::Int, scorer::Function, X::Boltzmann.Mat, ctx::Dict{Any,Any})

  #for plot in reporter.plots
  #end
end

pre = Dict(
  :in => [:W],
  :preprocessor => svd,
  :out => [:U, :s, :V]
)

p1 = Dict(
  :ys => [:X],
  :transforms => [x -> samplesToImg(generate(x[:,1:100], n_gibbs=15))],
  :title => "Sampling"
)

p2 = Dict(
  :ys => [:U, :V],
  :transforms => [mean, mean],
  :labels => ["U", "V"],
  :title => "Means",
  :incremental => true
)

X = randn(784, 2000)    # 2000 observations (examples) #  with 100 variables (features) each
X = (X + abs(minimum(X))) / (maximum(X) - minimum(X)) # scale X to [0..1]
rbm = GRBM(784, 50)     # define Gaussian RBM with 100 visible (input) 
                        #  and 50 hidden (output) variables
vr = VisualReporter(10, pre, [p1, p2])
fit(rbm, X, reporter=vr)        
