using Boltzmann
using Plots, Images
using Stats

# MNIST #############################################################
using Images

function samplesToImg(samples; c=10, r=10, h=28, w=28)
  f = zeros(r*h,c*w)
  for i=1:r, j=1:c
    f[(i-1)*h+1:i*h,(j-1)*w+1:j*w] = reshape(samples[:,(i-1)*c+j],h,w)
  end
  w_min = minimum(samples)
  w_max = maximum(samples)
  λ = x -> (x-w_min)/(w_max-w_min)
  map!(λ,f,f)
  colorview(Gray,f)
end
######################################################################

# importing function to extend
import Boltzmann.report

struct VisualReporter <: Boltzmann.BatchReporter
  every::Int
  pre::Dict{Symbol,Any}
  plots::Array{Dict{Symbol,Any},1}
  args::Dict{Symbol,Any}
end

# NOTE: :preprocessor is unique while :tranforms are array;
# it would probably be better to treat them in the same way.
function preprocessing(pre::Dict{Symbol,Any}, args::Dict)
  out = pre[:preprocessor](y_values(args, pre[:in])...)
  Dict(zip(pre[:out], out))
end

function update_args!(source::Dict, args::Dict...; pre::Dict=Dict())
  processed = isempty(pre) ? pre : preprocessing(pre, merge(source, args...))
  # NOTE: order is important! Processed args replace old args.
  merge!(source, args..., processed)
end

function get_plot_args(plot)
  # NOTE: the array of Symbols should be saved somewhere...
  filter((k,v) -> !(k in [:ys, :transforms, :incremental, :plot]), plot)
end

function y_values(args, ys)
  [typeof(ks) <: Tuple ? map(k -> args[k], ks) : args[ks] for ks in ys]
end

function apply_transforms(plot, ys)
  t = [typeof(x) <: Tuple ? f(x...) : f(x) for (f,x) in zip(plot[:transforms], ys)]
  # WARNING: there's probably a more robust way...
  if size(t)[1] == 1
    typeof(t[1]) <: Number ? t : t[1]
  else
    hcat(t')
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

  ## scalar numbers are converted to arrays to be interpreted as plotting series
  #@recipe function f(i::Int, n::Float64)
  #  [n]
  #end

  data = get_plot_data(p, args)
  p[:plot] = plot(1, data; get_plot_args(p)...)
end

function update_plot_incremental!(p, args)
  data = get_plot_data(p, args)
  for i=1:length(p[:ys])
    # NOTE: see if the following could be a special case of apply_transforms() - Done, to revise.
    push!(p[:plot], i, data[1,i]) #p[:transforms][i](args[p[:ys][i]]))
  end
end

function make_plots!(plots, args; init=false)
  for p in plots
    if haskey(p, :incremental) && p[:incremental]
      f = init ? init_plot_incremental! : update_plot_incremental!
      f(p, args)
    else
      plot_final!(p, args)
    end
  end
end

function args_aliases(rbm)
  Dict(:rbm => rbm, :W => rbm.W, :vbias => rbm.vbias, :hbias => rbm.hbias)
end

function VisualReporter(rbm::AbstractRBM, every::Int, pre::Dict{Symbol,Any}, plots::Array{Dict{Symbol,Any},1}; init=Dict())
  # initializing args is done updating "init" args
  args = Dict()
  update_args!(args, args_aliases(rbm), init; pre=pre)
  make_plots!(plots, args; init=true)
  plot(map(p -> p[:plot], plots)...)
  gui()
  println("Init done.")
  
  VisualReporter(every, pre, plots, args)
end

# activations
activations = []

function default_reporter(rbm::AbstractRBM, every::Int, X)
  pre = Dict(
    :in => [:W],
    :preprocessor => svd,
    :out => [:U, :s, :V]
  )
  
  p1 = Dict(
    :ys => [:W],
    :transforms => [x->x[:]],
    :title => "Weights",
    :seriestype => :histogram,
    :leg => false,
    :nbins => 50
  )
  
  p2 = Dict(
    :ys => [(:rbm, :X)],
    :transforms => [Boltzmann.pseudo_likelihood],
    :title => "PL",
    :incremental => true,
    :leg => false
  )
  
  p3 = Dict(
    :ys => [:W],
    :transforms => [x -> samplesToImg(x')],
    :title => "Features"
  )
  #p3= Dict(
  #  :ys => [:U, :V],
  #  :transforms => [mean, mean],
  #  :labels => ["U", "V"],
  #  :title => "Means",
  #  :incremental => true
  #)
  
  p4 = Dict(
    :ys => [:U],
    :transforms => [x->x[:]],
    :title => "U distribution",
    :seriestype => :histogram,
    :leg => false,
    :yscale => :log10
  )
  
  p5 = Dict(
    :ys => [:V],
    :transforms => [x->x[:]],
    :title => "V distribution",
    :seriestype => :histogram,
    :leg => false,
    :yscale => :log10
  )
  
  p6 = Dict(
    :ys => [(:rbm, :X)],
    :transforms => [(rbm, X) -> samplesToImg(generate(rbm, X[:,1:100], n_gibbs=15))],
    :title => "Sampling"
  )
  
  p8 = Dict(
    :ys => [:U, :V],
    :transforms => [kurtosis, kurtosis],
    :title => "Kurtosis",
    :incremental => true
  )
  
  p7 = Dict(
    :ys => [:s for i=1:50],
    :transforms => [x -> x[i] for i=1:50],
    :incremental => true,
    :title => "Singular values",
    :leg => false
  )
  
  function push_activations!(rbm::AbstractRBM, samples::Array{T,2}) where T
    m = mean(Boltzmann.sample_hiddens(rbm, samples)[1], 2)
    global activations = length(activations) == 0 ? m : hcat(activations, m)
    activations
  end
  
  p9 = Dict(
    :ys => [(:rbm, :X)],
    :transforms => [(rbm, X) -> push_activations!(rbm, X)],
    :seriestype => :heatmap,
    :title => "Activations"
  )
  
  VisualReporter(rbm, 100, pre, [p6, p1, p2, p3, p4, p5, p8, p7, p9], init=Dict(:X => X))
end

function report(reporter::VisualReporter, rbm::AbstractRBM, epoch::Int, current_batch::Int, scorer::Function, X::Boltzmann.Mat, ctx::Dict{Any,Any})
  println("Reporting")
  
  update_args!(reporter.args, Dict(:X => X), args_aliases(rbm), ctx; pre=reporter.pre)
  make_plots!(reporter.plots, reporter.args)
  plot(map(p -> p[:plot], reporter.plots)...)
  gui()
end
