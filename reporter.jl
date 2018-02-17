using Boltzmann
using Plots, Images

# font settings
f = Plots.font("Helvetica",4)
fs = Dict(:guidefont => f, :legendfont => f, :tickfont => f)
default(; fs...)

# importing function to extend
import Boltzmann.report

struct VisualReporter <: Boltzmann.BatchReporter
  every::Int
  pre::Dict{Symbol,Any}
  plots::Array{Dict{Symbol,Any},1}
  args::Dict{Symbol,Any}
  anim::Animation
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
  # WARNING: there's probably a more robust way... Exactly!
  # And that consists into using plot!()...
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

function VisualReporter(rbm::AbstractRBM, every::Int, plots::Array{Dict{Symbol,Any},1}; init=Dict(), pre::Dict{Symbol,Any} = Dict{Symbol,Any}())
  # initializing args is done updating "init" args
  args = Dict()
  update_args!(args, args_aliases(rbm), init; pre=pre)
  make_plots!(plots, args; init=true)
  plot(map(p -> p[:plot], plots)...)
  #gui()
  anim = Animation()
  #frame(anim)
  
  VisualReporter(every, pre, plots, args, anim)
end

function report(reporter::VisualReporter, rbm::AbstractRBM, epoch::Int, current_batch::Int, scorer::Function, X::Boltzmann.Mat, ctx::Dict{Any,Any})
  
  update_args!(reporter.args, Dict(:X => X), args_aliases(rbm), ctx; pre=reporter.pre)
  make_plots!(reporter.plots, reporter.args)
  plot(map(p -> p[:plot], reporter.plots)...)
  gui()
  frame(reporter.anim)
  
  # textual logging
  println("Epoch $epoch - Batch $current_batch: $(scorer(rbm,X))")
end
