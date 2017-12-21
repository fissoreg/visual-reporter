using Boltzmann
using Plots, Images
using Stats
using MNIST_utils

include("../mf-rbm/mf.jl")

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

function VisualReporter(rbm::AbstractRBM, every::Int, pre::Dict{Symbol,Any}, plots::Array{Dict{Symbol,Any},1}; init=Dict())
  # initializing args is done updating "init" args
  args = Dict()
  update_args!(args, args_aliases(rbm), init; pre=pre)
  make_plots!(plots, args; init=true)
  plot(map(p -> p[:plot], plots)...)
  gui()
  anim = Animation()
  #frame(anim)
  println("Init done.")
  
  VisualReporter(every, pre, plots, args, anim)
end

# activations
activations = []

# fixed points
fps = []

function preprop(rbm::RBM,X)
  U,s,V = svd(rbm.W)
  global fps = condensed_fps(rbm; fps=fps, params=Dict(:max_iter=>1000,:X=>X))
  U,s,V,fps
end

function fps_to_img(fps)
  # row size
  row = 10
  n = size(fps, 2)
  println("# fps: ", n)

  r = n > row ? div(n,row) : 1
  c = n > row ? row : n
  range = 1:(n > row ? r*row : n)
  
  samplesToImg(fps[:,range], r=r, c=c)
end

function default_reporter(rbm::AbstractRBM, every::Int, X)
  pre = Dict(
    :in => [:rbm,:X],
    :preprocessor => preprop,
    :out => [:U, :s, :V, :fps]
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
    m = mean(Boltzmann.sample_hiddens(rbm, samples), 2)
    global activations = length(activations) == 0 ? m : hcat(activations, m)
    activations
  end
  
  p9 = Dict(
    :ys => [(:rbm, :X)],
    :transforms => [(rbm, X) -> push_activations!(rbm, X)],
    :seriestype => :heatmap,
    :title => "Activations"
  )
  
  # highest singular value (time-series)
  was = []
  variances = []
  
  function phase_diagram!(W::Array{T,2}, wa::Real) where T
    push!(variances, var(W))
    push!(was, wa)
    (was ./ variances, 1 ./ variances)
  end
  
  p10 = Dict(
    :ys => [(:W, :s)],
    :transforms => [(W,s) -> phase_diagram!(W, s[1])],
    :seriestype => :scatter
  )

  function fp(rbm, X)
    mv, mh = fp_iter(rbm, X[:,1:100], tap_v_fp, tap_h_fp; max_iter = 20)
    samplesToImg(mv)
  end

  p11 = Dict(
    :ys => [(:rbm, :X)],
    :transforms => [fp],
    :title => "Fixed points"
  )

  p12 = Dict(
    :ys => [:fps],
    :transforms => [fps -> size(fps, 2)],
    :incremental => true,
    :title => "# fixed points",
    :leg => false
  )


  p13 = Dict(
    :ys => [:fps],
    :transforms => [fps_to_img],
    :title => "Fixed points"
  )

  VisualReporter(rbm, 100, pre, [p6, p1, p2, p3, p4, p5, p8, p7, p9, p10, p11, p12, p13], init=Dict(:X => X))
end

function report(reporter::VisualReporter, rbm::AbstractRBM, epoch::Int, current_batch::Int, scorer::Function, X::Boltzmann.Mat, ctx::Dict{Any,Any})
  println("Reporting")
  
  update_args!(reporter.args, Dict(:X => X), args_aliases(rbm), ctx; pre=reporter.pre)
  make_plots!(reporter.plots, reporter.args)
  plot(map(p -> p[:plot], reporter.plots)...)
  gui()
  #frame(reporter.anim)
end
