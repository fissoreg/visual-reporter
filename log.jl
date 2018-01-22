# rbm, vr and all it's needed is present in scope...

nm = 10
pli = 3
svi = 8
pdi = 10
vbi = 5
hbi = 6
ki = 7
nfpi = 12

# para-ferro line
pfl = 20

function get_series(pi, si; axis=:y)
  vr.plots[pi][:plot].series_list[si][axis]
end

#p2 PL
pl = get_series(pli, 1)

# sv
sv = hcat((get_series(svi, i) for i=1:nm)...)

# pd
pdy = get_series(pdi, 1)
pdx = get_series(pdi, 1; axis=:x)
pd = hcat(pdx[:,1:end-pfl], pdy[:,1:end-pfl])

# vbias
vb = hcat((get_series(vbi,i) for i=1:5)...)

# hbias
hb = hcat((get_series(hbi,i) for i=1:5)...)

# kurtosis
k = hcat((get_series(ki,i) for i=1:2)...)

# # of fixed points
nfp = get_series(nfpi,1)

writedlm(string(dir,"/PL.dat"), pl)
writedlm(string(dir,"/sv.dat"), sv)
writedlm(string(dir,"/pd.dat"), pd)
writedlm(string(dir,"/vbias.dat"), vb)
writedlm(string(dir,"/hbias.dat"), hb)
writedlm(string(dir,"/kurtosis.dat"), k)
writedlm(string(dir,"/n_fixed_points.dat"), nfp)
