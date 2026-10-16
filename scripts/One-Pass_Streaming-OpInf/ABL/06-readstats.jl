"""
Verifying the final statistical outputs of the ABL data from AMR-Wind.
"""

##
using NCDatasets

##
FILEPATH = occursin("scripts", pwd()) ? 
           joinpath(pwd(),"One-Pass_Streaming-OpInf/ABL") : 
           joinpath(pwd(), "scripts/One-Pass_Streaming-OpInf/ABL")
fn = "data/results/abl_statistics.nc"
datafile = joinpath(FILEPATH, fn)

##
ds = NCDataset(datafile)


##
using CairoMakie 

##
ustar = ds["ustar"][:]

fig = Figure()
ax = Axis(fig[1, 1], xlabel="t", ylabel="u*")
lines!(ax, ustar)
display(fig)

##
wstar = ds["wstar"][:]

fig = Figure()
ax = Axis(fig[1, 1], xlabel="t", ylabel="w*")
lines!(ax, wstar)
display(fig)

##
forcing_y = ds["abl_forcing_y"][:]

fig = Figure()
ax = Axis(fig[1, 1], xlabel="t", ylabel="forcing_y")
lines!(ax, forcing_y)
display(fig)


##
forcing_x = ds["abl_forcing_x"][:]

fig = Figure()
ax = Axis(fig[1, 1], xlabel="t", ylabel="forcing_x")
lines!(ax, forcing_x)
display(fig)


##
h = ds.group["mean_profiles"]["h"][:]

##
uw = ds.group["mean_profiles"]["u'w'_r"][:]
uw = reshape(uw, 96, :)

fig = Figure()
ax = Axis(fig[1, 1], ylabel="z", xlabel="u'w'_r")
lines!(ax, uw[:,1], h)
display(fig)

##
vw = ds.group["mean_profiles"]["v'w'_r"][:]
vw = reshape(vw, 96, :)

fig = Figure()
ax = Axis(fig[1, 1], ylabel="z", xlabel="v'w'_r")
lines!(ax, vw[:,1], h)
display(fig)

## 
u = ds.group["mean_profiles"]["u"][:]
u = reshape(u, 96, :)

## 
v = ds.group["mean_profiles"]["v"][:]
v = reshape(v, 96, :)


## Wind speed 
u_mean = sum(u, dims=2) / size(u, 2)
v_mean = sum(v, dims=2) / size(v, 2)
wind_speed = vec(sqrt.(u_mean.^2 .+ v_mean.^2))

fig = Figure()
ax = Axis(fig[1, 1], ylabel="z", xlabel="wind speed")
lines!(ax, wind_speed, h)
display(fig)


## Wind direction 
wind_direction = vec(atan.(v_mean, u_mean)) * 180 / pi 
wind_direction = wind_direction .- wind_direction[end] # set the first value to zero

fig = Figure()
ax = Axis(fig[1, 1], ylabel="z", xlabel="wind direction (radians)")
lines!(ax, wind_direction, h)
display(fig)
