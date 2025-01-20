module BittleExtract

using JSON
using Plots; default(grid=false, label="", linewidth=3,margin=20Plots.pt)

export updateObservationConfig, plotStand, RemoveVariable, RemoveVariable!

"""Remove a variable from the list of variables"""
function RemoveVariable(allVariable::Array{String}, varname::String)
    for (iter, key) in enumerate(allVariable)
        if key == varname
            allVariable = [allVariable[1:iter-1]; allVariable[iter+1:end]]
            return allVariable
        end
    end
end

"""Remove a variable from the list of variables"""
function RemoveVariable!(allVariable::Array{String}, varname::String)
    for (iter, key) in enumerate(allVariable)
        if key == varname
            allVariable = [allVariable[1:iter-1]; allVariable[iter+1:end]]
        end
    end
end

"""Update the observation configuration for easy plotting and storing of data based on a JSON config file

# Arguments
- env::PyObject: The environment object
- filename::String: The name of the JSON file to load the configuration from

# Returns
- use_config::Dict: The updated configuration
- allVariable::Vector: The list of all variables to be plotted
"""
function updateObservationConfig(env::PyObject; filename::String="JSONConfig.json")
    
    allVariable = ["JA", "JV", "JT", "BO", "BLV", "BAV", "FC", "Aprev"] # list of sorted variables (MUST BE SORTED to output)
    use_config = JSON.parsefile(filename) # load setup dict
        
    observation_config = env.unwrapped._observations_config
    for (key, val) in use_config
        
        if isa(observation_config[key], Bool) && observation_config[key]
        
            val["ActualSize"] = val["size"]
    
        elseif isa(observation_config[key], Bool) && !observation_config[key]
            
            val["ActualSize"] = 0
            allVariable = RemoveVariable(allVariable, key)# remove element if it is not active
    
        elseif (observation_config[key]["active"] |> sum) > 0
            
            val["ActualSize"] = (observation_config[key]["active"] |> sum)
    
        else
    
            val["ActualSize"] = 0
            allVariable = RemoveVariable(allVariable, key)
    
        end

        # convert label to plot label
        val["label"] = hcat(convert(Vector{String}, val["label"])...) 
    end

    # add range to Dict
    start_range = 1
    for name in allVariable
        end_range = use_config[name]["ActualSize"]
        range = UnitRange{Int64}(start_range:start_range+end_range-1)
        use_config[name]["range"] = range
        start_range += end_range
    end

    return use_config, allVariable
end

# deprecated
"""Plot the observation data"""
function plotStand(lastIter::Integer, observation::Union{VecOrMat}, observationConfig::Dict, allVariable::Vector)::Dict
    plotdict = Dict{}()
    for i in allVariable
        if i in ["FC", "Aprev"]
            plotdict[i] = heatmap(transpose(observation[1:lastIter, observationConfig[i]["range"]]),
                                  title=observationConfig[i]["name"], 
                                  xlabel=observationConfig[i]["xlabel"], 
                                  ylabel=observationConfig[i]["ylabel"],
                                  label=observationConfig[i]["label"],
                                  size=(800,800),
                                  yflip=true, 
                                #   xticks=0:20:lasti,
                                  c=[:white, :black],
                                  yticks=1:observationConfig[i]["ActualSize"],
                                  colorbar=(allVariable == "FC" ? true : false))
        else
            plotdict[i] = plot(observation[1:lastIter, observationConfig[i]["range"]], 
                               title=observationConfig[i]["name"], 
                               xlabel=observationConfig[i]["xlabel"], 
                               ylabel=observationConfig[i]["ylabel"],
                               label=observationConfig[i]["label"],
                               size=(800,800))
        end
    end
    plotdict["all"] = plot([plotdict[i] for i in allVariable]..., layout=(Int64(ceil(size(allVariable, 1) / 2)), 2))
    return plotdict
end

end # module