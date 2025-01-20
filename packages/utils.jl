module utils

export Above, ArgAbove, Below, ArgBelow, getFilenames, readData, getData, combineDict, printTotalReward, removeEmptyArray, rewardData, observationData, BLVData, BOData, inputData

using DelimitedFiles
include("./DataAnalysis.jl") # This is a custom package 

"""Find the data above the threshold"""
function Above(data, threshold)
    return data[data .> threshold]
end
"""Finds the argument of the data above the threshold"""
function ArgAbove(data, threshold)
    return findall(x -> x > threshold, data)
end
"""Find the data below the threshold"""
function Below(data, threshold)
    return data[data .< threshold]
end
"""Finds the argument of the data below the threshold"""
function ArgBelow(data, threshold)
    return findall(x -> x < threshold, data)
end

"""Finds all files that match Regular expression in a directory and its subdirectories"""
function getFilenames(Directory::String, needle::Union{AbstractString,AbstractPattern,AbstractChar})::Vector{String}

    files = readdir(Directory)
    data = Vector{String}()

    for item in files
        
        temp_path = joinpath(Directory, item)

        if isfile(temp_path) && occursin(needle, item)
            push!(data, temp_path)
        end

        if isdir(temp_path)
            data = vcat(data, getFilenames(temp_path, needle))
        end

    end

    return data
end

"""Read all the files into an Array/Matrix"""
function readData(files::Vector{String}; removeEmpty::Bool=false)
    dict = Dict{Any, VecOrMat}()
    for (iter, file) in enumerate(files)
        dict[iter] = readdlm(file)
    end
    result = cat([val for val in values(dict)]..., dims=3)
    if removeEmpty == true
        return removeEmptyArray(result)
    end
    return result
end

"""Get data from all the files in the dictionary using the regular expression and a dictionary of the data locations"""
function getData(dataDict, needle; removeEmpty::Bool=false)
    endDict = Dict()
    for (k, v) in dataDict
        endDict[k] = readData(getFilenames(v, needle), removeEmpty=removeEmpty)
    end
    return endDict
end

"""Combines data such that the keys are unique and the values are the data from the files"""
function combineDict(directories::Vector)::Dict
    datadict = Dict()
    for item in directories
        if isa(item, Dict)
            for (k, v) in item
                !isempty(v) ? datadict[k] = v : nothing
            end
        end
    end
    return datadict
end

### Print data of the reward
function printTotalReward(datadict::Dict{String, Union{Matrix, Vector, Array}})
    println("Total reward for the agents are:")
    for (key, value) in datadict
        res = stats(sum(value, dims=1))
        println("$(rpad("$key:", 6)) $(lpad(res[1], 18)) +- $(rpad(res[2], 18))")
    end
end

# remove empty data
function removeEmptyArray(data::Union{Matrix, Array})
    iter = size(data, 3)
    removeInd = []
    for index in 1:iter
        if all(data[:, end, index].==0)
            push!(removeInd, index)
        end
    end
    return data[:, :, setdiff(1:iter, removeInd)]
end


### NOTE : This function does not use the BittleExtract package. This shoudl be update such that the data is better handled

### Get data
"""Reads all the data from the reward files and returns a dictionary with the keys as the folder names and the values as the data from the files"""
function rewardData(location_dict::Dict; statistics::Bool=true, removeEmpty::Bool=false)
    datadict = combineDict([getData(location_dict, r".*?reward.csv", removeEmpty=removeEmpty), getData(location_dict, r".*?RewardPlot.csv", removeEmpty=removeEmpty)])
    
    for (key, value) in datadict
        value = reduceDims(value)
        if removeEmpty == true
            value = data[key][:, data[key][end, :] .!= 0]
        end
        if statistics == true
            value = DataAnalysis.stats(value)
        end
        datadict[key] = value
    end
    return convert(Dict{String, Union{Matrix, Vector, Array}}, datadict)
end

"""Read all the data from the observation files and returns a dictionary with the keys as the folder names and the values as the data from the files"""
function observationData(location_dict::Dict; statistics::Bool=true, removeEmpty::Bool=false)
    datadict = combineDict([getData(location_dict, r".*?Observation.csv", removeEmpty=removeEmpty), getData(location_dict, r".*?y_EFE.csv", removeEmpty=removeEmpty)])
    if statistics == true
        for (key, value) in datadict
            datadict[key] = DataAnalysis.stats(value, 3)
        end
    end
    return convert(Dict{String, Union{Matrix, Vector, Array}}, datadict)
end

"""Read all the data from the BLV files and returns a dictionary with the keys as the folder names and the values as the data from the files"""
function BLVData(location_dict::Dict; statistics::Bool=true, removeEmpty::Bool=false)
    returndict = Dict{String, Dict}()
    datadict = combineDict([getData(location_dict, r".*?Observation.csv", removeEmpty=removeEmpty), getData(location_dict, r".*?y_EFE.csv", removeEmpty=removeEmpty)])

    for key in keys(datadict)
        returndict[key] = Dict{String, VecOrMat}()

        for (iter, item) in enumerate(["vx", "vy", "vz"])
            # if removeEmpty == true
            #     datadict[key][item] = datadict[key][item][:, datadict[key][end, :] .!= 0]
            # end
            if statistics == true
                returndict[key][item] = DataAnalysis.stats(datadict[key][iter+18, :, :], 2)
            else
                returndict[key][item] = datadict[key][iter+18, :, :]
            end
        end
    end
    return returndict
end

function BOData(location_dict::Dict; statistics::Bool=true, removeEmpty::Bool=false)
    returndict = Dict{String, Dict}()
    datadict = combineDict([getData(location_dict, r".*?Observation.csv", removeEmpty=removeEmpty), getData(location_dict, r".*?y_EFE.csv", removeEmpty=removeEmpty)])

    for key in keys(datadict)
        returndict[key] = Dict{String, Union{VecOrMat, Array}}()

        for (iter, item) in enumerate(["roll", "pitch"])#, "yaw"])
            if statistics == true
                returndict[key][item] = DataAnalysis.stats(datadict[key][iter+16, :, :], 2)
            else
                returndict[key][item] = datadict[key][iter+16, :, :]
            end
        end
    end
    return returndict
end

"""Read all the data from the input files and returns a dictionary with the keys as the folder names and the values as the data from the files"""
function inputData(location_dict::Dict; statistics::Bool=true, removeEmpty::Bool=false)
    datadict = combineDict([getData(location_dict, r".*?u_EFE.csv", removeEmpty=removeEmpty), getData(location_dict, r".*?Action.csv", removeEmpty=removeEmpty)])
    
    for (key, value) in datadict
        value = reduceDims(value)
        if statistics == true
            value = DataAnalysis.stats(value)
        end
        datadict[key] = value
    end
    return convert(Dict{String, Union{Matrix, Vector, Array}}, datadict)
end

end