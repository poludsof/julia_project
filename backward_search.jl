
function dfs(nn, fix_inputs::Vector{Int}, start::Int, input, output)
    # println(fix_inputs)

    status, _ = adversarial(nn, input, output, fix_inputs)
    println("TEST ON:", length(fix_inputs), " status: ", status)
    if status == :success #  -> stop searching
        println("stop searching")
        return fix_inputs
    end    

    best_set = fix_inputs

    for i in start:length(fix_inputs)
        next_set = setdiff(fix_inputs, [fix_inputs[i]])
        new_fix_inputs = dfs(nn, next_set, i, input, output)
        if length(new_fix_inputs) < length(best_set)
            best_set = new_fix_inputs
        end
    end
    best_set
end

function minimal_set_dfs(nn::Chain, input, output)
    fix_inputs = collect(1:length(input))
    dfs(nn, fix_inputs, 1, input, output)
end


# function dfs_generate_subsets(arr::Vector{Int})
#     function dfs(current_set::Vector{Int})
#         println(current_set)

#         for i in eachindex(current_set)
#             next_set = setdiff(current_set, [current_set[i]])
#             dfs(next_set)
#         end
#     end

#     dfs(arr)
# end

# function dfs_unique_subsets(arr::Vector{Int})
#     function dfs(current_set::Vector{Int}, start::Int)
#         println(current_set)

#         for i in start:length(current_set)
#             next_set = setdiff(current_set, [current_set[i]])
#             dfs(next_set, i)
#         end
#     end

#     dfs(arr, 1)
# end


function dfs_optimized(nn, fix_inputs::Vector{Int}, start::Int, input, output)
    current_set = Set(fix_inputs)
    
    if haskey(checked_sets, current_set)
        if !checked_sets[current_set]
            return fix_inputs
        end
    else
        status, _ = adversarial(nn, input, output, fix_inputs=fix_inputs)
        println("TEST ON:", length(fix_inputs), " status: ", status)
        is_adversarial = (status != :feasible)
        checked_sets[current_set] = is_adversarial
        if !is_adversarial
            return fix_inputs
        end
    end    

    best_set = fix_inputs

    for i in start:length(fix_inputs)
        next_set = setdiff(fix_inputs, [fix_inputs[i]])
        new_fix_inputs = dfs_optimized(nn, next_set, i, input, output)
        
        if length(new_fix_inputs) < length(best_set)
            best_set = new_fix_inputs
        end
    end
    best_set
end

const checked_sets = Dict{Set{Int}, Bool}()
function minimal_set_dfs_optimized(nn::Chain, input, output)
    fix_inputs = collect(1:length(input))
    # checked_sets[] = Dict()
    empty!(checked_sets)
    dfs_optimized(nn, fix_inputs, 1, input, output)
end
