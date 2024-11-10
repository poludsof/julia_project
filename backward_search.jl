
# function dfs(nn, fix_inputs::Vector{Int}, start::Int, input, output)
#     # println(fix_inputs)

#     status, _ = adversarial(nn, input, output, fix_inputs)
#     println("TEST ON:", length(fix_inputs), " status: ", status)
#     if status == :success #  -> stop searching
#         println("stop searching")
#         return fix_inputs
#     end    

#     best_set = fix_inputs

#     for i in start:length(fix_inputs)
#         next_set = setdiff(fix_inputs, [fix_inputs[i]])
#         new_fix_inputs = dfs(nn, next_set, i, input, output)
#         if length(new_fix_inputs) < length(best_set)
#             best_set = new_fix_inputs
#         end
#     end
#     best_set
# end

global visited = Set{Vector{Int}}()

function dfs_cache(nn, fix_inputs::Vector{Int}, input, output, steps::Int, max_steps::Int, found_minimal_set::Ref{Bool})
    # Check if max steps have been reached
    if steps >= max_steps || found_minimal_set[]
        return fix_inputs
    end

    # Return if the current set has already been visited
    if in(fix_inputs, visited)
        return fix_inputs
    end

    push!(visited, fix_inputs)

    status, _ = adversarial(nn, input, output, fix_inputs)
    println("TEST ON:", length(fix_inputs), " status: ", status)
    
    if status == :success
        println("stop searching")
        return fix_inputs
    end
    
    if length(fix_inputs) <= 743
        found_minimal_set[] = true
        return fix_inputs
    end

    best_set = fix_inputs

    for i in 1:length(fix_inputs)
        next_set = setdiff(fix_inputs, [fix_inputs[i]])
        new_fix_inputs = dfs_cache(nn, next_set, input, output, steps + 1, max_steps, found_minimal_set)
        
        if found_minimal_set[]
            return new_fix_inputs
        end

        if length(new_fix_inputs) < length(best_set)
            best_set = new_fix_inputs
        end
    end

    return best_set
end

function minimal_set_dfs(nn::Chain, input, output)
    fix_inputs = collect(1:length(input))
    tmp_inputs = collect(1:4)
    println(tmp_inputs)
    global visited = Set{Vector{Int}}()
    found_minimal_set = Ref(false)
    result = dfs_cache(nn, fix_inputs, input, output, 0, 100, found_minimal_set)
    # dfs(nn, fix_inputs, 1, input, output)
    return result
end
