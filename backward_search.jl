
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

function dfs_cache(nn, fix_inputs::Vector{Int}, input, output)

    if in(fix_inputs, visited)
        # println("current set: ", fix_inputs, " already visited")
        return fix_inputs
    end

    # println("current set: ", fix_inputs)
    push!(visited, fix_inputs)

    status, _ = adversarial(nn, input, output, fix_inputs)
    println("TEST ON:", length(fix_inputs), " status: ", status)
    
    if status == :success
        println("stop searching")
        return fix_inputs
    end    

    best_set = fix_inputs

    for i in 1:length(fix_inputs)
        next_set = setdiff(fix_inputs, [fix_inputs[i]])
        if (length(next_set) == 0)
            return best_set
        end
        new_fix_inputs = dfs_cache(nn, next_set, input, output)
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
    global visited = Set{Vector{Int}}()  # Initialize the global cache
    dfs_cache(nn, fix_inputs, input, output)
    # dfs(nn, fix_inputs, 1, input, output)
end
