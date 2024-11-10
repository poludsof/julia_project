
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

function dfs_cache_non_recursive(nn, given_input_set::Vector{Int}, input, output, max_steps::Int)
    visited = Set{Vector{Int}}()
    stack = [(given_input_set, 0)]  # all (current subset, depth)
    best_set = given_input_set

    while !isempty(stack)
        current_set, steps = pop!(stack)

        if steps >= max_steps || in(current_set, visited)
            continue
        end

        push!(visited, current_set) # mark the current subset as visited
        status, _ = adversarial(nn, input, output, current_set)
        println("TEST ON:", length(current_set), " status: ", status)

        if status == :success
            println("stop searching this branch")
            continue
        else
            if length(current_set) < length(best_set)
                best_set = current_set
            end
        end

        if length(current_set) <= 770  # too long
            break
        end

        for i in 1:length(current_set)
            next_set = setdiff(current_set, [current_set[i]])  # remove one element

            if length(next_set) <= 0
                continue
            end

            push!(stack, (next_set, steps + 1))
        end
    end

    return best_set
end

function minimal_set_dfs(nn::Chain, input, output)
    given_input_set = collect(1:length(input))
    tmp_inputs = collect(1:4)
    println(tmp_inputs)
    global visited = Set{Vector{Int}}()
    # result = dfs_cache(nn, given_input_set, input, output, 0, 100, Ref(false))
    result = dfs_cache_non_recursive(nn, given_input_set, input, output, 100)
    # dfs(nn, given_input_set, 1, input, output)
    return result
end
