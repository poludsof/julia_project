

## TODO try this notation:
# function (h::Heuristic)(fix_inputs)
#     .....
#     fix_inputs
# end


""" no longer in use """
function dfs(nn::Chain, fix_inputs::Vector{Int}, start::Int, input::AbstractVector{<:Integer}, output::Int)
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


""" problem with found_minimal_set ? useless? """ # TODO
# global visited = Set{Vector{Int}}()
function dfs_cache(bs::Backward_search, given_input_set::Vector{Int}, steps::Int, max_steps::Int, found_minimal_set::Bool)
    if steps >= max_steps || found_minimal_set || in(given_input_set, visited)
        return given_input_set
    end

    push!(visited, given_input_set)

    status, _ = adversarial(bs.nn, bs.input, bs.output, given_input_set)
    println("TEST ON:", length(given_input_set), " status: ", status)
    
    if status == :success
        println("stop searching")
        return given_input_set
    end
    
    # if length(given_input_set) <= 743 # too long; TODO delete later 
    #     found_minimal_set = true
    #     return given_input_set
    # end

    # set is infeasible -> save as best set
    best_set = given_input_set

    for i in 1:length(given_input_set) 
        next_set = setdiff(given_input_set, [given_input_set[i]])
        new_fix_inputs = dfs_cache(bs, next_set, steps + 1, max_steps, found_minimal_set)
        
        if found_minimal_set
            return new_fix_inputs
        end

        if length(new_fix_inputs) < length(best_set)
            best_set = new_fix_inputs
        end
    end

    return best_set
end


""" Function using a non-recursive method, but with a stack and an array of visited sets """
function dfs_cache_non_recursive(bs::Backward_search, given_input_set::SBitSet{N,T}, max_steps::Int) where {N, T}
    visited_local = Set{SBitSet{N,T}}()
    stack = [(given_input_set, 0)]  # all (current subset, depth)
    best_set = given_input_set

    while !isempty(stack)
        @timeit to "pop!" current_set, steps = pop!(stack)

        if steps >= max_steps || in(current_set, visited_local)
            if steps >= max_steps
                println("Max steps reached")
            end
            continue
        end

        push!(visited_local, current_set) # mark the current subset as visited

        # samplovani
        if length(current_set) <= 760
            @timeit to "random" adv_founded = check_random_sets(bs, current_set)
            println("Random search result: ", adv_founded)
            if adv_founded
                # println("Random set FOUND")
                continue
            end
        end

        # if length(current_set) <= 710  # too long
        #     break
        # end

        @timeit to "milp" status, _ = adversarial(bs.nn, bs.input, bs.output, current_set)
        println("TEST ON:", length(current_set), " status: ", status)

        if status == :success
            println("stop searching this branch")
            continue
        end
            
        if length(current_set) < length(best_set)
            best_set = current_set
        end

        for i in current_set
            next_set = current_set ~ SBitSet{N,T}(i)
            if length(next_set) <= 0 || in(next_set, stack) || in(next_set, visited_local)
                continue
            end
            @timeit to "push!" push!(stack, (next_set, steps + 1))
        end
    end
    return best_set
end


""" Simple function that removes features without going back (not dfs) """
function tmp_backward(bs::Backward_search, given_input_set::SBitSet{N,T}) where {N, T}
    for i in given_input_set
        # println("i: ", i)
        if length(given_input_set) <= 700 # too long; TODO delete later
            break
        end
        
        status = check_random_sets(bs, given_input_set)
        # if status == false
        #     status, _ = adversarial(nn, input, output, given_input_set)
        # end

        println("length: ", length(given_input_set), " status: ", status)
        if status == false
            given_input_set = given_input_set ~ SBitSet{N,T}(i)    
            println("Deleted: ", i)    
        end
    end
    return given_input_set
end


function random_img(input::AbstractVector{<:Integer}, current_set::SBitSet{N,T}) where {N, T}
    return map(idx -> idx in current_set ? input[idx] : rand(0:1), 1:length(input))
end

function generate_unique_random_img_sets(input::AbstractVector{<:Integer}, current_set::SBitSet{N,T}, num_sets::Int) where {N, T}
    unique_sets = Set{Vector{Int}}()
    
    while length(unique_sets) < num_sets
        new_set = random_img(input, current_set)
        push!(unique_sets, new_set)
    end
    
    return collect(unique_sets)
end

function check_random_sets(bs::Backward_search, current_set::SBitSet{N,T}) where {N, T}
    number_sets = 2 ^ (length(bs.input) - length(current_set))
    number_sets = (number_sets > 1000 || number_sets == 0) ? 1000 : number_sets
    # println("Number of sets: ", number_sets)
    unique_sets = generate_unique_random_img_sets(bs.input, current_set, number_sets)

    ## Try to replace this:
    for img in unique_sets
        if argmax(nn(img))-1 != output
            return true
        end
    end

    ## with this:
    any(argmax(nn(img))-1 != bs.output for img in unique_sets)

    return false
end

function minimal_set_search(bs::Backward_search)
    given_input_set = collect(1:length(bs.input))
    tmp_inputs = collect(1:4)
    input_set = SBitSet{32, UInt32}(given_input_set)
    
    # result = dfs_cache(bs, given_input_set, 0, 100, false)
    # result = dfs_cache_non_recursive(bs, input_set, 100)
    result = dfs(bs.nn, given_input_set, 1, bs.input, bs.output)
    # result = tmp_backward(bs, input_set)
    return result
end
