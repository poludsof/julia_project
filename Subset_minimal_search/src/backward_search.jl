
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

# global visited = Set{Vector{Int}}()

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

function dfs_cache_non_recursive(nn::Chain, given_input_set::SBitSet{N,T}, input::AbstractVector{<:Integer}, output, max_steps::Int) where {N, T}
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
            @timeit to "random" adv_founded = check_random_sets(nn, input, output, current_set)
            println("Random search result: ", adv_founded)
            if adv_founded
                # println("Random set FOUND")
                continue
            end
            # if length(current_set) <= 750 && adv_founded == false
            #     println("returning current set due to adv_founded == false")
            #     return current_set
            # end
        end

        if length(current_set) <= 710  # too long
            break
        end

        @timeit to "milp" status, _ = adversarial(nn, input, output, current_set)
        # status = "fff"
        println("TEST ON:", length(current_set), " status: ", status)

        if status == :success
            println("stop searching this branch")
            continue
        end
            
        if length(current_set) < length(best_set)
            best_set = current_set
        end

        indx = 0
        for i in current_set
            next_set = current_set ~ SBitSet{N,T}(i)
            if length(next_set) <= 0 || in(next_set, stack) || in(next_set, visited_local)
                continue
            end
            indx += 1
            if indx > 1
                break
            end
            @timeit to "push!" push!(stack, (next_set, steps + 1))
        end
    end
    return best_set
end

function test_new_dfs(nn, given_input_set::SBitSet{N,T}, input, output) where {N, T}
    for i in given_input_set
        println("i: ", i)
        if length(given_input_set) <= 700
            break
        end
        
        status = check_random_sets(nn, input, output, given_input_set)
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


function find_minimal_subset_forward(nn::Chain, given_input_set::SBitSet{N,T}, image::AbstractVector{<:Integer}, target_digit::Int) where {N, T}
    subset = SBitSet{N, T}()
    candidate_subset = SBitSet{N, T}()

    for i in given_input_set
        candidate_subset = candidate_subset ∪ SBitSet{N, T}(i)

        status = check_random_sets(nn, image, target_digit, candidate_subset)
        if status == false
            return candidate_subset
        end

        println("Candidate subset: ", length(candidate_subset), " status: ", status)
    end
    
    return subset
end

function random_img(input::AbstractVector{<:Integer}, current_set::SBitSet{N,T}) where {N, T}
    random_img = rand(0:1, length(input))
    random_img = [idx in current_set ? input[idx] : random_img[idx] for idx in 1:length(input)]
    return random_img
end


function generate_unique_random_sets(input::AbstractVector{<:Integer}, current_set::SBitSet{N,T}, num_sets::Int) where {N, T}
    unique_sets = Set{Vector{Int}}()
    
    while length(unique_sets) < num_sets
        new_set = random_img(input, current_set)
        push!(unique_sets, new_set)
    end
    
    return collect(unique_sets)
end


function check_random_sets(nn::Chain, input::AbstractVector{<:Integer}, output, current_set::SBitSet{N,T}) where {N, T}
    number_sets = 2 ^ (length(input) - length(current_set))
    number_sets = (number_sets > 1000 || number_sets == 0) ? 1000 : number_sets
    println("Number of sets: ", number_sets)
    unique_sets = generate_unique_random_sets(input, current_set, number_sets)
    for img in unique_sets
        # println("Random set: ", img)
        result = argmax(nn(img)) - 1
        # println("Result: ", result)
        if result != output
            # println("Random set found adversarial example")
            return true
        end
    end
    # array = ones(Int, length(input))
    # array = [idx in current_set ? input[idx] : array[idx] for idx in 1:length(input)]
    # if (argmax(nn(array)) - 1) != output
    #     return true
    # end
    # println("Random set not found adversarial example")
    return false
end


function minimal_set_dfs(nn::Chain, input::AbstractVector{<:Integer}, output)
    given_input_set = collect(1:length(input))
    tmp_inputs = collect(1:4)

    input_set = SBitSet{32, UInt32}(given_input_set)
    # result = dfs_cache(nn, given_input_set, input, output, 0, 100, Ref(false))
    # result = dfs_cache_non_recursive(nn, input_set, input, output, 100)
    # dfs(nn, given_input_set, 1, input, output)
    # result = test_new_dfs(nn, input_set, input, output)
    result = find_minimal_subset_forward(nn, input_set, input, output)
    return result
end
