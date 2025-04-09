function beam_search(sm::Subset_minimal, ii::TT, isvalid::Function, heuristic_fun;  beam_size=5, time_limit=Inf, terminate_on_first_solution=true) where {TT}
    # stack with heuristic, error, ii
    initial_heuristic, full_error = heuristic_fun(ii)
    beam_sets = [initial_heuristic, full_error, ii]
    closed_list = Set{TT}()
    solutions = Set{TT}()
    beam_set = expand_beam(sm, beam_set, closed_list, ii, heuristic_fun, beam_size)
    print_beam(beam_set)

    start_time = time()
    iter_count = 1
    @timeit to "beam search" while !isvalid(beam_set[1][3])
        if time() - start_time > time_limit
            println("Timeout exceeded, returning last found solutions")
            return beam_set
        end

        v = @timeit to "isvalid" isvalid(beam_set[1][3])
        if v
            terminate_on_first_solution && return(beam_set)
            push!(solutions, beam_set[1][3])
        end

        println("THE END of $iter_count, best score: ", best_sets[1][2])    

        beam_set = expand_beam(sm, beam_set, closed_list, ii, heuristic_fun, beam_size)
        iter_count += 1
    end
end

solution_length(ii::Tuple) = solution_length.(ii)
solution_length(ii::SBitSet) = length(ii)
solution_length(::Nothing) = 0

new_subsets(ii::SBitSet, idim) = [push(ii, i) for i in setdiff(1:idim, ii)]

function new_subsets((I3, I2, I1)::T, idims::Tuple, heuristic_fun) where {T<:Tuple}
    new_subsets = T[]
    if I3 !== nothing
        append!(new_subsets, [(push(I3, i), I2, I1) for i in setdiff(1:idims[1], I3)])
    end
    if I2 !== nothing
        append!(new_subsets, [(I3, push(I2, i), I1) for i in setdiff(1:idims[2], I2)])
    end
    if I1 !== nothing
        append!(new_subsets, [(I3, I2, push(I1, i)) for i in setdiff(1:idims[3], I1)])
    end
    new_subsets
end

function expand_beam(sm::Subset_minimal, beam_sets, closed_list, ii, heuristic, beam_size)
    new_beam_set = []
    for new_subset in new_subsets(ii, sm.dims)
        if new_subset ∉ closed_list
            new_heuristic, new_error = @timeit to "heuristic"  heuristic_fun(new_subset)
            push!(new_beam_set, (new_heuristic, new_error, new_subset))
            if length(new_beam_set) > beam_size
                sort!(new_beam_set, by=x->x[1]) # sort by heuristic
                pop!(new_beam_set)
            end
        end
    end
    new_beam_set
end  


# Helper functions
function print_beam(sets)
    println("Size of beam: ", length(sets))
    for s in sets
        println("Set: ", s[1], " Score: ", s[2])
    end
end
