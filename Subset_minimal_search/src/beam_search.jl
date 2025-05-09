function beam_search(sm::Subset_minimal, ii::TT, isvalid::Function, heuristic_fun;  beam_size=5, time_limit=Inf, terminate_on_first_solution=true) where {TT}
    # stack with heuristic, error, ii
    println("Beam search")
    initial_heuristic, full_error = heuristic_fun(ii)
    println("Initial heuristic: ", initial_heuristic)
    beam_set = [(initial_heuristic, full_error, ii)]
    closed_list = Set{TT}()
    solutions = Set{TT}()
    beam_set = expand_beam(sm, beam_set, closed_list, heuristic_fun, beam_size)
    print_beam(beam_set, beam_size)

    println("Starting beam search...")

    start_time = time() 
    iter_count = 1
    @timeit to "beam search" while true
        if (time() - start_time > time_limit && beam_set[1][2] > 0.02) || beam_set[1][2] < 0
            println("Timeout exceeded, returning last found solutions")
            return iter_count, beam_set
        end

        v = @timeit to "isvalid" isvalid(beam_set[1][3])
        if v
            terminate_on_first_solution && return(iter_count, beam_set)
            push!(solutions, beam_set[1][3])
        end

        println("THE END of $iter_count, best error: ", beam_set[1][2])    

        beam_set = expand_beam(sm, beam_set, closed_list, heuristic_fun, beam_size)
        iter_count += 1
    end
end

new_subsets_beam(ii::SBitSet, idim) = [push(ii, i) for i in setdiff(1:idim, ii)]

function new_subsets_beam((I3, I2, I1)::T, idims::Tuple, heuristic_fun) where {T<:Tuple}
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
    println("new subsets")
    new_subsets
end

function expand_beam(sm::Subset_minimal, beam_sets, closed_list, heuristic_fun, beam_size)
    new_beam_set = []
    while !isempty(beam_sets)
        current_heuristic, current_error, ii = pop!(beam_sets)
        # println("Current subset: ", ii, " heuristic: ", current_heuristic, " error: ", current_error)

        for new_subset in new_subsets_beam(ii, sm.dims)
            if new_subset ∉ closed_list
                new_heuristic, new_error = @timeit to "heuristic"  heuristic_fun(new_subset)
                # println("New subset: ", new_subset, " heuristic: ", new_heuristic, " error: ", new_error)
                push!(new_beam_set, (new_heuristic, new_error, new_subset))
                if length(new_beam_set) > beam_size
                    sort!(new_beam_set, by=x->x[1]) # sort by heuristic
                    pop!(new_beam_set)
                end
            end
        end
    end
    new_beam_set
end


# Helper functions
function print_beam(sets, beam_size)
    println("Size of beam: ", beam_size)
    for s in sets
        println("Set: ", s[3], " heuristic: ", s[1], " error: ", s[2])
    end
end