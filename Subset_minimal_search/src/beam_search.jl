function beam_search(sm::Subset_minimal, ii::TT, isvalid::Function, heuristic_fun;  beam_size=5, time_limit=Inf, terminate_on_first_solution=true) where {TT}
    # stack with heuristic, error, ii
    println("Beam search")
        initial_heuristic = heuristic_fun(ii)
    if typeof(initial_heuristic) == Tuple
        initial_h, initial_hmax_err = initial_heuristic
    else
        initial_h = initial_heuristic
    end
    println("Initial heuristic: ", initial_h)

    beam_set = [(initial_h, ii)]
    closed_list = Set{TT}()
    solutions = Set{TT}()
    beam_set = expand_beam(sm, beam_set, closed_list, heuristic_fun, beam_size)
    print_beam(beam_set, beam_size)

    println("Starting beam search...")

    start_time = time() 
    step = 0
    @timeit to "beam search" while true
        if time() - start_time > time_limit || beam_set[1][1] < 0
            println("Timeout exceeded, returning last found solutions")
            return solutions
        end

        v = @timeit to "isvalid" isvalid(beam_set[1][2])
        if v
            push!(solutions, beam_set[1][2])
            terminate_on_first_solution && return solutions
        end

        println("THE END of $step, best h: ", beam_set[1][1])
        println("step: $step, length $(solution_length(beam_set[1][2])) with heuristic: ", beam_set[1][1])
        step += 1

        beam_set = expand_beam(sm, beam_set, closed_list, heuristic_fun, beam_size)
    end
    return solutions
end

new_subsets_beam(ii::SBitSet, idim) = [push(ii, i) for i in setdiff(1:idim, ii)]

function new_subsets_beam((I3, I2, I1)::T, idims::Tuple) where {T<:Tuple}
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
        current_h, ii = pop!(beam_sets)

        for new_subset in new_subsets_beam(ii, sm.dims)
            if new_subset ∉ closed_list
                new_heuristic = @timeit to "heuristic" heuristic_fun(new_subset)
                if typeof(new_heuristic) == Tuple
                    new_heuristic, new_error = new_heuristic
                end
                push!(new_beam_set, (new_heuristic, new_subset))
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
        println("Set: ", s[2], " hsum: ", s[1])
    end
end