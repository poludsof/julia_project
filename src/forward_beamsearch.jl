using ProbAbEx: Subset_minimal, expand_frwd, solution_length

function forward_beamsearch(sm::Subset_minimal, ii::TT, isvalid::Function, heuristic_fun; time_limit=Inf, beam_size = 100, branch_size = 10, terminate_on_first_solution=true, exclude_supersets = true, refine_with_backward=true) where {TT}
    initial_heuristic = heuristic_fun(ii)
    println("Initial heuristic: ", initial_heuristic)
    beam = [(initial_heuristic, ii)]
    closed_list = Set{TT}()
    solutions = Set{TT}()

    steps = 0
    start_time = time()
    smallest_solution = typemax(Int)

    @timeit to "forward search" while !isempty(beam)
        # steps > max_steps && break
        steps += 1
        if time() - start_time > time_limit
            println("Timeout exceeded, returning last found solutions")
            return solutions
        end

        # expand the beam while taking branch_size of best solutions selected by heuristic
        new_beam = map(beam) do (initial_heuristic, ii)
            push!(closed_list, ii)
            expanded_beam = Vector{eltype(beam)}()
            expanded_beam = @timeit to "expand_frwd" expand_frwd(sm, expanded_beam, closed_list, ii, heuristic_fun)
            n = max(branch_size, beam_size ÷ length(beam))
            expanded_beam[sortperm(first.(expanded_beam), rev = false)[1:n]]
        end
        new_beam = reduce(vcat, new_beam)
        new_beam = unique(last, new_beam)

        # then put valid solutions to solutions and 
        # the rest to beam, limiting the total count to beam_size
        # scores = map(isvalid ∘ last, new_beam)
        # @show argmin(isvalid ∘ last, new_beam)
        # @show (minimum(scores), maximum(scores))
        valid_beam = filter(<(0) ∘ isvalid ∘ last, new_beam)
        if !isempty(valid_beam)
            for (_, vii) in valid_beam
                if refine_with_backward
                    la = length(vii)
                    vii = @timeit to "backward_search"  backward_search(sm, vii, ii -> isvalid(ii) < 0, heuristic_fun, time_limit=200, terminate_on_first_solution=true)
                    la > length(vii) && println("solution refined by backward_search: ", la, "->", length(vii))
                end
                push!(solutions, vii)
                beam_size -= 1
            end
            beam_size ≤ 0 && break
            terminate_on_first_solution && return(solutions)
        end

        # remove valid solutions from the beam
        new_beam = setdiff(new_beam, valid_beam)
        isempty(new_beam) && break
        new_beam = new_beam[sortperm(first.(new_beam))]
        beam = length(new_beam) > beam_size ? new_beam[1:beam_size] : new_beam
        println("size of the beam size ", length(beam), "  best / worst in beam: ", minimum(first, beam), "/", maximum(first, beam))
    end
    return solutions
end
