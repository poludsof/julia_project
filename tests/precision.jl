using CSV
using DataFrames

# subset_data = CSV.read("/home/sofia/julia_project/Subset_minimal_search/tests/beam_search_uniform_all", DataFrame)
# subset_data = CSV.read("/home/sofia/julia_project/Subset_minimal_search/tests/bcwd_search_uniform_all", DataFrame)
# subset_data = CSV.read("/home/sofia/julia_project/Subset_minimal_search/tests/forward_search_uniform_all", DataFrame)
subset_data = CSV.read("/home/sofia/julia_project/forward_search_bernl_all", DataFrame)
# subset_data = CSV.read("/home/sofia/julia_project/tuned_forward_search_uniform_all", DataFrame)
# subset_data = CSV.read("/home/sofia/julia_project/ep_tuned_forward_search_uniform_09_1000", DataFrame)

function parse_solution(solution::String)
    return Set(parse.(Int, split(strip(solution, ['"']), ",")))
end

function calculate_coverage(img, solution_set)
    x = train_X_bin_neg[:, img]
    y = argmax(model(x))
    coverage_img = 0
    index = 1
    success = 0
    while coverage_img < 1000
        if index == img
            index += 1
            continue
        end
        xᵢ = train_X_bin_neg[:, index]
        yᵢ = argmax(model(x))
        if yᵢ == y
            coverage_img += 1
            match = 0
            for i in solution_set
                if x[i] == xᵢ[i]
                    match += 1
                end
            end
            if match == length(solution_set)
                success += 1
            end
        end
        index += 1
    end
    return success / 1000
end

function calculate_precision(img, solution_set)
    x = train_X_bin_neg[:, img]
    y = argmax(model(x))
    img_i = 0
    match_i = 1
    success = 0
    while match_i <= 100 && img_i < 60000
        img_i += 1
        if img_i == img
            continue
        end
        xᵢ = train_X_bin_neg[:, img_i]
        yᵢ = argmax(model(xᵢ))

        match = sum(xᵢ[i] == x[i] for i in solution_set)
        if match == length(solution_set)
            match_i += 1
            success += (yᵢ == y)
        end
    end
    return success / match_i
end

results = DataFrame(
    image=Int[],
    epsilon=Float64[],
    num_samples=Int[],
    coverage=Float64[],
    precision=Float64[]
)


function average_coverage(subset_data, epsilon, num_samples)
    condition = (subset_data.ϵ .== epsilon) .& (subset_data.num_samples .== num_samples)
    subset = subset_data[condition, :]
    coverage_list = Float64[]
    for row in eachrow(subset)
        image_idx = row.image
        solution_set = parse_solution(row.solution)
        coverage = calculate_coverage(image_idx, solution_set)
        push!(coverage_list, coverage)
    end
    coverage_list = coverage_list[coverage_list .> 0]
    return mean(coverage_list)
end

function average_precision(subset_data, epsilon, num_samples)
    condition = (subset_data.ϵ .== epsilon) .& (subset_data.num_samples .== num_samples)
    subset = subset_data[condition, :]
    precision_list = Float64[]
    # i = 0
    for row in eachrow(subset)
        # println("Row: ", i)
        # i += 1
        image_idx = row.image
        solution_set = parse_solution(row.solution)
        precision = calculate_precision(image_idx, solution_set)
        push!(precision_list, precision)
    end
    precision_list = precision_list[precision_list .> 0]
    return mean(precision_list) * 100
end

# target_row = subset_data[(subset_data.ϵ .== 0.99) .& (subset_data.num_samples .== 1000), :][2, :]
# target_image_idx = target_row.image
# target_solution = parse_solution(target_row.solution)

# coverage = calculate_coverage(target_image_idx, target_solution)

coverage = average_coverage(subset_data, 0.9, 1000)
println("Coverage: ", coverage)
coverage = average_coverage(subset_data, 0.99, 1000)
println("Coverage: ", coverage)
coverage = average_coverage(subset_data, 0.9, 10000)
println("Coverage: ", coverage)
coverage = average_coverage(subset_data, 0.99, 10000)
println("Coverage: ", coverage)


precision = average_precision(subset_data, 0.9, 1000)
println("Precision: ", precision)
precision = average_precision(subset_data, 0.99, 1000)
println("Precision: ", precision)
precision = average_precision(subset_data, 0.9, 10000)
println("Precision: ", precision)
precision = average_precision(subset_data, 0.99, 10000)
println("Precision: ", precision)
