
function adversarial(nn::Chain, input::AbstractVector{<:Integer}, output, fix_inputs=Int[]; optimizer = HiGHS.Optimizer, objective = :satisfy, paranoid = false, verbose = false, kwargs...)
	
	mathopt_model = Model(HiGHS.Optimizer)

	if !verbose
		set_silent(mathopt_model)
	end

	ivars = @variable(mathopt_model, [1:length(input)], Bin) # z vector
	ovars = setup_layer!(mathopt_model, nn, ivars) 

	for i in fix_inputs
		@constraint(mathopt_model, ivars[i] == input[i]) # first condition (zᵢ = xᵢ)
	end 
	
	setup_output_constraints!(mathopt_model, ovars, output + 1) # correct_class(output) is an index of correct class

	@objective(mathopt_model, Min, 0)

	optimize!(mathopt_model)

	status = JuMP.termination_status(mathopt_model)
	# write_to_file(mathopt_model, "/tmp/model.lp")
	status == JuMP.NO_SOLUTION && return(:infeasible, input)
	status == JuMP.INFEASIBLE && return(:infeasible, input)
	x = value.(ivars)
	y = value.(ovars)
	println("actual correct class: ", output)
	println("optimal correct class: ", argmax(y) - 1)
	# println("optimal correct class: ", argmax(input))
	x = [xᵢ > 0.5 ? 1 : -1 for xᵢ in x]
	(:success, x) 
end

function setup_layer!(mathopt_model, nn::Chain, ivars)
	ovars = setup_layer!(mathopt_model, nn[1], ivars)
	@show length(ovars)
	for i in 2:length(nn)
		ovars = setup_layer!(mathopt_model, nn[i], ovars)
	end
	ovars
end

function setup_layer!(mathopt_model, layer::Dense{typeof(identity)}, ivars)
	odim = size(layer.weight, 1)
	ovars = @variable(mathopt_model, [1:odim])
	for i in 1:odim
		wᵢ = layer.weight[i,:]
		bᵢ = layer.bias[i]
		@constraint(mathopt_model, dot(ivars, wᵢ) + bᵢ == ovars[i])
	end
	ovars
end

function setup_layer!(mathopt_model, layer::Dense{typeof(relu)}, ivars)
	odim = size(layer.weight, 1)
	y_vars = @variable(mathopt_model, [1:odim])
	s_vars = @variable(mathopt_model, [1:odim])
	z_vars = @variable(mathopt_model, [1:odim], Bin)
	@constraint(mathopt_model, s_vars .>= 0)
	@constraint(mathopt_model, y_vars .>= 0)
	M = 1e6
	for i in 1:odim
		wᵢ = layer.weight[i,:]
		bᵢ = layer.bias[i]
		@constraint(mathopt_model, dot(ivars, wᵢ) + bᵢ == y_vars[i] - s_vars[i])
		@constraint(mathopt_model, y_vars[i] ≤ M * (1-z_vars[i]))
		@constraint(mathopt_model, s_vars[i] ≤ M * z_vars[i])
	end
	y_vars
end

"""
	(∃i ≢ correct_class)(ovars[i] > ovars_correct_class)
"""
function setup_output_constraints!(mathopt_model, ovars, correct_class::Int) # correct_class is an index of correct class
	ii = setdiff(1:length(ovars), correct_class) # array of indexes of all classes except correct_class
	z_vars = @variable(mathopt_model, [1:length(ii)], Bin)
	M = 1e6
	oⱼ = ovars[correct_class]
	for (zᵢ, oᵢ) in zip(z_vars, ovars[ii])
		@constraint(mathopt_model, zᵢ * M ≥ oᵢ - oⱼ)
		@constraint(mathopt_model, oᵢ - oⱼ ≥  - (1 - zᵢ) * M)
	end
	@constraint(mathopt_model, sum(z_vars) ≥ 1)
end

function set_adversarial_objective!(mathopt_model, input::AbstractVector{<:Integer}, ivars, objective)
	if objective == :satisfy
		@objective(mathopt_model, Min, 0)
	elseif objective == :minimal
		w = [v > 0 ? -1 : 1 for v in input]
		@objective(mathopt_model, Min, dot(w, ivars))
	else
		error("unknown objective option, $(objective), allowed: satisfy, minimal")
	end
end
