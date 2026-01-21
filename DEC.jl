module DEC
    
    pythonpath = "/opt/homebrew/Caskroom/miniconda/base/envs/dec/bin/python3"
    if !isfile(pythonpath)
        throw(ErrorException("Python executable not found at $pythonpath"))
    else
        ENV["JULIA_CONDAPKG_BACKEND"] = "Current"
        ENV["JULIA_PYTHONCALL_EXE"] = "/opt/homebrew/Caskroom/miniconda/base/envs/dec/bin/python3"
    end

    using LinearAlgebra
    using PythonCall
    using Printf
    using JSON3

    qiskit_result = pyimport("qiskit.result")

    export SparseVecDict, SparseBlockCirculantMatrix
    export nnz, nonzero_keys, nonzero_values, nonzero_pairs

    function load_cal_samples(filepath_json::String)
        open(filepath_json, "r") do f
            return JSON3.read(f, Vector{Vector{Dict{String, Int}}})
        end
    end

    #============================================================#
    #          Sparse Vector Dictionary Implementation           #
    #============================================================#

    struct SparseVecDict{T<:Real} <: AbstractVector{T}
        data::Dict{BigInt, T}
        size::BigInt
        type::Type
        sortedkeys::Vector{BigInt}
        function SparseVecDict{T}(size::Int) where T
            if size < 1
                throw(ArgumentError("Size must be a positive integer"))
            end
            new{T}(Dict{BigInt, T}(), BigInt(size), T, Vector{BigInt}(), Vector{T}())
        end
        function SparseVecDict(data::Dict{BigInt, T}, size::BigInt) where T
            if any(k < 1 || k > size for k in keys(data))
                throw(ArgumentError("Keys must be in the range [1, size]"))
            end
            new{T}(data, size, T, sort(collect(keys(data))))
        end
        function SparseVecDict(v::AbstractArray{T}; atol::Real=1e-16) where T
            data = Dict{BigInt, T}()
            sortedkeys = zeros(BigInt, length(v))
            fini = 1
            for (i, val) in enumerate(v)
                if !isapprox(val, 0, atol=atol)
                    data[BigInt(i)] = val
                    sortedkeys[fini] = BigInt(i)
                    fini += 1
                end
            end
            return new{T}(data, BigInt(length(v)), T, sortedkeys[1:fini-1])
        end
        function SparseVecDict(v::SparseVecDict{T}; atol::Real=1e-16) where T
            nv = clean(v, atol=atol)
            return nv
        end
    end

    function nnz(v::SparseVecDict)
        return length(v.data)
    end

    function nonzero_keys(v::SparseVecDict)
        return v.sortedkeys
    end

    function nonzero_values(v::SparseVecDict)
        return collect(values(v.data))
    end

    function nonzero_pairs(v::SparseVecDict)
        return collect(v.data)
    end

    function clean(v::SparseVecDict; atol::Real=1e-16)
        newdata = Dict{BigInt, eltype(v)}()
        for (k, val) in v.data
            if !isapprox(val, 0, atol=atol)
                newdata[k] = val
            end
        end
        return SparseVecDict(newdata, v.size)
    end

    function Base.show(io::IO, ::MIME"text/plain", v::SparseVecDict)
        print(io, "SparseVecDict{$(v.type)}(size=$(v.size), nnz=$(length(v.data)))")
    end

    function Base.:*(v::SparseVecDict, n::Real; atol=1e-16)
        res_eltype = promote_type(eltype(v), typeof(n))
        if res_eltype == Any
            throw(ArgumentError("Could not promote element types"))
        end
        newdata = Dict{BigInt, res_eltype}()
        for (k, val) in v.data
            result = val * n
            if !isapprox(result, 0, atol=atol)
                newdata[k] = result
            end
        end
        return SparseVecDict(newdata, v.size)
    end

    function Base.:*(n::Real, v::SparseVecDict; atol=1e-16)
        return Base.:*(v, n; atol=atol)
    end

    function Base.:/(v::SparseVecDict, n::Real; atol=1e-16)
        if isapprox(n, 0, atol=atol)
            throw(DivideError())
        end
        newdata = Dict{BigInt, Float64}()
        for (k, val) in v.data
            result = val / n
            if !isapprox(result, 0, atol=atol)
                newdata[k] = result
            end
        end
        return SparseVecDict(newdata, v.size)
    end

    function Base.:+(v1::SparseVecDict, v2::SparseVecDict; atol=1e-16)
        if length(v1) != length(v2)
            throw(ArgumentError("Vector sizes must match"))
        end
        res_eltype = promote_type(eltype(v1), eltype(v2))
        if res_eltype == Any
            throw(ArgumentError("Could not promote element types"))
        end
        newdata = Dict{BigInt, res_eltype}()
        union_keys = union(nonzero_keys(v1), nonzero_keys(v2))
        for k in union_keys
            result = v1[k] + v2[k]
            if !isapprox(result, 0, atol=atol)
                newdata[k] = result
            end
        end
        return SparseVecDict(newdata, length(v1))
    end

    function Base.:+(v1::SparseVecDict, v2::AbstractArray{<:Real}; atol=1e-16)
        v2 = SparseVecDict(v2; atol=atol)
        return Base.:+(v1, v2; atol=atol)
    end

    function Base.:+(v1::AbstractArray{<:Real}, v2::SparseVecDict; atol=1e-16)
        return Base.:+(v2, v1; atol=atol)
    end

    function Base.:-(v1::SparseVecDict, v2::SparseVecDict; atol=1e-16)
        if length(v1) != length(v2)
            throw(ArgumentError("Vector sizes must match"))
        end
        res_eltype = promote_type(eltype(v1), eltype(v2))
        if res_eltype == Any
            throw(ArgumentError("Could not promote element types"))
        end
        newdata = Dict{BigInt, res_eltype}()
        union_keys = union(nonzero_keys(v1), nonzero_keys(v2))
        for k in union_keys
            result = v1[k] - v2[k]
            if !isapprox(result, 0, atol=atol)
                newdata[k] = result
            end
        end
        return SparseVecDict(newdata, length(v1))
    end

    function Base.:-(v1::SparseVecDict, v2::AbstractArray{<:Real}; atol=1e-16)
        v2 = SparseVecDict(v2; atol=atol)
        return Base.:-(v1, v2; atol=atol)
    end

    function Base.:-(v1::AbstractArray{<:Real}, v2::SparseVecDict; atol=1e-16)
        v1 = SparseVecDict(v1; atol=atol)
        return Base.:-(v1, v2; atol=atol)
    end

    function Base.:*(v1::SparseVecDict, v2::SparseVecDict; atol=1e-16)
        if length(v1) != length(v2)
            throw(ArgumentError("Vector sizes must match"))
        end
        v2 = SparseVecDict(v2; atol=atol)
        res_eltype = promote_type(eltype(v1), eltype(v2))
        if res_eltype == Any
            throw(ArgumentError("Could not promote element types"))
        end
        if nnz(v1) <= nnz(v2)
            sv = v1
            lv = v2
        else
            sv = v2
            lv = v1
        end
        newdata = Dict{BigInt, res_eltype}()
        for k in nonzero_keys(sv)
            if haskey(lv.data, k)
                result = sv[k] * lv[k]
                if !isapprox(result, 0, atol=atol)
                    newdata[k] = result
                end
            end
        end
        return SparseVecDict(newdata, v1.size)
    end

    function Base.:*(v1::SparseVecDict, v2::AbstractArray{<:Real}; atol=1e-16)
        v2 = SparseVecDict(v2; atol=atol)
        return Base.:*(v1, v2; atol=atol)
    end

    function Base.:*(v1::AbstractArray{<:Real}, v2::SparseVecDict; atol=1e-16)
        return Base.:*(v2, v1; atol=atol)
    end

    function Base.copy(v::SparseVecDict)
        return SparseVecDict(copy(v.data), v.size)
    end

    function Base.sum(v::SparseVecDict)
        s = zero(eltype(v))
        for val in nonzero_values(v)
            s += val
        end
        return s
    end

    function Base.abs(v::SparseVecDict)
        newdata = Dict{BigInt, eltype(v)}()
        for (k, val) in v.data
            newdata[k] = abs(val)
        end
        return SparseVecDict(newdata, v.size)
    end

    function LinearAlgebra.norm(v::SparseVecDict, p::Int=2)
        if p == 0
            return nnz(v)
        elseif p == 1
            norm = 0
            for val in nonzero_values(v)
                norm += abs(val)
            end
            return norm
        elseif p == Inf
            return maximum(abs.(nonzero_values(v)))
        elseif p < 0
            throw(ArgumentError("Unsupported norm type"))
        else
            norm = 0
            for val in nonzero_values(v)
                norm += abs(val)^p
            end
            return norm^(1/p)
        end
    end

    function counts_vector(cal_samples::Dict{String, Int})
        n = length(first(keys(cal_samples)))
        counts_vector_data = Dict{BigInt, Int}()
        for (bitstring, count) in cal_samples
            index = parse(Int, bitstring; base=2) + 1
            counts_vector_data[index] = count
        end
        return SparseVecDict(counts_vector_data, 2^BigInt(n))
    end

    function prob_vector(cal_samples::Dict{String, Int})
        counts_vec = counts_vector(cal_samples)
        total_counts = sum(counts_vec)
        prob_vector = counts_vec / total_counts
        return prob_vector
    end

    Base.size(v::SparseVecDict) = (v.size,)
    Base.length(v::SparseVecDict) = v.size
    Base.getindex(v::SparseVecDict, i::BigInt) = get(v.data, i, zero(eltype(v)))
    Base.getindex(v::SparseVecDict, i::Int) = get(v.data, BigInt(i), zero(eltype(v)))
    # Base.setindex!(v::SparseVecDict, val::T, i::BigInt) where T = (v.data[i] = val; v)
    # Base.setindex!(v::SparseVecDict, val::T, i::Int) where T = (v.data[BigInt(i)] = val; v)
    Base.eltype(v::SparseVecDict) = v.type

    #============================================================#
    #   Sparse Recursive-block-circulant Matrix Implementation   #
    #============================================================#

    struct SparseBlockCirculantMatrix{T} <: AbstractMatrix{T}
        col::SparseVecDict{T}
        dim::BigInt
        function SparseBlockCirculantMatrix(col::SparseVecDict{T}) where T
            if !isinteger(log2(length(col)))
                throw(ArgumentError("col must be of length a power of 2"))
            end
            new{T}(col, length(col))
        end
    end

    function Base.show(io::IO, ::MIME"text/plain", B::SparseBlockCirculantMatrix)
        print(io, "SparseBlockCirculantMatrix{$(eltype(B))}(size=$((B.dim, B.dim)), nnz_col=$(length(B.col.data)))")
    end

    function row_nonzero(B::SparseBlockCirculantMatrix, row::Union{Int, BigInt})
        row1_nonzeros = nonzero_keys(B.col)
        row_digits = digits(row - 1, base=2, pad=Int(log2(B.dim)))
        mask = sum(b*2^(i - 1) for (i, b) in enumerate(row_digits))
        this_row_nonzeros = [((x - 1) ⊻ mask) + 1 for x in row1_nonzeros]
        return this_row_nonzeros
    end

    function row_indices_to_first(B::SparseBlockCirculantMatrix, row::Union{Int, BigInt}, col::Union{Int, BigInt})
        row_digits = digits(row - 1, base=2, pad=Int(log2(B.dim)))
        mask = sum(b*2^(i - 1) for (i, b) in enumerate(row_digits))
        return ((col - 1) ⊻ mask) + 1
    end

    function row_indices_to_first(B::SparseBlockCirculantMatrix, row::Union{Int, BigInt}, cols::Vector{<:Union{Int, BigInt}})
        row_digits = digits(row - 1, base=2, pad=Int(log2(B.dim)))
        mask = sum(b*2^(i - 1) for (i, b) in enumerate(row_digits))
        return [((x - 1) ⊻ mask) + 1 for x in cols]
    end

    function Base.getindex(B::SparseBlockCirculantMatrix, i::Union{BigInt, Int}, j::Union{BigInt, Int})
        if i < 1 || i > B.dim || j < 1 || j > B.dim
            throw(ArgumentError("Indices out of bounds"))
        end
        j_first = row_indices_to_first(B, i, j)
        return get(B.col.data, j_first, zero(eltype(B)))
    end

    function vec_prod_conv(B::SparseBlockCirculantMatrix, v::SparseVecDict; atol::Real=1e-16)
        if length(v) != B.dim
            throw(ArgumentError("Vector length must match matrix dimension"))
        end
        res_eltype = promote_type(eltype(B), eltype(v))
        if res_eltype == Any
            throw(ArgumentError("Could not promote element types"))
        end
        newdata = Dict{BigInt, res_eltype}()
        for (i_b, v_b) in B.col.data
            for (i_v, v_v) in v.data
                resind = ((i_b-1) ⊻ (i_v-1)) + 1
                val = v_b * v_v
                if isapprox(val, 0, atol=atol)
                    continue
                else
                    newdata[resind] = get(newdata, resind, zero(res_eltype)) + val
                end
            end
        end
        result = SparseVecDict(newdata, B.dim)
        return clean(result; atol=atol)
    end

    function vec_prod_multithreaded(B::SparseBlockCirculantMatrix, v::SparseVecDict; atol::Real=1e-16)
        if length(v) != B.dim
            throw(ArgumentError("Vector length must match matrix dimension"))
        end
        res_eltype = promote_type(eltype(B), eltype(v))
        if res_eltype == Any
            throw(ArgumentError("Could not promote element types"))
        end
        nonzero_B_keys = nonzero_keys(B.col)
        nonzero_v_keys = nonzero_keys(v)
        t_results = [Dict{BigInt, res_eltype}() for _ in 1:Threads.nthreads()]
        Threads.@threads for i_b in nonzero_B_keys
            for i_v in nonzero_v_keys
                resind = ((i_b-1) ⊻ (i_v-1)) + 1
                val = v.data[i_v] * B.col.data[i_b]
                if isapprox(val, 0, atol=atol)
                    continue
                else
                    t_results[Threads.threadid()-1][resind] = get(t_results[Threads.threadid()-1], resind, zero(res_eltype)) + val
                end
            end
        end
        result = mergewith(+, t_results...)
        return clean(SparseVecDict(result, B.dim); atol=atol)
    end

    Base.eltype(B::SparseBlockCirculantMatrix) = eltype(B.col)
    Base.size(B::SparseBlockCirculantMatrix) = (B.dim, B.dim)
    Base.:*(B::SparseBlockCirculantMatrix, v::SparseVecDict; atol=1e-16) = vec_prod_multithreaded(B, v; atol=atol)
    Base.transpose(B::SparseBlockCirculantMatrix) = B  # Symmetric
    Base.adjoint(B::SparseBlockCirculantMatrix) = B  # Symmetric

    #============================================================#
    #               AMP algorithm implementation                 #
    #                     arXiv:1612.01183                       #
    #============================================================#

    struct AmpResult
        sol::AbstractArray{<:Real}
        coeffvec::AbstractArray{<:Real}
        converged::Bool
        iters::Int
        diffs::AbstractArray{<:Real}
    end

    function Base.show(io::IO, a::AmpResult)
        println(io, "AMP Result:")
        println(io, "  Converged: ", a.converged)
        println(io, "  Iterations: ", a.iters)
        println(io, @sprintf("  Last difference: %.3e", a.diffs[end]))
        println(io, "  Solution: ", a.sol)
    end

    function amp_cost_function(A::SparseBlockCirculantMatrix, x::SparseVecDict, y::SparseVecDict, alpha::Real)
        return norm(y - A*x, 2)^2/2 + alpha * norm(x, 0)
    end

    function amp_cost_function(yminusAx::SparseVecDict, x::SparseVecDict, alpha::Real)
        return norm(yminusAx, 2)^2/2 + alpha * norm(x, 0)
    end

    function soft_thresholding(v::SparseVecDict, λ::Real)
        newdata = Dict{BigInt, eltype(v)}()
        for (k, x) in v.data
            if abs(x) > λ
                newdata[k] = sign(x) * max(abs(x) - λ, 0)
            end
        end
        return SparseVecDict(newdata, length(v))
    end

    function invamp(A::SparseBlockCirculantMatrix, y::SparseVecDict; alpha::Real=1, normatol::Real=1e-8, max_iter::Int=1000)
        res_eltype = promote_type(eltype(A), eltype(y))
        if res_eltype == Any
            throw(ArgumentError("Could not promote element types"))
        end
        if length(y) != size(A, 1)
            throw(ArgumentError("Vector length must match matrix dimension"))
        end
        xt = SparseVecDict(Dict{BigInt, res_eltype}(), A.dim)
        vt = SparseVecDict(Dict{BigInt, res_eltype}(), A.dim)
        y = copy(y) / norm(y, 2)
        normdiff = Inf
        itercount = 1
        normdiffs = zeros(max_iter)
        yminusAxt = y
        while normdiff > normatol && itercount <= max_iter
            prev_norm = amp_cost_function(yminusAxt, xt, alpha)
            beta = norm(xt, 0) / Int(length(y)) #TODO: REDEFINE TO REMOVE BIG INT
            lambda = alpha * norm(vt, 2) / sqrt(Int(length(y))) #TODO: REDEFINE TO REMOVE BIG INT
            vt = yminusAxt + beta * vt
            xt = soft_thresholding(xt + (A' * vt), lambda)
            yminusAxt = y - A * xt
            normdiff = abs(amp_cost_function(yminusAxt, xt, alpha) - prev_norm)
            normdiffs[itercount] = normdiff
            itercount += 1
        end
        return AmpResult(xt, y, itercount < max_iter, itercount-1, normdiffs[1:itercount-1])
    end

    function invamp_atol(A::SparseBlockCirculantMatrix, y::SparseVecDict; alpha::Real=1, normatol::Real=1e-8, max_iter::Int=1000)
        res_eltype = promote_type(eltype(A), eltype(y))
        if res_eltype == Any
            throw(ArgumentError("Could not promote element types"))
        end
        if length(y) != size(A, 1)
            throw(ArgumentError("Vector length must match matrix dimension"))
        end
        xt = SparseVecDict(Dict{BigInt, res_eltype}(), A.dim)
        vt = SparseVecDict(Dict{BigInt, res_eltype}(), A.dim)
        y = copy(y) / norm(y, 2)
        normdiff = 10*normatol
        itercount = 1
        normdiffs = zeros(max_iter)
        yminusAxt = y
        while normdiff > normatol && itercount <= max_iter
            prev_norm = amp_cost_function(yminusAxt, xt, alpha)
            beta = norm(xt, 0) / Int(length(y)) #TODO: REDEFINE TO REMOVE BIG INT
            lambda = alpha * norm(vt, 2) / sqrt(Int(length(y))) #TODO: REDEFINE TO REMOVE BIG INT
            vt = +(yminusAxt, *(vt, beta, atol=normatol), atol=normatol) # vt = y - A*xt + beta * vt
            Adagtimesv = *(A, vt, atol=normatol) # A = A'
            xt = soft_thresholding(+(xt, Adagtimesv, atol=normatol), lambda)
            Atimest = *(A, xt, atol=normatol)
            yminusAxt = -(y, Atimest, atol=normatol)
            normdiff = abs(amp_cost_function(yminusAxt, xt, alpha) - prev_norm)
            normdiffs[itercount] = normdiff
            itercount += 1
        end
        return AmpResult(xt, y, itercount < max_iter, itercount-1, normdiffs[1:itercount-1])
    end

    function noise_matrix(c_vector::SparseVecDict{Int})
        total_counts = sum(c_vector)
        prob_vector = c_vector / total_counts
        return SparseBlockCirculantMatrix(prob_vector)
    end

    function countvec_to_dist(p_vector::SparseVecDict{<:Real})
        n = Int(log2(length(p_vector))) 
        if !isinteger(n)
            throw(ArgumentError("Count vector length must be a power of 2."))
        end
        qprob_dist = Dict(join(reverse(digits(i-1, base=2, pad=n))) => p for (i, p) in nonzero_pairs(p_vector))
        prob_dist = qiskit_result.QuasiDistribution(qprob_dist).nearest_probability_distribution().binary_probabilities(n)
        prob_dist = pyconvert(Dict{String, Float64}, prob_dist)
        return Dict(k => v / sum(values(prob_dist)) for (k, v) in prob_dist)
    end

    function get_noiseless_countvec(noisy_cv::SparseVecDict{Int}, cal_cv::SparseVecDict{Int}; alpha::Real=0.1, normatol::Real=1e-8, max_iter::Int=1000, unsafe::Bool=false)
        dim = length(noisy_cv)
        if length(cal_cv) != dim
            throw(ArgumentError("Noisy and calibration count vectors must have the same length."))
        end
        A = noise_matrix(cal_cv)
        noisy_pv = noisy_cv / sum(noisy_cv)
        sol = invamp(A, noisy_pv; alpha=alpha, normatol=normatol, max_iter=max_iter)
        if sol.converged
            return sol.sol / sum(abs(sol.sol))
        elseif !sol.converged && unsafe
            @warn "AMP did not converge, returning last iteration"
            return sol.sol / sum(abs(sol.sol))
        else
            throw(ErrorException("AMP did not converge"))
        end
    end

    function quasiprobability_dict(jobs_samples::Dict{String, Int}, cal_samples::Dict{String, Int}; alpha::Real=1, normatol::Real=1e-8, max_iter::Int=1000, unsafe::Bool=false)
        snoisy_count_vectors = counts_vector(jobs_samples)
        scal_count_vectors = counts_vector(cal_samples)
        clean_count_vector = get_noiseless_countvec(snoisy_count_vectors, scal_count_vectors; alpha=alpha, normatol=normatol, max_iter=max_iter, unsafe=unsafe)
        qprob_dist = Dict(join(reverse(digits(i-1, base=2, pad=Int(log2(length(clean_count_vector)))))) => p for (i, p) in non_zero_pairs(clean_count_vector))
        return qprob_dist
    end

    function quasiprobability_dict(jobs_samples::Vector{Dict{String, Int}}, cal_samples::Vector{Dict{String, Int}}; alpha::Union{Real, Vector{<:Real}}=1, normatol=1e-8, max_iter::Int=1000, unsafe::Bool=false)
        if length(jobs_samples) != length(cal_samples)
            throw(ArgumentError("Jobs samples and calibration samples must have the same length."))
        end
        if isa(alpha, Real)
            alpha = fill(alpha, length(jobs_samples))
        elseif length(alpha) != length(jobs_samples)
            throw(ArgumentError("Alpha vector length must match number of jobs samples."))
        end
        qprob_dists = [quasiprobability_dict(js, cs; alpha=a, normatol=normatol, max_iter=max_iter, unsafe=unsafe) for (js, cs, a) in zip(jobs_samples, cal_samples, alpha)]
        return qprob_dists
    end
end