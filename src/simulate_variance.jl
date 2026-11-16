#!/usr/bin/env julia

using PauliPropagation
using LinearAlgebra
using Printf
using Random
using Statistics
using CSV
using DataFrames
using TOML
using Base.Threads

# Selectable-initial-operator production variant.
# INITIAL_OPERATOR_MODE=random_pauli (default) or center_z (site L ÷ 2).
# One independent seed_P per realization; the same initial Pauli is reused across pT.
# The all-identity string is included, with probability 4^(-L).
# OBSERVABLE_SITE is unused and is not read.
# Version 2 optimization notes:
#   1) build each random Clifford realization once, not once per pT;
#   2) draw the T uniform table once and threshold it for all pT values;
#   3) parallelize independent realizations with Threads.@threads.
# The validated MomentCoeff / native PauliPropagation core is unchanged.

# ============================================================
# Configuration
# ============================================================

const L = parse(Int, get(ENV, "L", "10"))
const CHECKPOINT_LAYERS = let
    raw = strip(get(ENV, "CHECKPOINT_LAYERS", ""))
    @assert !isempty(raw) "CHECKPOINT_LAYERS must be specified, e.g. 4,8,12"
    layers = sort(unique(parse.(Int, split(raw, ","))))
    @assert !isempty(layers) "CHECKPOINT_LAYERS must contain at least one layer"
    @assert all(layer -> layer >= 1, layers) "All CHECKPOINT_LAYERS must be positive integers"
    layers
end
const MAX_N_LAYER = maximum(CHECKPOINT_LAYERS)

const Ns = parse(Int, get(ENV, "NS", "10000"))
const pT_list = Float64.(collect(0.0:0.1:1.0))

const MASTER_SEED_C = 123
const MASTER_SEED_T = 2026
const JOB_INDEX = parse(Int, get(ENV, "JOB_INDEX", "1"))
const SEED_JOB_STRIDE = parse(Int, get(ENV, "SEED_JOB_STRIDE", "1000000"))
const MAX_RNG_SEED = 2_147_483_647
const INITIAL_OPERATOR_MODE = lowercase(strip(get(ENV, "INITIAL_OPERATOR_MODE", "random_pauli")))
@assert INITIAL_OPERATOR_MODE in ("random_pauli", "center_z") "Invalid INITIAL_OPERATOR_MODE"
const CENTER_SITE = L ÷ 2
# center_z neither reads the P seed setting nor constructs a P RNG.
const MASTER_SEED_P = INITIAL_OPERATOR_MODE == "random_pauli" ? parse(Int, get(ENV, "MASTER_SEED_P", "314159")) : nothing

# :vector uses PauliPropagation.VectorPauliSum and its native vector backend.
# :dict is the production default: one independent realization per worker thread.
const BACKEND = Symbol(get(ENV, "BACKEND", "dict"))
const PARALLEL_REALIZATIONS = lowercase(get(ENV, "PARALLEL_REALIZATIONS", "true")) in ("1", "true", "yes", "on")

const PROGRESS_EVERY = parse(Int, get(ENV, "PROGRESS_EVERY", "5000"))
const RESUME = lowercase(get(ENV, "RESUME", "true")) in ("1", "true", "yes", "on")
const M0_WARNING_TOL = parse(Float64, get(ENV, "M0_WARNING_TOL", "1e-12"))
const M0_HARD_FAIL_TOL = parse(Float64, get(ENV, "M0_HARD_FAIL_TOL", "1e-8"))
const M0_DIAGNOSTIC_LOCK = ReentrantLock()
const OUTPUT_LOCK = ReentrantLock()
const OUTPUT_DIR = get(ENV, "OUTPUT_DIR", dirname(abspath(@__FILE__)))

# Optional reference files. If present, the script compares the new result
# realization-by-realization with a matching initial-operator run.
const REFERENCE_REALIZATIONS = length(ARGS) >= 1 ? ARGS[1] : nothing
const REFERENCE_AVERAGE = length(ARGS) >= 2 ? ARGS[2] : nothing

@assert iseven(L) && L >= 4
@assert MAX_N_LAYER >= 1
@assert PROGRESS_EVERY >= 1
@assert M0_WARNING_TOL > 0
@assert M0_HARD_FAIL_TOL >= M0_WARNING_TOL
@assert Ns >= 1
@assert JOB_INDEX >= 1
@assert SEED_JOB_STRIDE >= Ns "SEED_JOB_STRIDE must be >= NS to guarantee disjoint seed ranges across jobs"
@assert (JOB_INDEX - 1) * SEED_JOB_STRIDE + Ns <= MAX_RNG_SEED "JOB_INDEX/SEED_JOB_STRIDE exceed supported global realization range"
@assert BACKEND in (:vector, :dict)
@assert all(p -> 0.0 <= p <= 1.0, pT_list)

# ============================================================
# Moment-valued Pauli coefficient
# ============================================================
#
# For all paths Gamma ending at Pauli Q,
#   M_k(Q) = sum_{Gamma -> Q} p_Gamma W_Gamma^k,  k=0,1,2.
#
# PauliPropagation normally multiplies a coefficient by an amplitude a.
# For the dephased path problem we need path probability |a|^2 instead.
# Therefore scalar multiplication is defined as
#
#   (M0,M1,M2) * a = |a|^2 (M0,M1,M2).
#
# Merge is ordinary componentwise addition.  With these two operations,
# native Clifford propagation, native T branching, and native merge produce
# exactly the dephased moment algebra used by the original implementation.

struct MomentCoeff
    M0::Float64
    M1::Float64
    M2::Float64
end

MomentCoeff(x::Real) = MomentCoeff(Float64(x), 0.0, 0.0)

Base.zero(::Type{MomentCoeff}) = MomentCoeff(0.0, 0.0, 0.0)
Base.zero(::MomentCoeff) = zero(MomentCoeff)
Base.float(m::MomentCoeff) = m

@inline Base.:+(a::MomentCoeff, b::MomentCoeff) = MomentCoeff(
    a.M0 + b.M0,
    a.M1 + b.M1,
    a.M2 + b.M2,
)

@inline function Base.:*(m::MomentCoeff, a::Number)
    s = Float64(abs2(a))
    return MomentCoeff(s * m.M0, s * m.M1, s * m.M2)
end

@inline Base.:*(a::Number, m::MomentCoeff) = m * a

# PauliPropagation uses this for type-stable numerical bookkeeping.
PauliPropagation.numcoefftype(::Type{MomentCoeff}) = Float64

@inline function accumulate_weight(m::MomentCoeff, w::Int)
    wf = Float64(w)
    return MomentCoeff(
        m.M0,
        m.M1 + wf * m.M0,
        m.M2 + 2.0 * wf * m.M1 + wf^2 * m.M0,
    )
end

# ============================================================
# Exact single-qubit Clifford group used by the original code
# ============================================================

const SIGMA_X = ComplexF64[0 1; 1 0]
const SIGMA_Y = ComplexF64[0 -im; im 0]
const SIGMA_Z = ComplexF64[1 0; 0 -1]
const PAULI_MATS = (SIGMA_X, SIGMA_Y, SIGMA_Z)

const ID1 = ComplexF64[1 0; 0 1]

const X1 = (ID1 + im * SIGMA_X) / sqrt(2)
const X2 = (ID1 - im * SIGMA_X) / sqrt(2)
const Y1 = (ID1 + im * SIGMA_Y) / sqrt(2)
const Y2 = (ID1 - im * SIGMA_Y) / sqrt(2)
const Z1 = (ID1 + im * SIGMA_Z) / sqrt(2)
const Z2 = (ID1 - im * SIGMA_Z) / sqrt(2)

const H1 = (SIGMA_X + SIGMA_Z) / sqrt(2)
const H2 = (SIGMA_X - SIGMA_Z) / sqrt(2)
const H3 = (SIGMA_X + SIGMA_Y) / sqrt(2)
const H4 = (SIGMA_X - SIGMA_Y) / sqrt(2)
const H5 = (SIGMA_Y + SIGMA_Z) / sqrt(2)
const H6 = (SIGMA_Y - SIGMA_Z) / sqrt(2)

const L1 = (ID1 + im * ( SIGMA_X + SIGMA_Y + SIGMA_Z)) / 2
const L2 = (ID1 + im * ( SIGMA_X + SIGMA_Y - SIGMA_Z)) / 2
const L3 = (ID1 + im * ( SIGMA_X - SIGMA_Y + SIGMA_Z)) / 2
const L4 = (ID1 + im * (-SIGMA_X + SIGMA_Y + SIGMA_Z)) / 2
const L5 = (ID1 + im * ( SIGMA_X - SIGMA_Y - SIGMA_Z)) / 2
const L6 = (ID1 + im * (-SIGMA_X + SIGMA_Y - SIGMA_Z)) / 2
const L7 = (ID1 + im * (-SIGMA_X - SIGMA_Y + SIGMA_Z)) / 2
const L8 = (ID1 + im * (-SIGMA_X - SIGMA_Y - SIGMA_Z)) / 2

const G24_MATS = (
    ID1, SIGMA_X, SIGMA_Y, SIGMA_Z,
    X1, X2, Y1, Y2, Z1, Z2,
    H1, H2, H3, H4, H5, H6,
    L1, L2, L3, L4, L5, L6, L7, L8,
)

const G24_SYMBOLS = ntuple(
    i -> Symbol("MOMENT_G24_", lpad(i - 1, 2, '0')),
    24,
)

function exact_singleq_clifford_lookup(U::AbstractMatrix{<:Complex})
    lookup = Tuple{UInt8,Int64}[(0x00, 1)]
    Udag = adjoint(U)

    for input_index in 1:3
        transformed = Udag * PAULI_MATS[input_index] * U
        column = [
            real(tr(PAULI_MATS[output_index] * transformed) / 2)
            for output_index in 1:3
        ]
        output_index = argmax(abs.(column))
        @assert isapprox(abs(column[output_index]), 1.0; atol=1e-12, rtol=0.0)
        sign = Int64(round(Int, column[output_index]))
        push!(lookup, (UInt8(output_index), sign))
    end

    return lookup
end

function register_g24_cliffords!()
    for (symbol, matrix) in zip(G24_SYMBOLS, G24_MATS)
        lookup = exact_singleq_clifford_lookup(matrix)
        if haskey(clifford_map, symbol)
            @assert clifford_map[symbol] == lookup
        else
            clifford_map[symbol] = lookup
        end
    end
    return nothing
end

# ============================================================
# Random Clifford circuit: preserved exactly from old code
# ============================================================

struct Frozen2QBlock
    left_kind1::UInt8
    left_kind2::UInt8
    kind2q::UInt8
    right_kind1::UInt8
    right_kind2::UInt8
end

function draw_weighted_twoq_kind(rng::AbstractRNG)
    u = rand(rng)
    if u < 1 / 20
        return UInt8(0)  # identity
    elseif u < 2 / 20
        return UInt8(1)  # SWAP
    elseif u < 11 / 20
        return UInt8(2)  # iSWAP
    else
        return UInt8(3)  # CNOT
    end
end

function draw_frozen_block(rng::AbstractRNG)
    # Preserve the validated RNG draw order exactly.
    left1 = UInt8(rand(rng, 0:23))
    left2 = UInt8(rand(rng, 0:23))
    right1 = UInt8(rand(rng, 0:23))
    right2 = UInt8(rand(rng, 0:23))
    kind2q = draw_weighted_twoq_kind(rng)
    return Frozen2QBlock(left1, left2, kind2q, right1, right2)
end

function append_g24!(circuit::Vector{Gate}, kind::UInt8, q::Int)
    push!(circuit, CliffordGate(G24_SYMBOLS[Int(kind) + 1], q))
    return circuit
end

function append_twoq_choice!(circuit::Vector{Gate}, q1::Int, q2::Int, kind::UInt8)
    if kind == 0
        return circuit
    elseif kind == 1
        push!(circuit, CliffordGate(:SWAP, (q1, q2)))
    elseif kind == 2
        # Exact Clifford decomposition of iSWAP.
        push!(circuit, CliffordGate(:S, q1))
        push!(circuit, CliffordGate(:S, q2))
        push!(circuit, CliffordGate(:SWAP, (q1, q2)))
        push!(circuit, CliffordGate(:CZ, (q1, q2)))
    elseif kind == 3
        push!(circuit, CliffordGate(:CNOT, (q1, q2)))
    else
        error("Unknown two-qubit block type: $kind")
    end
    return circuit
end

function append_frozen_block!(circuit::Vector{Gate}, q1::Int, q2::Int, block::Frozen2QBlock)
    # Keep the validated gate-order convention unchanged.
    append_g24!(circuit, block.right_kind1, q1)
    append_g24!(circuit, block.right_kind2, q2)
    append_twoq_choice!(circuit, q1, q2, block.kind2q)
    append_g24!(circuit, block.left_kind1, q1)
    append_g24!(circuit, block.left_kind2, q2)
    return circuit
end

function build_even_clifford_layer(rng::AbstractRNG, L::Int)
    circuit = Gate[]
    for q1 in 1:2:(L - 1)
        append_frozen_block!(circuit, q1, q1 + 1, draw_frozen_block(rng))
    end
    return circuit
end

function build_odd_clifford_layer(rng::AbstractRNG, L::Int)
    circuit = Gate[]
    for q1 in 2:2:(L - 2)
        append_frozen_block!(circuit, q1, q1 + 1, draw_frozen_block(rng))
    end
    return circuit
end

function build_clifford_layers(rng::AbstractRNG, L::Int, n_layer::Int)
    return [
        isodd(layer) ? build_even_clifford_layer(rng, L) : build_odd_clifford_layer(rng, L)
        for layer in 1:n_layer
    ]
end

# ============================================================
# T random table: draw once per realization, reuse for all pT
# ============================================================
#
# The old implementation reinitialized rng_T with the same seed for every pT,
# so each pT saw exactly the same sequence of uniforms and only changed the
# threshold u < pT.  We preserve that semantics exactly, but draw the uniforms
# only once per realization.

function draw_T_uniform_table(rng::AbstractRNG)
    uniforms = fill(NaN, MAX_N_LAYER, L)

    for layer in 1:MAX_N_LAYER
        active_sites = isodd(layer) ? (1:L) : (2:(L - 1))
        for q in active_sites
            uniforms[layer, q] = rand(rng)
        end
    end

    return uniforms
end

function T_pattern_from_uniforms(uniforms::AbstractMatrix{<:Real}, pT::Float64)
    @assert 0.0 <= pT <= 1.0
    pattern = falses(MAX_N_LAYER, L)

    for layer in 1:MAX_N_LAYER
        active_sites = isodd(layer) ? (1:L) : (2:(L - 1))
        @inbounds for q in active_sites
            pattern[layer, q] = uniforms[layer, q] < pT
        end
    end

    return pattern
end

# ============================================================
# Native PauliPropagation moment engine
# ============================================================

# Packed encoding: 0=I, 1=X, 2=Y, 3=Z; independent draw at every site.
function draw_random_pauli_term(rng::AbstractRNG)
    term = zero(getinttype(L))
    for q in 1:L
        term = setpauli(term, rand(rng, 0:3), q)
    end
    return term
end

function initial_moment_state(initial_term)
    p0 = PauliString(L, initial_term, MomentCoeff(1.0, 0.0, 0.0))
    if BACKEND == :vector
        return VectorPauliSum(p0)
    else
        return PauliSum(p0)
    end
end

# PauliSum backend: update coefficients in place without changing keys.
function apply_noise_bookkeeping!(state::PauliSum)
    for pstr in keys(state.terms)
        state.terms[pstr] = accumulate_weight(state.terms[pstr], countweight(pstr))
    end
    return state
end

# VectorPauliSum backend: contiguous arrays; countweight works directly on packed Paulis.
function apply_noise_bookkeeping!(state::VectorPauliSum)
    @assert length(state.terms) == length(state.coeffs)
    @inbounds for i in eachindex(state.terms)
        state.coeffs[i] = accumulate_weight(state.coeffs[i], countweight(state.terms[i]))
    end
    return state
end

function apply_CTD_layer!(state, clifford_layer::Vector{Gate}, t_pattern::AbstractVector{Bool})
    # Preserve old semantics exactly:
    #   1) old code propagated the whole Clifford layer with heisenberg=true,
    #   2) then applied active T gates,
    #   3) then accumulated Pauli weight once.
    propagate!(
        clifford_layer,
        state;
        heisenberg=true,
        min_abs_coeff=0.0,
    )

    t_gates = Gate[TGate(q) for q in 1:L if t_pattern[q]]
    if !isempty(t_gates)
        # T gates act on distinct sites and commute, so batching the T sublayer
        # is equivalent to the old site-by-site application.
        propagate!(
            t_gates,
            state;
            heisenberg=true,
            min_abs_coeff=0.0,
        )
    end

    apply_noise_bookkeeping!(state)
    return state
end

function summarize_moment_state(state)
    total_M0 = 0.0
    total_M1 = 0.0
    total_M2 = 0.0

    for m in coefficients(state)
        total_M0 += m.M0
        total_M1 += m.M1
        total_M2 += m.M2
    end

    mean_W = total_M1 / total_M0
    mean_W2 = total_M2 / total_M0

    return (
        n_paulis=length(state),
        total_M0=total_M0,
        mean_W=mean_W,
        mean_W2=mean_W2,
        variance=mean_W2 - mean_W^2,
    )
end

function run_prepared_realization_checkpoints(
    clifford_layers,
    T_uniforms,
    initial_term,
    pT::Float64,
    realization_index::Int,
)
    T_pattern = T_pattern_from_uniforms(T_uniforms, pT)
    state = initial_moment_state(initial_term)
    checkpoint_variances = Vector{Float64}(undef, length(CHECKPOINT_LAYERS))
    checkpoint_max_paulis = Vector{Int}(undef, length(CHECKPOINT_LAYERS))
    checkpoint_index = 1
    running_max_paulis = 0

    for layer in 1:MAX_N_LAYER
        apply_CTD_layer!(
            state,
            clifford_layers[layer],
            view(T_pattern, layer, :),
        )

        summary = summarize_moment_state(state)
        running_max_paulis = max(running_max_paulis, summary.n_paulis)
        err_M0 = abs(summary.total_M0 - 1.0)

        if !isfinite(err_M0) || err_M0 > M0_WARNING_TOL
            lock(M0_DIAGNOSTIC_LOCK) do
                @printf(
                    "M0 WARNING: realization=%d pT=%.1f layer=%d n_paulis=%d M0=%.17g |M0-1|=%.3e\n",
                    realization_index,
                    pT,
                    layer,
                    summary.n_paulis,
                    summary.total_M0,
                    err_M0,
                )
                flush(stdout)
            end
        end

        if !isfinite(err_M0) || err_M0 > M0_HARD_FAIL_TOL
            error(
                "M0 normalization error exceeded hard threshold: " *
                "realization=$(realization_index), pT=$(pT), layer=$(layer), " *
                "n_paulis=$(summary.n_paulis), M0=$(summary.total_M0), " *
                "|M0-1|=$(err_M0), threshold=$(M0_HARD_FAIL_TOL)"
            )
        end

        if checkpoint_index <= length(CHECKPOINT_LAYERS) && layer == CHECKPOINT_LAYERS[checkpoint_index]
            checkpoint_variances[checkpoint_index] = summary.variance
            checkpoint_max_paulis[checkpoint_index] = running_max_paulis
            checkpoint_index += 1
        end
    end

    @assert checkpoint_index == length(CHECKPOINT_LAYERS) + 1
    return checkpoint_variances, checkpoint_max_paulis
end

function prepare_realization(seed_C::Int, seed_T::Int, seed_P::Int)
    rng_C = MersenneTwister(seed_C)
    rng_T = MersenneTwister(seed_T)

    clifford_layers = build_clifford_layers(rng_C, L, MAX_N_LAYER)
    T_uniforms = draw_T_uniform_table(rng_T)

    initial_term = INITIAL_OPERATOR_MODE == "random_pauli" ?
        draw_random_pauli_term(MersenneTwister(seed_P)) :
        setpauli(zero(getinttype(L)), 3, CENTER_SITE)
    return clifford_layers, T_uniforms, initial_term
end

# ============================================================
# Deterministic, non-overlapping seed assignment across jobs
# ============================================================

@inline function global_realization_id(local_index::Int)
    return (JOB_INDEX - 1) * SEED_JOB_STRIDE + local_index
end

@inline function seed_from_global(master_seed::Int, global_id::Int)
    # Addition modulo MAX_RNG_SEED is a bijection on the allowed seed space.
    # Therefore distinct global_id values map to distinct seeds within each RNG stream.
    return mod(master_seed - 1 + global_id, MAX_RNG_SEED) + 1
end

function seeds_for_local(local_index::Int)
    gid = global_realization_id(local_index)
    seed_C = seed_from_global(MASTER_SEED_C, gid)
    seed_T = seed_from_global(MASTER_SEED_T, gid)
    seed_P = INITIAL_OPERATOR_MODE == "random_pauli" ? seed_from_global(MASTER_SEED_P, gid) : 0
    return gid, seed_C, seed_T, seed_P
end

# ============================================================
# Incremental checkpoint / resume I/O
# ============================================================

function operator_tag()
    return INITIAL_OPERATOR_MODE == "random_pauli" ?
        "random_pauli_seedP$(MASTER_SEED_P)" : "center_z_site$(CENTER_SITE)"
end

function checkpoint_tag()
    return join(CHECKPOINT_LAYERS, "-")
end

function run_tag()
    return "$(operator_tag())_L$(L)_checkpoints$(checkpoint_tag())_Ns$(Ns)_job$(lpad(JOB_INDEX, 3, '0'))_$(BACKEND)"
end

function progress_path()
    return joinpath(OUTPUT_DIR, "realization_progress_$(run_tag()).csv")
end

function realization_path(checkpoint_layer::Int)
    stem = "$(operator_tag())_L$(L)_checkpoints$(checkpoint_tag())_layers$(checkpoint_layer)_Ns$(Ns)_job$(lpad(JOB_INDEX, 3, '0'))_$(BACKEND)"
    return joinpath(OUTPUT_DIR, "variance_realizations_$(stem).csv")
end

function average_path(checkpoint_layer::Int)
    stem = "$(operator_tag())_L$(L)_checkpoints$(checkpoint_tag())_layers$(checkpoint_layer)_Ns$(Ns)_job$(lpad(JOB_INDEX, 3, '0'))_$(BACKEND)"
    return joinpath(OUTPUT_DIR, "variance_average_$(stem).csv")
end

function progress_row_dataframe(
    local_index::Int,
    global_id::Int,
    seed_C::Int,
    seed_T::Int,
    seed_P::Int,
    initial_pauli::String,
    Vlocal::Matrix{Float64},
    max_paulis_local::Vector{Int},
)
    df = DataFrame(
        job_index=[JOB_INDEX],
        realization=[local_index],
        global_realization=[global_id],
        seed_C=[seed_C],
        seed_T=[seed_T],
        seed_P=INITIAL_OPERATOR_MODE == "random_pauli" ? [seed_P] : Union{Missing,Int}[missing],
        initial_operator_mode=[INITIAL_OPERATOR_MODE],
        center_site=INITIAL_OPERATOR_MODE == "center_z" ? Union{Missing,Int}[CENTER_SITE] : Union{Missing,Int}[missing],
        initial_pauli=[initial_pauli],
        L=[L],
        max_N_LAYER=[MAX_N_LAYER],
    )
    for (k, checkpoint_layer) in enumerate(CHECKPOINT_LAYERS)
        df[!, Symbol("max_n_paulis_layers$(checkpoint_layer)")] = [max_paulis_local[k]]
        for (j, pT) in enumerate(pT_list)
            df[!, Symbol("layers$(checkpoint_layer)_pT_$(pT)")] = [Vlocal[j, k]]
        end
    end
    return df
end

function checkpoint_row_dataframe(progress_row::DataFrame, checkpoint_layer::Int)
    df = DataFrame(
        job_index=progress_row.job_index,
        realization=progress_row.realization,
        global_realization=progress_row.global_realization,
        seed_C=progress_row.seed_C,
        seed_T=progress_row.seed_T,
        seed_P=progress_row.seed_P,
        initial_operator_mode=progress_row.initial_operator_mode,
        center_site=progress_row.center_site,
        initial_pauli=progress_row.initial_pauli,
        L=fill(L, nrow(progress_row)),
        N_LAYER=fill(checkpoint_layer, nrow(progress_row)),
        max_n_paulis=progress_row[!, Symbol("max_n_paulis_layers$(checkpoint_layer)")],
    )
    for pT in pT_list
        df[!, Symbol("pT_$(pT)")] = progress_row[!, Symbol("layers$(checkpoint_layer)_pT_$(pT)")]
    end
    return df
end

function append_csv(path::String, df::DataFrame)
    exists = isfile(path) && filesize(path) > 0
    CSV.write(path, df; append=exists, writeheader=!exists)
    return nothing
end

function load_progress_safely(path::String)
    if !isfile(path) || filesize(path) == 0
        return DataFrame()
    end
    try
        return CSV.read(path, DataFrame)
    catch err
        # A hard kill can theoretically leave only the final CSV row incomplete.
        # Recover by dropping exactly one trailing nonempty line and retrying.
        lines = readlines(path)
        if length(lines) <= 1
            rethrow(err)
        end
        @warn "Progress CSV could not be parsed; attempting recovery by dropping the final line" path exception=(err, catch_backtrace())
        open(path, "w") do io
            for line in lines[1:end-1]
                println(io, line)
            end
        end
        return CSV.read(path, DataFrame)
    end
end

function validate_progress!(df::DataFrame)
    nrow(df) == 0 && return nothing
    required = [:job_index, :realization, :global_realization, :seed_C, :seed_T, :initial_operator_mode, :L, :max_N_LAYER]
    for col in required
        @assert col in propertynames(df) "Resume file is missing required column $(col)"
    end
    @assert all(df.job_index .== JOB_INDEX) "Resume file JOB_INDEX mismatch"
    @assert all(df.L .== L) "Resume file L mismatch"
    @assert all(df.max_N_LAYER .== MAX_N_LAYER) "Resume file depth mismatch"
    @assert all(df.initial_operator_mode .== INITIAL_OPERATOR_MODE) "Resume file initial-operator mode mismatch"
    @assert all((1 .<= df.realization) .& (df.realization .<= Ns)) "Resume file contains realization outside 1:NS"
    @assert length(unique(df.realization)) == nrow(df) "Resume file contains duplicate realization rows"
    for row in eachrow(df)
        gid, sc, st, sp = seeds_for_local(row.realization)
        @assert row.global_realization == gid "Resume global realization mismatch at local realization $(row.realization)"
        @assert row.seed_C == sc && row.seed_T == st "Resume C/T seed mismatch at local realization $(row.realization)"
        if INITIAL_OPERATOR_MODE == "random_pauli"
            @assert row.seed_P == sp "Resume P seed mismatch at local realization $(row.realization)"
        end
    end
    return nothing
end

function rebuild_checkpoint_realization_csvs(progress_df::DataFrame)
    for checkpoint_layer in CHECKPOINT_LAYERS
        path = realization_path(checkpoint_layer)
        if nrow(progress_df) == 0
            isfile(path) && rm(path)
        else
            ordered = sort(progress_df, :realization)
            CSV.write(path, checkpoint_row_dataframe(ordered, checkpoint_layer))
        end
    end
    return nothing
end

function write_average_csvs(progress_df::DataFrame)
    n = nrow(progress_df)
    n == 0 && return nothing
    for checkpoint_layer in CHECKPOINT_LAYERS
        cdf = checkpoint_row_dataframe(progress_df, checkpoint_layer)
        means = Float64[]
        sems = Float64[]
        for pT in pT_list
            values = Float64.(cdf[!, Symbol("pT_$(pT)")])
            push!(means, mean(values))
            push!(sems, n > 1 ? std(values; corrected=true) / sqrt(n) : NaN)
        end
        average_df = DataFrame(
            job_index=fill(JOB_INDEX, length(pT_list)),
            initial_operator_mode=fill(INITIAL_OPERATOR_MODE, length(pT_list)),
            center_site=fill(INITIAL_OPERATOR_MODE == "center_z" ? CENTER_SITE : missing, length(pT_list)),
            pT=pT_list,
            mean_variance=means,
            SEM=sems,
            Ns_completed=fill(n, length(pT_list)),
            Ns_target=fill(Ns, length(pT_list)),
            L=fill(L, length(pT_list)),
            N_LAYER=fill(checkpoint_layer, length(pT_list)),
        )
        CSV.write(average_path(checkpoint_layer), average_df)
    end
    return nothing
end

function persist_completed_realization!(progress_row::DataFrame)
    lock(OUTPUT_LOCK) do
        # The progress CSV is the source of truth for resume. Write it first.
        append_csv(progress_path(), progress_row)
        # Keep the familiar per-checkpoint realization CSVs current too.
        for checkpoint_layer in CHECKPOINT_LAYERS
            append_csv(realization_path(checkpoint_layer), checkpoint_row_dataframe(progress_row, checkpoint_layer))
        end
    end
    return nothing
end

# ============================================================
# Ensemble scan with realization-level resume
# ============================================================

function scan_ensemble(pending_realizations::Vector{Int}, completed_at_start::Int)
    started = time()
    total_completed = Threads.Atomic{Int}(completed_at_start)
    print_lock = ReentrantLock()

    function run_one(s::Int)
        gid, seed_C, seed_T, seed_P = seeds_for_local(s)
        clifford_layers, T_uniforms, initial_term = prepare_realization(seed_C, seed_T, seed_P)
        initial_pauli = inttostring(initial_term, L)

        Vlocal = Matrix{Float64}(undef, length(pT_list), length(CHECKPOINT_LAYERS))
        max_paulis_local = zeros(Int, length(CHECKPOINT_LAYERS))

        @inbounds for (j, pT) in enumerate(pT_list)
            checkpoint_variances, checkpoint_max_paulis = run_prepared_realization_checkpoints(
                clifford_layers, T_uniforms, initial_term, pT, s
            )
            for k in eachindex(CHECKPOINT_LAYERS)
                Vlocal[j, k] = checkpoint_variances[k]
                max_paulis_local[k] = max(max_paulis_local[k], checkpoint_max_paulis[k])
            end
        end

        row = progress_row_dataframe(s, gid, seed_C, seed_T, seed_P, initial_pauli, Vlocal, max_paulis_local)
        persist_completed_realization!(row)

        n_done = Threads.atomic_add!(total_completed, 1) + 1
        if n_done == 1 || n_done % PROGRESS_EVERY == 0 || n_done == Ns
            lock(print_lock) do
                @printf("Completed %d/%d realizations in this job (%d newly completed this invocation, %.2f s)\n",
                    n_done, Ns, n_done - completed_at_start, time() - started)
                flush(stdout)
            end
        end
        return nothing
    end

    if PARALLEL_REALIZATIONS && Threads.nthreads() > 1
        Threads.@threads :dynamic for idx in eachindex(pending_realizations)
            run_one(pending_realizations[idx])
        end
    else
        for s in pending_realizations
            run_one(s)
        end
    end

    return time() - started
end

# ============================================================
# Main
# ============================================================

function main()
    register_g24_cliffords!()
    mkpath(OUTPUT_DIR)

    println("Julia ", VERSION)
    println("PauliPropagation ", pkgversion(PauliPropagation))
    println("L=$L, max_layers=$MAX_N_LAYER, Ns=$Ns, pT points=$(length(pT_list))")
    println("checkpoints = ", CHECKPOINT_LAYERS)
    println("backend=$BACKEND, Julia threads=$(Threads.nthreads()), parallel_realizations=$PARALLEL_REALIZATIONS")
    println("M0 diagnostic: warning_tol=$M0_WARNING_TOL, hard_fail_tol=$M0_HARD_FAIL_TOL")
    println("resume=$RESUME, job_index=$JOB_INDEX, seed_job_stride=$SEED_JOB_STRIDE")
    println("global realization range = $(global_realization_id(1)):$(global_realization_id(Ns))")
    println("initial_operator_mode=$INITIAL_OPERATOR_MODE")
    println(INITIAL_OPERATOR_MODE == "random_pauli" ?
        "initial Pauli: iid uniform I/X/Y/Z (identity included), master_seed_P=$MASTER_SEED_P" :
        "initial Pauli: fixed Z at site $CENTER_SITE (sites numbered 1 through L)")

    ppath = progress_path()
    if !RESUME && isfile(ppath)
        error("Progress file already exists and RESUME=false: $ppath")
    end

    progress_df = RESUME ? load_progress_safely(ppath) : DataFrame()
    validate_progress!(progress_df)

    # Rebuild per-checkpoint CSVs from the source-of-truth progress CSV.
    # This repairs any interruption that happened between individual file writes.
    rebuild_checkpoint_realization_csvs(progress_df)

    completed_set = Set(Int.(nrow(progress_df) == 0 ? Int[] : progress_df.realization))
    pending = [s for s in 1:Ns if !(s in completed_set)]

    println("Resume status: $(length(completed_set))/$Ns already complete; $(length(pending)) pending")
    println("Progress CSV: ", ppath)

    elapsed_seconds = 0.0
    if !isempty(pending)
        elapsed_seconds = scan_ensemble(pending, length(completed_set))
    else
        println("All realizations are already complete; no propagation work needed.")
    end

    final_progress = load_progress_safely(ppath)
    validate_progress!(final_progress)
    final_progress = sort(final_progress, :realization)
    rebuild_checkpoint_realization_csvs(final_progress)
    write_average_csvs(final_progress)

    n_completed = nrow(final_progress)
    println("\nCompleted rows on disk: $n_completed/$Ns")
    for checkpoint_layer in CHECKPOINT_LAYERS
        println("Saved checkpoint layers=$checkpoint_layer:")
        println("  ", realization_path(checkpoint_layer))
        println("  ", average_path(checkpoint_layer))
    end

    metadata = Dict(
        "L" => L,
        "max_N_LAYER" => MAX_N_LAYER,
        "checkpoint_layers" => CHECKPOINT_LAYERS,
        "Ns_target" => Ns,
        "Ns_completed" => n_completed,
        "job_index" => JOB_INDEX,
        "seed_job_stride" => SEED_JOB_STRIDE,
        "global_realization_start" => global_realization_id(1),
        "global_realization_end" => global_realization_id(Ns),
        "pT_list" => pT_list,
        "master_seed_C" => MASTER_SEED_C,
        "master_seed_T" => MASTER_SEED_T,
        "seed_scheme" => "seed = mod(master_seed - 1 + global_realization, 2147483647) + 1; disjoint global-realization ranges guarantee disjoint seeds within each RNG stream",
        "initial_operator_mode" => INITIAL_OPERATOR_MODE,
        "initial_pauli_distribution" => INITIAL_OPERATOR_MODE == "random_pauli" ? "iid uniform over {I,X,Y,Z} on every site; all-identity included" : "fixed single-site Z at L/2",
        "initial_pauli_site_order" => "left-to-right sites 1 through L",
        "initial_pauli_reuse" => INITIAL_OPERATOR_MODE == "random_pauli" ? "one draw per realization, reused across all pT values and checkpoints" : "same fixed operator for all realizations, pT values, and checkpoints",
        "circuit_reuse" => "one depth-nL Clifford realization per realization, shared by all pT; checkpoints are prefixes of the same circuit trajectory",
        "max_n_paulis_definition" => "for each checkpoint: maximum number of active Pauli terms seen over all pT values and all layers up to that checkpoint",
        "rng_type" => "MersenneTwister",
        "backend" => string(BACKEND),
        "parallel_realizations" => PARALLEL_REALIZATIONS,
        "resume_enabled" => RESUME,
        "progress_csv" => ppath,
        "M0_warning_tolerance" => M0_WARNING_TOL,
        "M0_hard_fail_tolerance" => M0_HARD_FAIL_TOL,
        "julia_threads" => Threads.nthreads(),
        "julia_version" => string(VERSION),
        "PauliPropagation_version" => string(pkgversion(PauliPropagation)),
        "elapsed_seconds_this_invocation" => elapsed_seconds,
        "coefficient_type" => "MomentCoeff(M0,M1,M2)",
        "branching_rule" => "MomentCoeff * a := abs2(a) * MomentCoeff",
        "merge_rule" => "componentwise addition",
    )
    if INITIAL_OPERATOR_MODE == "random_pauli"
        metadata["master_seed_P"] = MASTER_SEED_P
    else
        metadata["center_site"] = CENTER_SITE
    end

    metadata_path = joinpath(OUTPUT_DIR, "run_metadata_$(run_tag()).toml")
    open(metadata_path, "w") do io
        TOML.print(io, metadata)
    end

    @printf("Elapsed propagation time this invocation: %.6f s\n", elapsed_seconds)
    println("Metadata: ", metadata_path)
    println("Saved outputs under: ", abspath(OUTPUT_DIR))
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
