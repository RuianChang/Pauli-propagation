# Pauli-Path Weight Statistics with Deterministic Moment Propagation

**Heisenberg-picture simulation of random Clifford+T circuits using [PauliPropagation.jl](https://github.com/SparqleSim/PauliPropagation.jl).**

This repository computes the **variance of the accumulated Pauli-path weight** in quantum circuits without enumerating every path, explicitly inserting random dephasing Paulis, or taking numerical derivatives. It extends the native Pauli propagation engine with a custom three-moment coefficient type and uses multithreaded, reproducible ensemble simulations. The example below shows the depth-four (`D=4`) results.

![D=4 path-weight variance density for N = 4, 6, 8, 10, 12, 14 and the exact thermodynamic-limit curve](figures/d4_variance_density.png)

*Figure: Ensemble-averaged variance density, `V/(ND)`, for open-boundary, random Clifford+T brickwork circuits at depth `D=4`. Markers show finite-size simulations for `N=4,6,8,10,12,14`; the black dashed curve is the exact thermodynamic-limit prediction from a separate replica tensor-network calculation. Figure supplied by the author; the underlying finite-size CSV data were not included in this repository snapshot.*

## Highlights

- **Custom moment propagation:** each active Pauli string carries `(M0, M1, M2)`, retaining the first two moments of the **path-weight distribution**, rather than just an ordinary complex Pauli amplitude.
- **Native sparse operator evolution:** reuse `PauliPropagation.jl`'s Clifford conjugation, T-gate branching, Pauli-string merging, and `PauliSum`/`VectorPauliSum` data structures.
- **No noise simulation or finite differences:** depolarizing noise motivates the observable, but only the noiseless Clifford+T gates and a deterministic weight-update rule are needed to calculate its zero-noise susceptibility.
- **Ensemble-scale computation:** reuse each sampled Clifford circuit and T-gate random-number table across all values of `pT`; parallelize independent realizations using Julia threads.
- **Reproducible runs:** deterministic seed assignment across jobs, depth checkpoints, CSV outputs, normalization diagnostics, and realization-level checkpoint/resume.

## Background

A Heisenberg-evolved Pauli operator can be expanded over **Pauli paths** `Γ`, each representing a sequence of Pauli strings through the circuit, with path amplitude `c_Γ`. For a path of depth `D`, define the accumulated **path weight**

$$
W_\Gamma = \sum_{\ell=1}^{D} w(P_{\Gamma,\ell}),
\qquad w(P)=\text{number of non-identity factors in }P.
$$

After random-Pauli dephasing removes interference between distinct paths, the squared amplitudes define normalized path probabilities `p_Γ=|c_Γ|²`. The dephased noisy-echo response to local depolarizing noise of strength `γ` is

$$
F(\gamma)=\sum_\Gamma p_\Gamma e^{-2\gamma W_\Gamma},
\qquad \sum_\Gamma p_\Gamma=1.
$$

Its zero-noise curvature measures the path-weight variance:

$$
V_{\mathcal R}
=\langle W^2\rangle_{\mathcal R}-\langle W\rangle_{\mathcal R}^2
=\left.\frac14\frac{\partial^2}{\partial\gamma^2}\ln F(\gamma)\right|_{\gamma=0}.
$$

Here `𝓡=(P,U)` specifies an initial Pauli string `P` and a random circuit realization `U`. A Clifford-only circuit has a single unbranched Pauli path for each input Pauli, and therefore `V_𝓡=0`. Adding T gates creates branching and potentially nonzero path-weight fluctuations.

## Numerical method: three moments per Pauli string

For a partially propagated path that ends at Pauli string `Q`, track

$$
M_k^{(\ell)}(Q)=\sum_{\Gamma_\ell\to Q}p_{\Gamma_\ell}W_{\Gamma_\ell}^{k},
\qquad k=0,1,2.
$$

Initialize the selected input Pauli with `(M0,M1,M2)=(1,0,0)`. For every complete circuit layer:

1. **Clifford + T propagation:** each Pauli transition with amplitude `a` sends `(M0,M1,M2) → |a|²(M0,M1,M2)`. Contributions arriving at the same Pauli string are *added componentwise*, not coherently combined as amplitudes.
2. **Weight accumulation:** after the entire Clifford+T layer, let `w=w(Q)` and update

   $$
   \begin{aligned}
   M_0' &= M_0,\\
   M_1' &= M_1+wM_0,\\
   M_2' &= M_2+2wM_1+w^2M_0.
   \end{aligned}
   $$

   Both right-hand sides use the **pre-update** moments. This accounts for local depolarizing-noise weight once per complete layer, without explicitly applying noise.
3. **Read out a checkpoint:** sum each moment over all final Pauli strings, `m_k=Σ_Q M_k(Q)`, and calculate

   $$V_{\mathcal R}=\frac{m_2}{m_0}-\left(\frac{m_1}{m_0}\right)^2.$$

   The exact zeroth moment is `m0=1`; the implementation checks normalization at every layer.

The ensemble mean is calculated **after** computing each realization's variance:

$$
V(N,D,p_T)=\mathbb E_{P,U}[V_{\mathcal R}]
\approx\frac1{N_s}\sum_{r=1}^{N_s}V_{\mathcal R_r},
\qquad v_D=\frac{V(N,D,p_T)}{ND}.
$$

This is the mean of the **within-realization variances**; it is not the variance of the ensemble-averaged path weight. The default input `P` is sampled uniformly from `{I,X,Y,Z}^N` (including the identity string). The circuit `U` samples independent two-qubit Clifford blocks and single-qubit T gates.

## Circuit and implementation details

- **Geometry:** an even number `N=L≥4` of qubits in an open chain, with alternating brickwork pairs `(1,2),(3,4),...` and `(2,3),(4,5),...`.
- **Layer order:** sampled two-qubit Clifford layer → independent T gates on the active sites (probability `pT`) → one weight bookkeeping step on **every** qubit's Pauli string.
- **Boundary convention:** in even layers the first and last qubits are not eligible for T insertion. They still contribute to the Pauli weight.
- **Clifford sampling:** two-qubit blocks are built from independent single-qubit Cliffords surrounding one of `I`, `SWAP`, `iSWAP`, or `CNOT`; the code uses respective class weights `1/20, 1/20, 9/20, 9/20` to implement the manuscript's uniformly sampled two-qubit Clifford ensemble.
- **Sparse representation:** Pauli strings are stored with native packed representations; custom `MomentCoeff` values are added on term collisions. `BACKEND=dict` (default) uses `PauliSum`; `BACKEND=vector` uses `VectorPauliSum`.
- **Precision / truncation:** `Float64` moment values; `min_abs_coeff=0.0`, so there is no user-selected coefficient truncation. Floating-point rounding and memory growth still apply.
- **Complexity:** the number of active Pauli strings may grow rapidly with the number of T-induced branches. This is *not* a general polynomial-time simulator of arbitrary Clifford+T circuits.

## Installation and quick start

Requires **Julia 1.10+** and `PauliPropagation.jl`, `CSV.jl`, and `DataFrames.jl`. From the repository root, install the Julia project dependencies:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Run a small local demonstration with four threads and save outputs to `results/`:

```bash
mkdir -p results
L=8 CHECKPOINT_LAYERS=4 NS=100 JOB_INDEX=1 \
  INITIAL_OPERATOR_MODE=random_pauli BACKEND=dict \
  OUTPUT_DIR=results JULIA_NUM_THREADS=4 \
  julia --project=. --threads=4 src/simulate_variance.jl
```

The source scans `pT = 0.0, 0.1, ..., 1.0` (set inside the Julia file). Example alternatives:

```bash
# Record multiple prefix depths of the same circuits
L=8 CHECKPOINT_LAYERS=2,4 NS=100 JOB_INDEX=1 OUTPUT_DIR=results \
  julia --project=. --threads=4 src/simulate_variance.jl

# Propagate a fixed Z operator on site L/2 instead of a random input Pauli
L=8 CHECKPOINT_LAYERS=4 NS=100 JOB_INDEX=2 \
  INITIAL_OPERATOR_MODE=center_z OUTPUT_DIR=results \
  julia --project=. --threads=4 src/simulate_variance.jl
```

**Important:** use different `JOB_INDEX` values for separate independent jobs, and keep `SEED_JOB_STRIDE ≥ NS` (default: 1,000,000). Reusing the same job index with `RESUME=true` **resumes that job** rather than creating a new independent sample. Do not run concurrent writers for the same job index / output files.

### HPC / Slurm

A portable version of the submitted TACC Slurm launcher is in `scripts/run_slurm.sh`. Add your cluster's partition/account directives as needed. For example:

```bash
JOB_INDEX=1 L=14 NS=10000 CHECKPOINT_LAYERS=4 sbatch scripts/run_slurm.sh
```

The repository version removes the original machine-specific Julia installation path, cluster account number, and personal scheduler email. It keeps the original computation's sampling and checkpointing semantics.

## Outputs and reproducibility

For every requested checkpoint depth the Julia script writes:

| File pattern (under `OUTPUT_DIR`) | Contents |
| --- | --- |
| `variance_average_*.csv` | `pT`, `mean_variance`, `SEM`, `Ns_completed`, `L`, `N_LAYER`, plus metadata. Divide `mean_variance` and `SEM` by `L*N_LAYER` to obtain a density. |
| `variance_realizations_*.csv` | One row per realization, with per-`pT` variance columns, seeds, input operator, and maximum active Pauli count. |
| `realization_progress_*.csv` | Incrementally persisted source of truth for resume and recovery. |
| `run_metadata_*.toml` | Julia / PauliPropagation versions, random seed scheme, backend, checkpoints, and run parameters. |

Independent streams for Clifford gates, T gates, and (in random-input mode) initial Paulis are derived from deterministic global realization IDs. Each sampled Clifford circuit and underlying T uniform table is **shared across all `pT` values** for that realization; T insertions use the threshold `u<pT`. The same initial Pauli is reused at all `pT` values. Threading changes execution order but not per-realization random seeds.

For a quick analytic sanity check, `pT=0` should give zero variance; `m0` should remain unity up to floating-point tolerance. The saved normalization diagnostics enforce a hard failure if `|m0-1|` exceeds the configured tolerance.

## Depth-four (`D=4`) results

The figure at the top shows a monotonic increase of `v4(N,pT)` with T-gate insertion probability. At fixed nonzero `pT`, the finite-size curves for `N=4,6,8,10,12,14` lie below and approach the thermodynamic-limit curve as `N` grows. In particular, the numerics do **not** collapse exactly onto the infinite-size line at these sizes.

The exact thermodynamic-limit density was obtained **separately** through a two-replica tensor-network transfer calculation (it is not computed by the Julia program in this repository):

$$
v_4^{(\infty)}(p)=\frac{p\,Q(p)}{4\times233280000000},\qquad p=p_T.
$$

<details>
<summary>Show the exact polynomial Q(p)</summary>

$$
\begin{aligned}
Q(p)={}&232626816000-191121545088p+128513835360p^2\\
&-70449107184p^3+30986183880p^4-10835143470p^5\\
&+2935613475p^6-597691575p^7+88172775p^8\\
&-8898600p^9+549375p^{10}-15625p^{11}.
\end{aligned}
$$

</details>

Here `p Q(p)/233280000000` is the variance **per qubit** `V/N`; the plotted spacetime variance density is `V/(4N)`, which introduces the additional factor of four. The analytical curve obeys `v4(infinity,0)=0` and `v4(infinity,1)≈0.1308928855`. The pure-Python implementation and values on the `pT=0:0.1:1` grid are provided in `scripts/d4_analytic.py` and `data/d4_thermodynamic_limit.csv`.

To recreate the plotted comparison after running your own finite-size simulations, install `matplotlib` and pass the generated average CSVs (from multiple `L` values) to:

```bash
python -m pip install matplotlib
python scripts/plot_d4.py --inputs results/variance_average_*.csv \
  --output figures/d4_reproduced.png
```

**Data note:** the uploaded figure is included, but the original per-size simulation CSVs, their sample counts, and error bars were not supplied. The figure is therefore shown as a reference result, not misrepresented as independently reproduced from data in this archive. The plotting script reproduces the *method of comparison* using newly generated CSVs, not the exact pictured markers.

## Attribution and references

This implementation builds on **[PauliPropagation.jl](https://github.com/SparqleSim/PauliPropagation.jl)**. The `PauliSum`/`VectorPauliSum`, Clifford/T propagation routines, and underlying package belong to its contributors; the `MomentCoeff` moment algebra and application to path-weight variance are implemented in the included research script.

> M. S. Rudolph, T. Jones, Y. Teng, A. Angrisani, and Z. Holmes, **“Pauli Propagation: A Computational Framework for Simulating Quantum Systems,”** *PRX Quantum* **7**, 032001 (2026). [DOI: 10.1103/6vd7-l9bn](https://doi.org/10.1103/6vd7-l9bn) · [arXiv:2505.21606](https://arxiv.org/abs/2505.21606).

Source figure, numerical script, and theory summary: Rui-An Chang, research on Pauli-path statistics in random Clifford+T circuits (manuscript in preparation, November 2026). The unpublished manuscript itself is **not distributed** in this archive.

## Scope and limitations

- This is a research implementation of a specialized moment observable, **not** a fault-tolerant decoder, general stabilizer simulator, or CUDA/GPU library.
- The numerical engine is an unmodified copy of the provided Julia research code, renamed for this repository; the Slurm wrapper was made portable. No raw D=4 finite-size CSVs were attached.
- The available environment did **not** include Julia, so the full Julia simulation was not executed as part of archive preparation. Before citing performance or publishing as a reproducible software release, run the small smoke test and verify compatibility with your installed `PauliPropagation.jl` version; consider committing the resulting verified `Manifest.toml`.
- The author should confirm coauthor approval before publicly releasing unpublished figures, analytical formulas, and code.
