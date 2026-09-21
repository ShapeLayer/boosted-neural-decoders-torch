# Boosted LDPC decoders - PyTorch

A PyTorch port of the implementation of [*Boosting learning for LDPC codes*](https://github.com/ghy1228/LDPC_Error_Floor).  

## Getting Started

```sh
uv sync
uv run main.py --mode base --smoke
uv run main.py --mode post --smoke
uv run main.py --mode collect --config configs/collect.json --smoke
```

`--smoke` checks a small run; base uses 2 decoding iterations, while post retains the 20 frozen iterations and adds 2 trainable iterations. Collection uses the configured initial weights, or initialization values if no weight file is set.

Alternatively:

```sh
uv run main_Base.py --smoke  # uv run main.py --mode base
uv run main_Post.py --smoke  # uv run main.py --mode post
```

With `device=auto`, the runtime selects CUDA, then MPS, then CPU. You can also specify `--device cpu`, `--device cuda:0`, or `--device mps`. If the specified GPU is not available, an error will be raised. If you have `"device": "cpu"` in your experiment JSON settings, overwrite it with `--device auto` or change that value.

```sh
uv run ldpc-torch --mode base --device cuda --output-directory runs/gpu-base
uv run ldpc-torch --mode post --write-config my-post.json
```

## Compared to the original implementation

| Origin | This | Content |
|---|---|---|
| `BaseGraph/` | `base_graphs/` | Base graph and protograph matrices |
| `Inputs/` | `uncorrected_samples/` | Training, validation, and test failure vectors |
| `Weights/` | `pretrained_weights/` | Trained weights for base/post experiments |
| `Results/` | `reference_results/` | Reference results for comparison and validation |


- The VN-first flooding schedule, per-edge cyclic-shift direction, UCN's previous-APP decision, channel/check weight placement, and clipping order are preserved.
- The original MS handling of `0→0.0001→0`, the `10000` sentinel, and replacement of connected zeros with the multiplication identity in SP are preserved. SP puncturing uses the original generated value `+0.001`.
- 6/5/-5/4/3-bit quantization, ties-to-even rounding, straight-through gradients including clipping boundaries, and the FER sigmoid surrogate are preserved.
- `LegacyAdam` preserves the TensorFlow v1 epsilon placement, which differs from PyTorch's default Adam. Only trainable weights are projected into range after each update.
- `frame_error_rate` counts a frame as successful if it is correct at any iteration; `frame_error_rate_last` checks only the final iteration. The default BER preserves the original signed-error cancellation and full-label-length denominator. For conventional BER on nonzero codewords, use `error_metrics(..., legacy_bit_error_rate=False)`.
- With the default `legacy_code_rate=true`, code-rate calculation preserves the original behavior of counting each inactive `(0,0)` range as one bit. For example, the WiMAX default is `431/574`; `false` uses the proper `432/576`. Changing this compatibility setting also changes the noise variance at the same SNR.
- AWGN generation preserves NumPy RandomState and the original operation order to reproduce samples from existing seeds. The model, loss, gradients, and optimizer use PyTorch. Random information bits can be generated with `AWGNSampler.sample(..., generator_matrix=...)`; the default experiments use all-zero codewords.
- Previously non-runnable or missing behavior is implemented for node iteration sharing (5), UCN iteration sharing, block transfer without validation, and one-line failure files. Incorrectly reading rows from files with different weight groups is not reproduced; missing files and configuration errors raise exceptions.
- Dense `(E·Z)²` lifting matrices and `[B,Z,E,E]` check tensors are not created. Neighbor gather and cyclic indices perform the same operations. Full training histories are not guaranteed to be bitwise identical across all environments because floating-point summation order and RNG initialization implementations may differ.

### Configuration

Copy [base.json](configs/base.json), [post.json](configs/post.json), [collect.json](configs/collect.json) and modify them. The `decoder` and `experiment` sections in the configuration files override the default values. The files specify mode-specific settings but omit `data_directory`, which defaults to the repository root. `--mode` selects defaults before the JSON overrides are applied; it does not override `sampling_mode` from JSON.

| Origin | This | Content |
|---|---|---|
| `z_value` | `lifting_factor` | Lifting factor |
| `iters_max` | `iterations` | Maximum number of decoding iterations |
| `fixed_iter` | `frozen_iterations` | Number of initial decoding iterations whose weights remain frozen |
| `iter_step` | `iteration_block_size` | Number of iterations per block |
| `fixed_init` | `retrain_previous_iterations` | Number of previous iterations to retrain with each new block, excluding the frozen prefix |
| `sharing[0]` | `check_weight_sharing` | Check weight sharing |
| `sharing[1]` | `unsatisfied_weight_sharing` | Unsatisfied-check weight sharing |
| `sharing[2]` | `channel_weight_sharing` | Channel weight sharing |
| `etha_*` | `loss_decay`, `loss_decay_multiplier`, `loss_decay_interval` | Loss decay parameters |
| `learn_rate_*` | `learning_rate`, `learning_rate_multiplier`, `learning_rate_interval` | Learning rate decay parameters |
| `SNR_Matrix` | `signal_to_noise_ratios_db` | SNR values in dB |
| `sampling_type` | `sampling_mode`: `awgn`, `uncorrected`, `collect` | Sampling mode |
| `decoding_type` | `decoding_algorithm`: `sum_product`, `min_sum`, `quantized_min_sum`, `legacy_min_sum` | Decoding algorithm |
| `loss_type` | `loss_function`: `binary_cross_entropy`, `soft_bit_error`, `frame_error` | Loss function |
| `opt_result_print` | `selection_metric`: `bit_error_rate_last`, `frame_error_rate_last`, `frame_error_rate`, `loss` | Selection metric for best weights |
| `init_from_file` | `initial_weights` | Initial weights for new training iterations |
| Previous Opt Weight files | `frozen_weights` | Weights from previous optimization |

Weight sharing modes are `0=none`, `1=edge/iteration`, `2=node/iteration`, `3=iteration scalar`, `4=edge/iteration sharing`, and `5=node/iteration sharing`. Iteration sharing reuses the same parameters in subsequent iterations after `frozen_iterations`. Channel sharing supports `0,2,3,5`. UCN must use mode 0 or the same sharing mode as check nodes. `initial_*_weight=-1` uses truncated-normal initialization.

Puncturing and shortening use 1-based, inclusive ranges. `(0,0)` disables them; see below for the original-compatible SNR code-rate behavior. With `systematic_only=true`, only the first `N_proto-M_proto` variable nodes are used for losses and metrics. The number of batches is floored as in the original, so leftover samples are discarded. Evaluation requires at least one batch. When `validation_samples=0`, the final epoch's weights are passed to the next block. For AWGN evaluation, `validation_samples` and `test_samples` are counts per SNR. Uncorrected-data evaluation uses stored LLRs directly and does not regenerate noise for the configured SNRs. The trainable iteration count (`iterations - frozen_iterations`) must be divisible by the block size.

## Process

1. Get `best_weights_end20.txt` from the base training run.
2. Set its path as `initial_weights` in `configs/collect.json`, choose the SNR and `validation_samples`, and run collection.
3. Split collected `Uncor.txt` rows into disjoint train/validation/test files, placing them at `uncorrected_samples/[Uncor]_<graph_name>.txt`, `_Valid.txt`, and `_Test.txt`.
4. Set the base weight path as `frozen_weights` in the post configuration and run it.

```sh
uv run main.py --config configs/collect.json
uv run main.py --config configs/post.json
```

Collection defaults to `initial_weights=null` and does not automatically load base results. For separate collection runs, use different seeds and output directories: rerunning with the same seed repeats the candidate stream and can append duplicate data. The default post configuration requires 10,000 training rows and 5,000 rows each for validation and test; adjust these counts or collect more candidates as needed.

Collection evaluates `floor(validation_samples / batch_size) * batch_size` candidate frames; it does not run indefinitely until a target number of failed frames is gathered. As in the original, `Uncor.txt` records only frames that fail at every decoding iteration. Running collection again in the same output directory appends to the file. The file contains three metadata columns followed by sign-inverted LLRs, preserving the original one-decimal-place format. No new `Uncor.txt` is created when no frames fail.


The provided default post run automatically uses `pretrained_weights/C0_wman_N0576_R34_z24_Opt_Weight_End20.txt`. Set `frozen_weights` to use a base model trained directly by you. Weight files are read by interpreting their sharing modes and groups. If the base file has no UCN weights, a new UCN group is initialized from the base check weights. Scalar-to-node/edge and check-node-to-edge expansions are supported; arbitrary reductions and incompatible shapes raise an error.

## Python API

```python
import torch
from ldpc_torch import DecoderConfig, Protograph, NeuralLDPCDecoder, decoding_loss

config = DecoderConfig(iterations=20, lifting_factor=24)
graph = Protograph.from_file('base_graphs/wman_N0576_R34_z24.txt', 24)
decoder = NeuralLDPCDecoder(graph, config)
channel_llrs = torch.randn(4, graph.variable_nodes, 24)
output = decoder(channel_llrs)
labels = torch.zeros(4, graph.variable_nodes * 24)
loss = decoding_loss(output.target_llrs, labels, loss_function='frame_error')
loss.backward()
```

Input LLRs are $\log(P(1)/P(0))$. Negative values are decoded as bit 0, and values greater than or equal to zero as bit 1. `posterior_llrs` and `target_llrs` have shape `[iteration, batch, bit]`, while `check_messages` has shape `[batch, check-major edge, lifting]`. As in the original, each iteration's posterior adds check messages to the unweighted channel LLR (quantized in QMS), then clips the result to `[-llr_clip, llr_clip]`.

`save_checkpoint` / `load_checkpoint` save and restore configuration, graph, and weights in `.pt` format. These checkpoints are intended for inference and weight reuse; resuming from an interruption with optimizer/RNG state is not supported. Use `save_legacy_weights` / `load_legacy_weights` for the existing text format.

## Module Layout

| File | Role |
|---|---|
| `config.py` | Explicit decoder/experiment settings and validation |
| `graph.py` | Graph connections, cyclic-shift indices, and neighbor lists |
| `decoder.py` | SP/MS/QMS decoding and weight sharing |
| `quantization.py` | Quantization and the original surrogate gradient |
| `data.py` | AWGN, failure-vector I/O, and code rate |
| `losses.py`, `metrics.py` | Per-iteration losses, BER, and FER |
| `optimizer.py` | PyTorch implementation of the TensorFlow v1 Adam equations |
| `weights.py` | Existing text format and native checkpoints |
| `training.py`, `cli.py` | Block training, evaluation, collection, and execution settings |

## Verification

```sh
uv run python -m pytest -q
# To include direct comparisons with the original TensorFlow implementation:
uv sync --extra reference
uv run --extra reference python -m pytest -q
```
### Mixed Codeword Generalization

This script requires `runs/base/best_decoder_end20.pt` and `runs/post/best_decoder_end30.pt` from base/post runs with the default iteration counts. Smoke runs create checkpoints with different iteration counts and do not satisfy this prerequisite.

```sh
uv run --extra reference python scripts/verify_mixed_codewords.py \
  --frames-per-snr 1000 --reference-frames-per-snr 100 \
  --report runs/full_workflow_audit/mixed_codewords_mps.json
```

Compare the post-training and failure-collection workflow against the original with `scripts/verify_post_workflow.py`. See [Full Workflow and Mixed Codeword Verification](docs/full_workflow_and_mixed_codewords.md) for detailed results and an explanation of the original weight loading and `LLR=0` asymmetry. For mixed labels, use standard BER (`legacy_bit_error_rate=False`) and BCE that reflects the labels. The FER and soft-BER surrogates are intended only for all-zero codewords.
