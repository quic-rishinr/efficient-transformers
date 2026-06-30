# Qwen3.5-397B-A17B 2-layer disagg order comparison

Raw run root: `/home/rishinr/qwen3_5_397b_2layer_order_compare_20260626_024219`

## Graphs
- `qwen3_5_397b_2layer_prefill_first_timeline.png`
- `qwen3_5_397b_2layer_decode_first_timeline.png`
- `qwen3_5_397b_2layer_order_comparison.png`

## Overall peaks
- Previous order: vision → prefill → decode: 285.105 GB at 110.05s; status=failed
- New order: vision → decode → prefill: 285.036 GB at 109.24s; status=failed

## Notes
- Both runs continued past the known prefill compile QAIC VA-space failure using `--continue-on-compile-error`.
- RSS is sampled from the Python parent process plus child processes at 50 ms cadence.
