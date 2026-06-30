# Qwen3-VL 235B Full-Layer Disagg Profiling Result

- Status: failed
- Model: Qwen/Qwen3-VL-235B-A22B-Instruct
- Resolved text layers: 94
- Resolved vision depth: 27
- Requested path: vision compile, prefill compile, decode compile, runtime decode steps=8
- Completed path: vision compile and full 94-layer prefill export/merge; failed during prefill qaic-compile before decode/runtime.
- Peak RSS: 50.94 GB at 79.06 elapsed minutes
- Peak operation: prefill: onnx.save
- Failure: RuntimeError("Compilation failed!\nCompiler command: ['/opt/qti-aic/exec/qaic-compile', '-aic-hw', '-aic-hw-version=ai100', '-m=/home/rishinr/qwen3_vl_235b_full_layer_disagg_e2e_final_20260626_070152/qeff_home/Qwen3VLMoeForConditionalGeneration/Qwen3VLDecoderWrapper-18e78fa9cf00b85f/merged_0-94.onnx', '-retained-state', '-convert-to-fp16', '-mxfp6-matmul', '-aic-num-cores=16', '-split-model-io', '-mos=1', '-aic-enable-depth-first', '-sub-functions', '-network-specialization-config=/home/rishinr/qwen3_vl_235b_full_layer_disagg_e2e_final_20260626...

## Phase Peaks
- vision_export: 12.38 GB peak, 0.05 to 0.42 elapsed minutes
- vision_compile: 11.50 GB peak, 0.64 to 0.92 elapsed minutes
- prefill_layerwise_export: 9.91 GB peak, 0.99 to 66.26 elapsed minutes
- prefill_merge_save: 50.94 GB peak, 66.26 to 104.46 elapsed minutes
- prefill_compile_failed: 1.51 GB peak, 104.46 to 104.46 elapsed minutes

## Artifacts
- Annotated graph: profiling_outputs/qwen3_vl_235b_full_layer_disagg_e2e/qwen3_vl_235b_full_layer_disagg_e2e_memory_timeline.png
- Raw profiler graph: profiling_outputs/qwen3_vl_235b_full_layer_disagg_e2e/raw_memory_timeline.png
- Summary JSON: profiling_outputs/qwen3_vl_235b_full_layer_disagg_e2e/profile_summary.json
- Phase CSV: profiling_outputs/qwen3_vl_235b_full_layer_disagg_e2e/component_phases.csv
- Sample CSV: profiling_outputs/qwen3_vl_235b_full_layer_disagg_e2e/samples_timeseries.csv
- Run log: profiling_outputs/qwen3_vl_235b_full_layer_disagg_e2e/run.log
