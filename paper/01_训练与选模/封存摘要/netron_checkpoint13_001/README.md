# BG128 checkpoint 13 Netron model

Open `BG128_checkpoint13_actor.onnx` in Netron.

This ONNX contains the selected checkpoint's token encoder, both graph
propagation layers, readout, C128 encoders, scorer, and legal-action mask. It
has a single-observation batch dimension. The edge count is dynamic.

The ONNX does **not** build the business graph or derive C128 from the raw
observation. Callers must provide `states_1519`, `c128`, and the graph tensors
(`edge_index`, `edge_type`, `edge_group`, `edge_attributes_14`,
`node_quantities_19x21`, `candidate_dump_ids`, `ego_ids`). The reference
preprocessing is `business_graph_summary_v1.build_graph` and
`single_critic_task_query_v1.c_features.build_features`, using the frozen
BG128 feature configuration. The output is four masked action logits; illegal
actions are negative infinity.

`export_manifest.json` records the source checkpoint and artifact hashes.
Export validation checked ONNX structure with `onnx.checker` and compared
ONNX Runtime CPU outputs against the original PyTorch Actor for two real
observations with different edge counts.
