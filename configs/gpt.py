"""GPT-family config adapter.

This file is used by `ModelAnalyzer` to extract model dimensions from a
`transformers` config object (AutoConfig) and to define a per-layer compute graph.

It is primarily intended to support OpenAI GPT-OSS style configs (model_type
`gpt_oss`) which use GQA (num_key_value_heads) and MoE (num_local_experts,
num_experts_per_tok).
"""


def get_num_attention_heads(model_params):
	return getattr(model_params, "num_attention_heads")


def get_hidden_size(model_params):
	return getattr(model_params, "hidden_size")


def get_num_key_value_heads(model_params):
	# GQA / MQA when present; fallback to MHA.
	return getattr(model_params, "num_key_value_heads", getattr(model_params, "num_attention_heads"))


def get_norm_layers(model_params):
	# GPT-OSS style blocks typically have two norms like Llama.
	return ["attn_norm", "mlp_norm"]


def get_num_hidden_layers(model_params):
	return getattr(model_params, "num_hidden_layers")


def get_intermediate_size(model_params):
	# Some GPT variants use `intermediate_size`; keep a safe fallback.
	hidden_size = get_hidden_size(model_params)
	return getattr(model_params, "intermediate_size", hidden_size * 4)


def get_vocab_size(model_params):
	return getattr(model_params, "vocab_size")


def _get_experts_per_token(model_params) -> int:
	return int(getattr(model_params, "num_experts_per_tok", getattr(model_params, "experts_per_token", 1)))


def _get_num_local_experts(model_params) -> int:
	return int(getattr(model_params, "num_local_experts", 0))


def post_process(model_params, args):
	hidden_size = get_hidden_size(model_params)
	vocab_size = get_vocab_size(model_params)
	layers = []
	for stage in ["prefill", "decode"]:
		layers.append(
			{
				"name": "lm_head",
				"stage": stage,
				"OPs": args["batchsize"] * hidden_size * vocab_size * 1,
				"load_weight": hidden_size * vocab_size * args["w_byte"],
				"load_act": hidden_size * args["a_byte"],
				"store_act": vocab_size * args["a_byte"],
			}
		)
	return layers


def get_linear_layers(model_params, tp_size: int):
	"""Return linear layers as {name: [in_features, out_features]}.

	Notes for MoE:
	- We model expert MLP projections with an *effective* multiplier of
	  `num_experts_per_tok` by scaling dimensions. This keeps the rest of the
	  analyzer unchanged while reflecting MoE compute/memory scaling.
	"""

	hidden_size = get_hidden_size(model_params)
	intermediate_size = get_intermediate_size(model_params)
	key_value_heads = get_num_key_value_heads(model_params)
	attention_heads = get_num_attention_heads(model_params)

	experts_per_tok = _get_experts_per_token(model_params)
	num_local_experts = _get_num_local_experts(model_params)

	if tp_size > 1:
		assert hidden_size % tp_size == 0
		assert intermediate_size % tp_size == 0
		# Following the pattern in other configs: shard KV heads across TP.
		if key_value_heads != attention_heads:
			assert key_value_heads % tp_size == 0

	# GQA: K/V projection output dim equals hidden_size * (n_kv / n_q)
	kv_out = hidden_size * key_value_heads // attention_heads

	layers = {
		"q_proj": [hidden_size, hidden_size // tp_size],
		"k_proj": [hidden_size, kv_out // tp_size],
		"v_proj": [hidden_size, kv_out // tp_size],
		"out_proj": [hidden_size // tp_size, hidden_size],
	}

	# Router / gating (MoE). If config doesn't expose MoE fields, we fall back
	# to dense MLP (no router).
	if num_local_experts and experts_per_tok > 0:
		layers["router"] = [hidden_size, num_local_experts]
		# Effective MoE compute: scale by experts_per_tok.
		layers["gate_proj"] = [hidden_size * experts_per_tok, intermediate_size // tp_size]
		layers["up_proj"] = [hidden_size * experts_per_tok, intermediate_size // tp_size]
		layers["down_proj"] = [intermediate_size * experts_per_tok // tp_size, hidden_size]
	else:
		# Dense fallback (Llama-like SwiGLU)
		layers["gate_proj"] = [hidden_size, intermediate_size // tp_size]
		layers["up_proj"] = [hidden_size, intermediate_size // tp_size]
		layers["down_proj"] = [intermediate_size // tp_size, hidden_size]

	return layers


def get_total_weight_bytes(model_params, tp_size: int, w_byte: float) -> float:
	"""Total *stored* weight bytes (per TP rank).

	This differs from per-inference `load_weight`:
	- For MoE, inference only activates `num_experts_per_tok` experts, but
	  Flash capacity must store *all* `num_local_experts` experts.
	- lm_head / embedding are counted once (not multiplied by num_hidden_layers).
	"""

	hidden_size = get_hidden_size(model_params)
	intermediate_size = get_intermediate_size(model_params)
	num_layers = get_num_hidden_layers(model_params)
	attention_heads = get_num_attention_heads(model_params)
	key_value_heads = get_num_key_value_heads(model_params)
	vocab_size = get_vocab_size(model_params)

	if tp_size > 1:
		assert hidden_size % tp_size == 0
		assert intermediate_size % tp_size == 0
		if key_value_heads != attention_heads:
			assert key_value_heads % tp_size == 0

	kv_out = hidden_size * key_value_heads // attention_heads
	expert_intermediate_shard = intermediate_size // tp_size
	hidden_shard = hidden_size // tp_size

	# Attention weights per transformer block (per TP rank).
	attn_weights = 0
	attn_weights += hidden_size * hidden_shard  # q_proj
	attn_weights += hidden_size * (kv_out // tp_size)  # k_proj
	attn_weights += hidden_size * (kv_out // tp_size)  # v_proj
	attn_weights += hidden_shard * hidden_size  # out_proj

	num_local_experts = _get_num_local_experts(model_params)
	experts_per_tok = _get_experts_per_token(model_params)

	# Router weights per block (not TP-sharded here; small relative to experts).
	router_weights = 0
	if num_local_experts and experts_per_tok > 0:
		router_weights = hidden_size * num_local_experts

	# MLP expert weights per block.
	# SwiGLU: gate_proj + up_proj (hidden->intermediate) and down_proj (intermediate->hidden)
	dense_mlp_weights = 3 * hidden_size * expert_intermediate_shard
	if num_local_experts and experts_per_tok > 0:
		mlp_weights = num_local_experts * dense_mlp_weights
	else:
		mlp_weights = dense_mlp_weights

	transformer_weights = num_layers * (attn_weights + router_weights + mlp_weights)

	# Token embeddings + lm_head.
	# If tied, count once; otherwise count both.
	tied = bool(getattr(model_params, "tie_word_embeddings", False))
	embed = vocab_size * hidden_size
	if tied:
		embed_and_head = embed
	else:
		embed_and_head = 2 * embed

	total_params = transformer_weights + embed_and_head
	return total_params * w_byte


# name, input_names
transformer_layer_graph = {
	"input": [],
	"attn_norm": ["input"],
	"q_proj": ["attn_norm"],
	"k_proj": ["attn_norm"],
	"v_proj": ["attn_norm"],
	"qk_matmul": ["q_proj", "k_proj"],
	"softmax": ["qk_matmul"],
	"sv_matmul": ["softmax", "v_proj"],
	"out_proj": ["sv_matmul"],
	"attn_add": ["input", "out_proj"],
	"mlp_norm": ["attn_add"],
	"router": ["mlp_norm"],
	"gate_proj": ["mlp_norm"],
	"up_proj": ["mlp_norm"],
	"mlp_act": ["up_proj", "gate_proj"],
	"down_proj": ["mlp_act"],
	"mlp_add": ["attn_add", "down_proj"],
	"output": ["mlp_add"],
}


flashattention_transformer_layer_graph = {
	"input": [],
	"attn_norm": ["input"],
	"q_proj": ["attn_norm"],
	"k_proj": ["attn_norm"],
	"v_proj": ["attn_norm"],
	"fused_attention": ["q_proj", "k_proj", "v_proj"],
	"out_proj": ["fused_attention"],
	"attn_add": ["input", "out_proj"],
	"mlp_norm": ["attn_add"],
	"router": ["mlp_norm"],
	"gate_proj": ["mlp_norm"],
	"up_proj": ["mlp_norm"],
	"mlp_act": ["up_proj", "gate_proj"],
	"down_proj": ["mlp_act"],
	"mlp_add": ["attn_add", "down_proj"],
	"output": ["mlp_add"],
}

