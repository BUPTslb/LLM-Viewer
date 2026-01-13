"""Local (non-HuggingFace) GPT-family model params.

The project supports two ways of providing model shapes:
- HuggingFace: `source=huggingface` (default) uses `AutoConfig.from_pretrained()`.
- Local:        `--source <name>` loads `model_params/<name>.py` and looks up
				`model_params[model_id]`.

This file intentionally hard-codes GPT-OSS parameters so it does NOT depend on
any external JSON files.
"""

from easydict import EasyDict


# Add/adjust keys as you need.
_gpt_oss_default = EasyDict(
	{
		"architectures": ["GptOssForCausalLM"],
		"attention_bias": True,
		"attention_dropout": 0.0,
		"eos_token_id": 200002,
		"experts_per_token": 4,
		"head_dim": 64,
		"hidden_act": "silu",
		"hidden_size": 2880,
		"initial_context_length": 4096,
		"initializer_range": 0.02,
		"intermediate_size": 2880,
		"layer_types": [
			"sliding_attention",
			"full_attention",
			"sliding_attention",
			"full_attention",
			"sliding_attention",
			"full_attention",
			"sliding_attention",
			"full_attention",
			"sliding_attention",
			"full_attention",
			"sliding_attention",
			"full_attention",
			"sliding_attention",
			"full_attention",
			"sliding_attention",
			"full_attention",
			"sliding_attention",
			"full_attention",
			"sliding_attention",
			"full_attention",
			"sliding_attention",
			"full_attention",
			"sliding_attention",
			"full_attention",
			"sliding_attention",
			"full_attention",
			"sliding_attention",
			"full_attention",
			"sliding_attention",
			"full_attention",
			"sliding_attention",
			"full_attention",
			"sliding_attention",
			"full_attention",
			"sliding_attention",
			"full_attention",
		],
		"max_position_embeddings": 131072,
		"model_type": "gpt_oss",
		"num_attention_heads": 64,
		"num_experts_per_tok": 4,
		"num_hidden_layers": 36,
		"num_key_value_heads": 8,
		"num_local_experts": 128,
		"output_router_logits": False,
		"pad_token_id": 199999,
		"quantization_config": {
			"modules_to_not_convert": [
				"model.layers.*.self_attn",
				"model.layers.*.mlp.router",
				"model.embed_tokens",
				"lm_head",
			],
			"quant_method": "mxfp4",
		},
		"rms_norm_eps": 1e-05,
		"rope_scaling": {
			"beta_fast": 32.0,
			"beta_slow": 1.0,
			"factor": 32.0,
			"original_max_position_embeddings": 4096,
			"rope_type": "yarn",
			"truncate": False,
		},
		"rope_theta": 150000,
		"router_aux_loss_coef": 0.9,
		"sliding_window": 128,
		"swiglu_limit": 7.0,
		"tie_word_embeddings": False,
		"transformers_version": "4.55.0.dev0",
		"use_cache": True,
		"vocab_size": 201088,
	}
)


model_params = {
	# Common id used in your CLI example.
	"openai/gpt-oss-120b": _gpt_oss_default,
	# Optional aliases for convenience.
	"gpt-oss-120b": _gpt_oss_default,
}

