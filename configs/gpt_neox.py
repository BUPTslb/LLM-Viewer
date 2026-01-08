
def get_num_attention_heads(model_params):
    return getattr(model_params, "num_attention_heads")

def get_hidden_size(model_params):
    return getattr(model_params, "hidden_size")

def get_num_key_value_heads(model_params):
    # GPT-NeoX uses standard Multi-Head Attention
    return getattr(model_params, "num_attention_heads")

def get_norm_layers(model_params):
    # Single LayerNorm before the parallel blocks
    return ["norm"]

def get_num_hidden_layers(model_params):
    return getattr(model_params, "num_hidden_layers")

def get_intermediate_size(model_params):
    # Can be specified directly or is 4 * hidden_size
    return getattr(model_params, "intermediate_size", getattr(model_params, "hidden_size") * 4)

def get_vocab_size(model_params):
    return getattr(model_params, "vocab_size")

def post_process(model_params,args):
    hiddensize=get_hidden_size(model_params)
    vocab_size=get_vocab_size(model_params)
    layers=[]
    for stage in ["prefill", "decode"]:
        layers.append({
            'name': 'lm_head',
            'stage':stage,
            'OPs':args['batchsize']*hiddensize*vocab_size*1,
            'load_weight':hiddensize*vocab_size *args['w_byte'],
            'load_act':hiddensize*args['a_byte'],
            'store_act':vocab_size*args['a_byte'],
        })
    return layers

def get_linear_layers(model_params, tp_size: int):
    hidden_size = get_hidden_size(model_params)
    intermediate_size = get_intermediate_size(model_params)
    
    if tp_size > 1:
        assert hidden_size % tp_size == 0
        assert intermediate_size % tp_size == 0

    return {
        # Combined QKV projection for MHA
        "query_key_value": [hidden_size, 3 * (hidden_size // tp_size)],
        # Attention output projection
        "dense": [hidden_size // tp_size, hidden_size],
        # MLP layers
        "dense_h_to_4h": [hidden_size, intermediate_size // tp_size],
        "dense_4h_to_h": [intermediate_size // tp_size, hidden_size],
    }

# Computation graph for GPT-NeoX
transformer_layer_graph = {
    "input": [],
    "norm": ["input"],
    # Attention branch
    "query_key_value": ["norm"],
    "qk_matmul": ["query_key_value"], # Simplified
    "softmax": ["qk_matmul"],
    "sv_matmul": ["softmax", "query_key_value"], # Simplified
    "dense": ["sv_matmul"],
    # MLP branch
    "dense_h_to_4h": ["norm"],
    "mlp_act": ["dense_h_to_4h"],
    "dense_4h_to_h": ["mlp_act"],
    # Additive merge
    "attn_add": ["dense", "input"],
    "mlp_add": ["dense_4h_to_h", "attn_add"],
    "output": ["mlp_add"]
}

flashattention_transformer_layer_graph = {
    "input": [],
    "norm": ["input"],
    # Attention branch
    "query_key_value": ["norm"],
    "fused_attention": ["query_key_value"],
    "dense": ["fused_attention"],
    # MLP branch
    "dense_h_to_4h": ["norm"],
    "mlp_act": ["dense_h_to_4h"],
    "dense_4h_to_h": ["mlp_act"],
    # Additive merge
    "attn_add": ["dense", "input"],
    "mlp_add": ["dense_4h_to_h", "attn_add"],
    "output": ["mlp_add"]
}
