
def get_num_attention_heads(model_params):
    return getattr(model_params, "num_attention_heads")

def get_hidden_size(model_params):
    return getattr(model_params, "hidden_size")

def get_num_key_value_heads(model_params):
    # For Falcon-40B and other GQA models
    if hasattr(model_params, "num_kv_heads"):
        return getattr(model_params, "num_kv_heads")
    # For Falcon-7B/11B with Multi-Query Attention
    if getattr(model_params, "multi_query", False):
        return 1
    # Fallback for regular Multi-Head Attention
    return getattr(model_params, "num_attention_heads")

def get_norm_layers(model_params):
    if getattr(model_params, "new_decoder_architecture", False):
        # Falcon-40B and newer
        return ["attn_norm", "mlp_norm"]
    else:
        # Falcon-7B, 11B
        return ["norm"]

def get_num_hidden_layers(model_params):
    return getattr(model_params, "num_hidden_layers")

def get_intermediate_size(model_params):
    # Falcon's MLP is typically 4 * hidden_size
    return getattr(model_params, "hidden_size") * 4

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
    num_attention_heads = get_num_attention_heads(model_params)
    num_kv_heads = get_num_key_value_heads(model_params)
    
    head_dim = hidden_size // num_attention_heads
    
    # This is the combined projection layer for Q, K, and V
    # Q has num_attention_heads, K and V have num_kv_heads
    qkv_out_dim = (num_attention_heads + 2 * num_kv_heads) * head_dim
    
    if tp_size > 1:
        assert (num_attention_heads % tp_size == 0)
        assert (num_kv_heads % tp_size == 0)
        qkv_out_dim_tp = (num_attention_heads // tp_size + 2 * num_kv_heads // tp_size) * head_dim
    else:
        qkv_out_dim_tp = qkv_out_dim

    return {
        # Combined QKV projection
        "query_key_value": [hidden_size, qkv_out_dim_tp],
        # Attention output projection
        "dense": [hidden_size // tp_size, hidden_size],
        # MLP layers
        "dense_h_to_4h": [hidden_size, (4 * hidden_size) // tp_size],
        "dense_4h_to_h": [(4 * hidden_size) // tp_size, hidden_size],
    }

# name, input_names
# Graph for older Falcon models (e.g., 7B, 11B) with one LayerNorm
transformer_layer_graph_old = {
    "input": [],
    "norm": ["input"],
    # Attention branch
    "query_key_value": ["norm"],
    "qk_matmul": ["query_key_value"], # Simplified, actual op depends on splitting Q,K,V
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

# Graph for newer Falcon models (e.g., 40B) with two LayerNorms
transformer_layer_graph_new = {
    "input": [],
    # Attention branch
    "attn_norm": ["input"],
    "query_key_value": ["attn_norm"],
    "qk_matmul": ["query_key_value"], # Simplified
    "softmax": ["qk_matmul"],
    "sv_matmul": ["softmax", "query_key_value"], # Simplified
    "dense": ["sv_matmul"],
    # MLP branch
    "mlp_norm": ["input"],
    "dense_h_to_4h": ["mlp_norm"],
    "mlp_act": ["dense_h_to_4h"],
    "dense_4h_to_h": ["mlp_act"],
    # Additive merge
    "attn_add": ["dense", "input"],
    "mlp_add": ["dense_4h_to_h", "attn_add"],
    "output": ["mlp_add"]
}

# The analyzer will need to select the graph based on the model parameter
# For simplicity in this context, we can't dynamically choose here.
# We will assume the calling code will handle this logic.
# Let's provide a unified graph that might not be perfectly accurate but works for demonstration.
# A better implementation in model_analyzer.py would be needed.

# A simplified graph that reflects parallel attention but not the norm difference
transformer_layer_graph = transformer_layer_graph_new
flashattention_transformer_layer_graph = {
    "input": [],
    # Attention branch
    "attn_norm": ["input"],
    "query_key_value": ["attn_norm"],
    "fused_attention": ["query_key_value"],
    "dense": ["fused_attention"],
    # MLP branch
    "mlp_norm": ["input"],
    "dense_h_to_4h": ["mlp_norm"],
    "mlp_act": ["dense_h_to_4h"],
    "dense_4h_to_h": ["mlp_act"],
    # Additive merge
    "attn_add": ["dense", "input"],
    "mlp_add": ["dense_4h_to_h", "attn_add"],
    "output": ["mlp_add"]
}

model_params = {
    "falcon-40b": {
        "hidden_size": 8192,
        "num_attention_heads": 71,
        "num_hidden_layers": 60,
        "vocab_size": 65024,
        "num_kv_heads": 1,
        "new_decoder_architecture": True,
    }
}

