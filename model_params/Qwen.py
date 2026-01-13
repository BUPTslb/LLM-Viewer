from easydict import EasyDict

model_params = {
    "Qwen/Qwen1.5-1.8B-Chat": EasyDict(
        hidden_size=2048,
        intermediate_size=5504,
        num_attention_heads=16,
        num_hidden_layers=24,
        num_key_value_heads=16,
        vocab_size=151936,
    ),
    "Qwen/Qwen1.5-4B-Chat": EasyDict(
        hidden_size=2560,
        intermediate_size=13696,
        num_attention_heads=20,
        num_hidden_layers=28,
        num_key_value_heads=4,
        vocab_size=151936,
    ),
    "Qwen/Qwen3-32B": EasyDict(
        hidden_size=5120,
        intermediate_size=25600,
        num_attention_heads=64,
        num_hidden_layers=64,
        num_key_value_heads=8,
        head_dim=128,
        hidden_act="silu",
        vocab_size=151936,
        max_position_embeddings=40960,
        rms_norm_eps=1e-6,
        rope_theta=1000000,
        attention_dropout=0.0,
        attention_bias=False,
        tie_word_embeddings=False,
        use_cache=True,
        torch_dtype="bfloat16",
    )
}
