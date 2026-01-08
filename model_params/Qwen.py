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
    )
}
