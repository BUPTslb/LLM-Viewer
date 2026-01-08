from easydict import EasyDict

model_params = {
    "gpt-neox-20b": EasyDict(
        hidden_size=6144,
        intermediate_size=24576,
        num_attention_heads=64,
        num_hidden_layers=44,
        vocab_size=50432,
    )
}
