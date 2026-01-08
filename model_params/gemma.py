from easydict import EasyDict

model_params = {
    "gemma-2-2b": EasyDict(
        hidden_size=2304,
        intermediate_size=9216,
        num_attention_heads=8,
        num_hidden_layers=26,
        num_key_value_heads=4,
        vocab_size=256000,
    )
}
