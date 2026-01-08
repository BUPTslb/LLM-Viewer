from easydict import EasyDict

model_params = {
    "Mistral-7B-Instruct-v0.1": EasyDict(
        hidden_size=4096,
        intermediate_size=14336,
        num_attention_heads=32,
        num_hidden_layers=32,
        num_key_value_heads=8,
        vocab_size=32000,
    )
}
