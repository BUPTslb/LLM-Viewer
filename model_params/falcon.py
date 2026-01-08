from easydict import EasyDict

model_params = {
    "tiiuae/falcon-11b": EasyDict(
        hidden_size=4544,
        num_attention_heads=71,
        num_hidden_layers=32,
        multi_query=True,
        new_decoder_architecture=False,
        vocab_size=65024,
    ),
    "tiiuae/falcon-40b": EasyDict(
        hidden_size=8192,
        num_attention_heads=128,
        num_hidden_layers=60,
        num_kv_heads=8,
        new_decoder_architecture=True,
        vocab_size=65024,
    )
}
