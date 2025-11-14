from easydict import EasyDict

model_params = {
    "zai-org/chatglm2-6b": EasyDict(
        hidden_size=4096,
        ffn_hidden_size=13696,
        num_layers=28,
        num_attention_heads=32,
        multi_query_attention=True,
        multi_query_group_num=2,
        padded_vocab_size=65024,
    ),
    "chatglm2-6b": EasyDict(
        hidden_size=4096,
        ffn_hidden_size=13696,
        num_layers=28,
        num_attention_heads=32,
        multi_query_attention=True,
        multi_query_group_num=2,
        padded_vocab_size=65024,
    )
}
