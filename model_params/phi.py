from easydict import EasyDict

model_params = {
    "phi-2": EasyDict(
        n_embd=2560,
        n_head=32,
        n_layer=32,
        vocab_size=51200,
    )
}
