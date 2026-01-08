import os
import importlib
from hardwares.hardware_params import hardware_params
from roofline_model import roofline_analyze
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from utils import str_number, str_number_time
import math

ALL_DATA_NAMES = [
    "OPs",
    "memory_access",
    "load_weight",
    "load_act",
    "store_act",
    "load_kv_cache",
    "store_kv_cache",
    "inference_time",
]


class ModelAnalyzer:
    def __init__(self, model_id, hardware, config_file=None, source="huggingface", weight_storage: str = "dram"):
        self.model_id = model_id
        self.hardware = hardware

        if weight_storage not in {"dram", "flash"}:
            raise ValueError(f"Unsupported weight_storage='{weight_storage}', expected 'dram' or 'flash'.")
        self.weight_storage = weight_storage
        if config_file is None:
            # get the current file directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # auto search the config
            for file in os.listdir(current_dir + "/configs"):
                if file.endswith(".py") and file.replace(".py", "") in model_id:
                    config_file = "configs/" + file
                # print(f"auto search config file {config_file} {file} {model_id}")
        assert config_file is not None, "config file is not found, please specify it manually."
        print(f"use config file {config_file} for {model_id}")
        if "Qwen" in model_id:
            source = "Qwen"
            config_file = "configs/Qwen.py"
            print(f"Detected Qwen model, forcing source to '{source}' and config_file to '{config_file}'")
        elif "Mistral" in model_id:
            source = "Mistral"
            config_file = "configs/Mistral.py"
            print(f"Detected Mistral model, forcing source to '{source}' and config_file to '{config_file}'")
        elif "chatglm" in model_id:
            source = "chatglm"
            config_file = "configs/chatglm3.py" # Assuming chatglm3.py is the correct config for chatglm2-6b
            print(f"Detected ChatGLM model, forcing source to '{source}' and config_file to '{config_file}'")
        
        print(f"DEBUG: Initial source: {source}")
        print(f"DEBUG: Initial config_file: {config_file}")

        if source == "huggingface":
            # Check if it's a Qwen model that should use the custom source
            if "Qwen" in model_id:
                # Load from custom model_params/Qwen.py
                if not os.path.exists(f"model_params/Qwen.py"):
                    raise Exception(f"model_params/Qwen.py is not found")
                module = importlib.import_module(f"model_params.Qwen")
                self.model_params = module.model_params[model_id]
            elif "Mistral" in model_id:
                # Load from custom model_params/Mistral.py
                if not os.path.exists(f"model_params/Mistral.py"):
                    raise Exception(f"model_params/Mistral.py is not found")
                module = importlib.import_module(f"model_params.Mistral")
                self.model_params = module.model_params[model_id]
            elif "chatglm" in model_id:
                # Load from custom model_params/chatglm.py
                if not os.path.exists(f"model_params/chatglm.py"):
                    raise Exception(f"model_params/chatglm.py is not found")
                module = importlib.import_module(f"model_params.chatglm")
                self.model_params = module.model_params[model_id]
            else:
                # For other huggingface models, use AutoConfig
                self.model_params = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        else:
            # For explicitly specified non-huggingface sources
            if not os.path.exists(f"model_params/{source}.py"):
                raise Exception(f"model_params/{source}.py is not found")
            module = importlib.import_module(f"model_params.{source}")
            self.model_params = module.model_params[model_id]
        
        print(f"DEBUG: Type of self.model_params after loading: {type(self.model_params)}")
        print(f"DEBUG: self.model_params attributes: {dir(self.model_params)}")
        
        self.config = importlib.import_module(config_file.replace("/", ".").replace(".py", ""))
        print(f"DEBUG: self.config module: {self.config.__name__}")

        # temporary variables
        self.results = None
        self.w_bit = None
        self.a_bit = None
        self.kv_bit = None
        self.batchsize = None
        self.seqlen = None

    def estimate_kv_cache_size(self, seq_len, batch_size, dtype_bytes):
        num_layers = self.config.get_num_hidden_layers(self.model_params)
        hidden_size = self.config.get_hidden_size(self.model_params)
        return (
            num_layers
            * 2
            * hidden_size
            * seq_len
            * batch_size
            * dtype_bytes
        )

    def _analyze_to_results(
        self,
        stage,
        name,
        OPs=0,
        load_weight=0,
        load_act=0,
        store_act=0,
        load_kv_cache=0,
        store_kv_cache=0,
    ):

        dram_bandwidth, max_OPS, onchip_buffer = self.get_hardware_info()

        kv_access = load_kv_cache + store_kv_cache
        if self.weight_storage == "flash":
            flash_access = load_weight
            dram_access = load_act + store_act + kv_access
        else:
            flash_access = 0
            dram_access = load_weight + load_act + store_act + kv_access

        # Backward-compatible: keep "memory_access" as total bytes moved.
        memory_access = dram_access + flash_access

        # Roofline / inference time fusion:
        # compute_time = OPs / max_OPS
        # memory_time  = dram_time + flash_time
        # inference_time = max(compute_time, memory_time)
        roofline_memory_access = dram_access if dram_access > 0 else 1
        arithmetic_intensity = OPs / roofline_memory_access
        compute_time = 0 if max_OPS == 0 else (OPs / max_OPS)

        dram_time = 0 if dram_bandwidth == 0 else (dram_access / dram_bandwidth)
        if flash_access > 0:
            flash_bandwidth = hardware_params[self.hardware].get("flash_bandwidth")
            if flash_bandwidth is None:
                raise ValueError(
                    f"Hardware '{self.hardware}' does not define 'flash_bandwidth' but weight_storage='flash' was requested."
                )
            flash_time = flash_access / flash_bandwidth
        else:
            flash_time = 0

        # Simple serial memory-time model: DRAM transfer + Flash transfer.
        memory_time = dram_time + flash_time

        inference_time = max(compute_time, memory_time)
        bound = "memory" if memory_time > compute_time else "compute"
        performance = 0 if inference_time == 0 else (OPs / inference_time)

        # Kept for backward compatibility; currently equals inference_time.
        inference_time_total = inference_time

        # Weight memory access classification (does not change existing layer definitions).
        load_weight_dram = load_weight if self.weight_storage != "flash" else 0
        load_weight_flash = load_weight if self.weight_storage == "flash" else 0

        self.results[stage][name] = {
            "OPs": OPs,
            "memory_access": memory_access,
            "dram_access": dram_access,
            "flash_access": flash_access,
            "arithmetic_intensity": arithmetic_intensity,
            "performance": performance,
            "bound": bound,
            "load_weight": load_weight,
            "load_weight_dram": load_weight_dram,
            "load_weight_flash": load_weight_flash,
            "load_act": load_act,
            "store_act": store_act,
            "load_kv_cache": load_kv_cache,
            "store_kv_cache": store_kv_cache,
            "inference_time": inference_time,
            "dram_time": dram_time,
            "flash_time": flash_time,
            "memory_time": memory_time,
            "compute_time": compute_time,
            "inference_time_total": inference_time_total,
        }

    def save_csv(self, save_path=None):
        if save_path is None:
            save_path = f"output/{self.model_id[:self.model_id.rfind('/')]}"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_path += f"{self.model_id[self.model_id.rfind('/'):]}"

        decode_file_name = f"{save_path}_decode.csv"
        prefill_file_name = f"{save_path}_prefill.csv"
        print(f"save to {decode_file_name} and {prefill_file_name}")

        for file_name, stage in [
            (decode_file_name, "decode"),
            (prefill_file_name, "prefill"),
        ]:
            with open(file_name, "a+") as f:

                f.write(
                    f"\n\n=== {self.model_id} {self.hardware} w_bit={self.w_bit} a_bit={self.a_bit} kv_bit={self.kv_bit} batchsize={self.batchsize} seqlen={self.seqlen} tp_size={self.tp_size} ===\n"
                )
                # legend
                f.write(
                    f"layer_name,OPs,Access,dram_access,flash_access,arithmetic_intensity,performance,bound,load_weight,load_weight_dram,load_weight_flash,load_act,store_act,load_kv_cache,store_kv_cache,inference_time,compute_time,dram_time,flash_time,memory_time,inference_time_total\n"
                )
            with open(file_name, "a+") as f:
                for layer_name, result in self.results[stage].items():
                    f.write(
                        f"{layer_name},{str_number(result['OPs'])},{str_number(result['memory_access'])}B,{str_number(result['dram_access'])}B,{str_number(result['flash_access'])}B,{str_number(result['arithmetic_intensity'])},{str_number(result['performance'])},"
                        f"{result['bound']},{str_number(result['load_weight'])}B,{str_number(result['load_weight_dram'])}B,{str_number(result['load_weight_flash'])}B,{str_number(result['load_act'])}B,{str_number(result['store_act'])}B,{str_number(result['load_kv_cache'])}B,"
                        f"{str_number(result['store_kv_cache'])}B,{str_number_time(result['inference_time'])}s,{str_number_time(result['compute_time'])}s,{str_number_time(result['dram_time'])}s,{str_number_time(result['flash_time'])}s,{str_number_time(result['memory_time'])}s,{str_number_time(result['inference_time_total'])}s\n"
                    )

    def analyze(
        self,
        seqlen,
        batchsize,
        w_bit=16,
        a_bit=16,
        kv_bit=None,
        use_flashattention=False,
        kv_token_ratio=1,
        tp_size: int = 1
    ):
        """
        seqlen: sequence length
        batchsize: batch size
        w_bit: weight bit
        a_bit: activation bit
        kv_bit: key and value bit. if it is None, it will be the same as a_bit
        use_flashattention: use flash attention/flash decoding
        kv_token_ratio: use this for KV compression
        tp_size: the number of devices for tensor parallelism to use

        return is a dict with the following format:
        {
            "decode": {
                    "layer_name": {
                            "OPs": "",
                            "memory_access": "",
                            "arithmetic_intensity": "",
                            "performance": "",
                            "bound": "",
                            "load_weight": "",
                            "load_act": "",
                            "store_act": "",
                            "load_kv_cache": "",
                            "store_kv_cache": "",
                            "inference_time": ""
                    }
            },
            "prefill": {
                    "layer_name": {
                            "OPs": "",
                            "memory_access": "",
                            "arithmetic_intensity": "",
                            "performance": "",
                            "bound": "",
                            "load_weight": "",
                            "load_act": "",
                            "store_act": "",
                            "load_kv_cache": "",
                            "store_kv_cache": "",
                            "inference_time": ""
                    }
            },
            "total_results": {
                "decode": {},
                "prefill": {}
            }
        }
        """
        assert seqlen > 0
        assert batchsize > 0
        self.results = {"decode": {}, "prefill": {}}
        if kv_bit is None:
            kv_bit = a_bit
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.kv_bit = kv_bit
        self.batchsize = batchsize
        self.seqlen = seqlen
        self.tp_size = tp_size

        w_byte = self.w_bit / 8
        a_byte = self.a_bit / 8
        kv_byte = self.kv_bit / 8

        config = self.config
        model_params = self.model_params
        
        print(f"DEBUG: In analyze method, type of model_params: {type(model_params)}")
        print(f"DEBUG: In analyze method, model_params attributes: {dir(model_params)}")
        print(f"DEBUG: In analyze method, config module: {config.__name__}")

        num_attention_heads = config.get_num_attention_heads(model_params)
        hidden_size = config.get_hidden_size(model_params)
        num_key_value_heads = config.get_num_key_value_heads(model_params)
        num_hidden_layers = config.get_num_hidden_layers(model_params)

        for name, (ic, oc) in config.get_linear_layers(model_params, tp_size).items():
            # for linear layers
            is_kv_proj = name in ["k_proj", "v_proj"]
            is_normal_proj = not is_kv_proj
            self._analyze_to_results(
                "decode",
                name,
                OPs=ic * oc * batchsize * 2,
                load_weight=ic * oc * w_byte,
                load_act=ic * batchsize * a_byte,
                store_act=0 if is_kv_proj else oc * batchsize * a_byte,
                load_kv_cache=0,
                store_kv_cache=(0 if is_normal_proj else oc * batchsize * kv_byte),
            )
            # for prefill
            self._analyze_to_results(
                "prefill",
                name,
                OPs=ic * oc * batchsize * seqlen * 2,
                load_weight=ic * oc * w_byte,
                load_act=ic * batchsize * seqlen * a_byte,
                store_act=(0 if is_kv_proj else oc * batchsize * seqlen * a_byte),
                load_kv_cache=0,
                store_kv_cache=(0 if is_normal_proj else oc * batchsize * seqlen * kv_byte),
            )

        # for attention
        head_size = hidden_size // num_attention_heads
        # for decode
        qk_matmul_OPs = seqlen * head_size * num_attention_heads * batchsize * 2
        sv_matmul_OPs = 1 * head_size * seqlen * num_attention_heads * batchsize * 2
        # the softmax operation takes five steps:
        # max_x=max(x)
        # x=x-max_x
        # x_exp=exp(x)
        # sum_x_exp=sum(x_exp)
        # y=x_exp/sum(x_exp)
        softmax_OPs = batchsize * num_attention_heads * seqlen * 1 * 5
        if use_flashattention:
            name = f"fused_attention"
            bandwidth, max_OPS, onchip_buffer = self.get_hardware_info()
            # flashattention-2 https://arxiv.org/pdf/2307.08691.pdf
            block_size_r = min(math.ceil(onchip_buffer / (kv_byte * head_size)), head_size)
            n_blocks_r = math.ceil(1 / block_size_r)
            q_numel = (1) * head_size * batchsize * num_attention_heads * a_byte
            o_numel = 1 * seqlen * batchsize * num_attention_heads * a_byte
            self._analyze_to_results(
                "decode",
                name,
                OPs=qk_matmul_OPs + sv_matmul_OPs + softmax_OPs,
                load_weight=0,
                load_act=q_numel,
                store_act=o_numel * 2,  # initialize O and save O
                load_kv_cache=n_blocks_r * (seqlen) * head_size * batchsize * num_key_value_heads * kv_byte * 2,
                store_kv_cache=0,
            )

        else:
            name = f"qk_matmul"
            self._analyze_to_results(
                "decode",
                name,
                OPs=qk_matmul_OPs,
                load_weight=0,
                load_act=(1) * head_size * batchsize * num_attention_heads * a_byte,
                store_act=1 * seqlen * batchsize * num_attention_heads * a_byte,
                load_kv_cache=(seqlen) * head_size * batchsize * num_key_value_heads * kv_byte,
                store_kv_cache=0,
            )
            name = f"sv_matmul"
            self._analyze_to_results(
                "decode",
                name,
                OPs=sv_matmul_OPs,
                load_weight=0,
                load_act=(1 * seqlen * batchsize * num_attention_heads) * a_byte,
                store_act=1 * head_size * batchsize * num_attention_heads * a_byte,
                load_kv_cache=(seqlen * head_size * batchsize * num_key_value_heads) * kv_byte,
                store_kv_cache=0,
            )

            name = f"softmax"
            # max sub exp sum div
            self._analyze_to_results(
                "decode",
                name,
                OPs=softmax_OPs,
                load_weight=0,
                load_act=batchsize * num_attention_heads * seqlen * 1 * a_byte,
                store_act=batchsize * num_attention_heads * seqlen * 1 * a_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )

        for name in config.get_norm_layers(model_params):
            # sum sub pow sum div mul add
            self._analyze_to_results(
                "decode",
                name,
                OPs=batchsize * hidden_size * 1 * 7,
                load_weight=0,
                load_act=batchsize * hidden_size * 1 * a_byte,
                store_act=batchsize * hidden_size * 1 * a_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )

        for name in ["attn_add", "mlp_add"]:
            self._analyze_to_results(
                "decode",
                name,
                OPs=batchsize * hidden_size * 1,
                load_weight=0,
                load_act=batchsize * hidden_size * 1 * a_byte,
                store_act=batchsize * hidden_size * 1 * a_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )
        for name in ["mlp_act"]:
            self._analyze_to_results(
                "decode",
                name,
                OPs=batchsize * hidden_size * 1 * 2,
                load_weight=0,
                load_act=batchsize * hidden_size * 1 * a_byte * 2,
                store_act=batchsize * hidden_size * 1 * a_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )

        # for prefill
        qk_matmul_OPs = seqlen * seqlen * head_size * num_attention_heads * batchsize * 2
        sv_matmul_OPs = seqlen * head_size * seqlen * num_attention_heads * batchsize * 2
        softmax_OPs = batchsize * num_attention_heads * seqlen * seqlen * 5
        if use_flashattention:
            name = f"fused_attention"
            bandwidth, max_OPS, onchip_buffer = self.get_hardware_info()
            # flashattention-2 https://arxiv.org/pdf/2307.08691.pdf
            block_size_r = min(math.ceil(onchip_buffer / (kv_byte * head_size)), head_size)
            n_blocks_r = math.ceil(seqlen / block_size_r)
            q_numel = seqlen * head_size * batchsize * num_attention_heads * a_byte
            o_numel = seqlen * seqlen * batchsize * num_attention_heads * a_byte
            self._analyze_to_results(
                "prefill",
                name,
                OPs=qk_matmul_OPs + sv_matmul_OPs + softmax_OPs,
                load_weight=0,
                load_act=q_numel,
                store_act=o_numel * 2,  # initialize O and save O
                load_kv_cache=n_blocks_r * (seqlen) * head_size * batchsize * num_key_value_heads * kv_byte * 2,
                store_kv_cache=0,
            )
        else:
            name = f"qk_matmul"
            self._analyze_to_results(
                "prefill",
                name,
                OPs=qk_matmul_OPs,
                load_weight=0,
                load_act=seqlen * head_size * batchsize * num_key_value_heads * a_byte,
                store_act=seqlen * seqlen * batchsize * num_attention_heads * a_byte,
                load_kv_cache=seqlen * head_size * batchsize * num_key_value_heads * kv_byte,
                store_kv_cache=0,
            )
            name = f"sv_matmul"
            self._analyze_to_results(
                "prefill",
                name,
                OPs=sv_matmul_OPs,
                load_weight=0,
                load_act=seqlen * seqlen * batchsize * num_attention_heads * a_byte,
                store_act=seqlen * head_size * batchsize * num_attention_heads * a_byte,
                load_kv_cache=seqlen * head_size * batchsize * num_key_value_heads * kv_byte,
                store_kv_cache=0,
            )
            name = f"softmax"
            self._analyze_to_results(
                "prefill",
                name,
                OPs=softmax_OPs,
                load_weight=0,
                load_act=batchsize * num_attention_heads * seqlen * seqlen * a_byte,
                store_act=batchsize * num_attention_heads * seqlen * seqlen * a_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )
        for name in config.get_norm_layers(model_params):
            self._analyze_to_results(
                "prefill",
                name,
                OPs=batchsize * hidden_size * seqlen * 7,
                load_weight=0,
                load_act=batchsize * hidden_size * seqlen * a_byte,
                store_act=batchsize * hidden_size * seqlen * a_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )
        for name in ["attn_add", "mlp_add"]:
            self._analyze_to_results(
                "prefill",
                name,
                OPs=batchsize * hidden_size * seqlen * 1,
                load_weight=0,
                load_act=batchsize * hidden_size * seqlen * a_byte,
                store_act=batchsize * hidden_size * seqlen * a_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )
        for name in ["mlp_act"]:
            self._analyze_to_results(
                "prefill",
                name,
                OPs=batchsize * hidden_size * seqlen * 1 * 2,
                load_weight=0,
                load_act=batchsize * hidden_size * seqlen * a_byte * 2,
                store_act=batchsize * hidden_size * seqlen * a_byte,
                load_kv_cache=0,
                store_kv_cache=0,
            )

        # lm_head (accounted before totals/footprints so it's included everywhere)
        args = {"batchsize": batchsize, "a_byte": a_byte, "w_byte": w_byte}
        for layer_info in self.config.post_process(self.model_params, args):
            self._analyze_to_results(**layer_info)

        # compute total
        total_results = {"decode": {}, "prefill": {}}
        for data_name in ALL_DATA_NAMES:
            total_results["decode"][data_name] = 0
            total_results["prefill"][data_name] = 0
        for stage in ["decode", "prefill"]:
            for layer_name, result in self.results[stage].items():
                for data_name in ALL_DATA_NAMES:
                    total_results[stage][data_name] += result[data_name] * num_hidden_layers

        # Weight access breakdown by storage mode.
        for stage in ["decode", "prefill"]:
            total_results[stage]["load_weight_dram"] = 0
            total_results[stage]["load_weight_flash"] = 0
            for _, result in self.results[stage].items():
                total_results[stage]["load_weight_dram"] += result["load_weight_dram"] * num_hidden_layers
                total_results[stage]["load_weight_flash"] += result["load_weight_flash"] * num_hidden_layers

        # DRAM/Flash access split and timing.
        for stage in ["decode", "prefill"]:
            total_results[stage]["dram_access"] = 0
            total_results[stage]["flash_access"] = 0
            total_results[stage]["dram_time"] = 0
            total_results[stage]["flash_time"] = 0
            total_results[stage]["memory_time"] = 0
            total_results[stage]["compute_time"] = 0
            total_results[stage]["inference_time_total"] = 0
            for _, result in self.results[stage].items():
                total_results[stage]["dram_access"] += result["dram_access"] * num_hidden_layers
                total_results[stage]["flash_access"] += result["flash_access"] * num_hidden_layers
                total_results[stage]["dram_time"] += result["dram_time"] * num_hidden_layers
                total_results[stage]["flash_time"] += result["flash_time"] * num_hidden_layers
                total_results[stage]["memory_time"] += result["memory_time"] * num_hidden_layers
                total_results[stage]["compute_time"] += result["compute_time"] * num_hidden_layers
                total_results[stage]["inference_time_total"] += result["inference_time_total"] * num_hidden_layers

        # Result output structure upgrade (comparison-friendly aliases).
        # Keep existing keys; add explicit names for plotting/inspection.
        for stage in ["decode", "prefill"]:
            total_results[stage]["dram_memory_time"] = total_results[stage]["dram_time"]
            total_results[stage]["flash_memory_time"] = total_results[stage]["flash_time"]
            total_results[stage]["total_memory_time"] = total_results[stage]["memory_time"]
            # Explicitly expose inference_time in the upgraded structure.
            # (Already present via ALL_DATA_NAMES aggregation.)
            total_results[stage]["inference_time"] = total_results[stage]["inference_time"]

        # Flash capacity check (Flash-Direct core constraint).
        # Treat the sum of per-layer weight bytes (unique parameters) as the total model weight size.
        if self.weight_storage == "flash":
            hw = hardware_params[self.hardware]
            flash_capacity = hw.get("flash_capacity")
            if flash_capacity is None:
                raise ValueError(
                    f"Hardware '{self.hardware}' does not define 'flash_capacity' but weight_storage='flash' was requested."
                )

            total_weight_bytes = total_results["prefill"]["load_weight"]
            if total_weight_bytes > flash_capacity:
                raise ValueError(
                    "Model weights exceed flash capacity"
                    f" (weights={int(total_weight_bytes)} bytes, flash_capacity={int(flash_capacity)} bytes)"
                )

        # memory footprint
        kv_cache_size = self.estimate_kv_cache_size(seqlen, batchsize, kv_byte)
        weight_kv_footprint = total_results["prefill"]["load_weight"] + kv_cache_size
        decode_tmp_act = 0
        for layer_name, result in self.results["decode"].items():
            decode_tmp_act += result["store_act"]
        total_results["decode"]["memory_consumption"] = decode_tmp_act + weight_kv_footprint
        total_results["decode"]["memory_consumption_tmp_act"] = decode_tmp_act
        total_results["decode"]["memory_consumption_weight"] = total_results["prefill"]["load_weight"]
        total_results["decode"]["memory_consumption_kv_cache"] = kv_cache_size
        prefill_tmp_act = 0
        for layer_name, result in self.results["prefill"].items():
            prefill_tmp_act += result["store_act"]
        total_results["prefill"]["memory_consumption"] = prefill_tmp_act + weight_kv_footprint
        total_results["prefill"]["memory_consumption_tmp_act"] = prefill_tmp_act
        total_results["prefill"]["memory_consumption_weight"] = total_results["prefill"]["load_weight"]
        total_results["prefill"]["memory_consumption_kv_cache"] = kv_cache_size

        # DRAM footprint breakdown (capacity semantics).
        # - weights: DRAM-resident weights (0 in Flash-Direct mode)
        # - kv_cache: KV cache footprint (assumed DRAM)
        # - activations_peak: simple residual activation peak estimate
        # - temp_peak: peak temporary buffer estimate (max intermediate tensor, closer to real peak)
        for stage in ["prefill", "decode"]:
            activations_peak = batchsize * hidden_size * (seqlen if stage == "prefill" else 1) * a_byte
            # Peak temporary buffer: use the maximum of per-layer activation IO as a proxy
            # for the largest intermediate tensor materialized at any point.
            temp_peak = 0
            for _, r in self.results[stage].items():
                temp_peak = max(temp_peak, r.get("load_act", 0), r.get("store_act", 0))
            total_results[stage]["memory_consumption_tmp_act_peak"] = temp_peak
            dram_weights = total_results["prefill"].get("load_weight_dram", total_results["prefill"]["load_weight"])
            flash_weights = total_results["prefill"].get("load_weight_flash", 0)

            total_results[stage]["dram_footprint"] = {
                "weights": dram_weights,
                "kv_cache": kv_cache_size,
                "activations_peak": activations_peak,
                "temp_peak": temp_peak,
            }
            total_results[stage]["memory_footprint"] = {
                "dram": {
                    "weights": dram_weights,
                    "kv_cache": kv_cache_size,
                    "activations_peak": activations_peak,
                    "temp_peak": temp_peak,
                    "total": dram_weights + kv_cache_size + activations_peak + temp_peak,
                },
                "flash": {
                    "weights": flash_weights,
                },
            }

        # DRAM capacity check (core constraint).
        hw = hardware_params[self.hardware]
        dram_capacity = hw.get("dram_capacity")
        # If Flash-extended hardware is used, dram_capacity should be defined.
        if dram_capacity is None and ("flash_capacity" in hw or self.weight_storage == "flash"):
            raise ValueError(
                f"Hardware '{self.hardware}' does not define 'dram_capacity' but a flash-capable/flash mode was requested."
            )
        if dram_capacity is not None:
            worst_required = 0
            worst_stage = None
            for stage in ["prefill", "decode"]:
                fp = total_results[stage]["dram_footprint"]
                total_dram_required = (
                    fp["weights"]
                    + fp["kv_cache"]
                    + fp["activations_peak"]
                    + fp["temp_peak"]
                )
                if total_dram_required > worst_required:
                    worst_required = total_dram_required
                    worst_stage = stage
                if total_dram_required > dram_capacity:
                    raise RuntimeError(
                        "DRAM OOM"
                        f" (stage={stage}, required={int(total_dram_required)} bytes, dram_capacity={int(dram_capacity)} bytes)"
                    )
            total_results["dram_total_required_peak"] = worst_required
            total_results["dram_total_required_peak_stage"] = worst_stage

        # for stage in ["prefill", "decode"]:
        #     self._analyze_to_results(
        #         stage,
        #         name,
        #         OPs=batchsize * hidden_size * vocab_size * 1,
        #         load_weight=hidden_size * vocab_size,
        #         load_act=hidden_size * a_byte,
        #         store_act=vocab_size * a_byte,
        #         load_kv_cache=0,
        #         store_kv_cache=0,
        #     )
        #     for data_name in ALL_DATA_NAMES:
        #         total_results[stage][data_name] += self.results[stage][name][data_name]

        self.results["total_results"] = total_results
        return self.results

    def analyze_generate_task(
        self,
        prompt_len,
        gen_len,
        batchsize,
        w_bit=16,
        a_bit=16,
        kv_bit=None,
        use_flashattention = False,
        tp_size: int = 1
    ):
        prefill_result = self.analyze(
            prompt_len,
            batchsize,
            w_bit,
            a_bit,
            kv_bit,
            use_flashattention=use_flashattention,
            tp_size=tp_size
        )
        prefill_time = inference_time = prefill_result["total_results"]["prefill"]["inference_time"]

        for i in range(prompt_len, prompt_len + gen_len):
            result = self.analyze(i, batchsize, w_bit, a_bit, kv_bit, use_flashattention=use_flashattention, tp_size=tp_size)
            inference_time += result["total_results"]["decode"]["inference_time"]
        return {"inference_time": inference_time, "prefill_time": prefill_time}

    def get_hardware_info(self):
        hw = hardware_params[self.hardware]
        # Backward compatible: existing entries use "bandwidth".
        # Flash-extended entries may use explicit "dram_bandwidth".
        bandwidth = hw.get("dram_bandwidth", hw.get("bandwidth"))
        if bandwidth is None:
            raise KeyError(f"Hardware '{self.hardware}' must define 'bandwidth' or 'dram_bandwidth'.")

        use_int8 = self.w_bit <= 8 and self.a_bit <= 8 and self.kv_bit <= 8
        if use_int8 and "INT8" in hw:
            max_OPS = hw["INT8"]
        else:
            max_OPS = hw["FP16"]

        onchip_buffer = hw["onchip_buffer"]
        return bandwidth, max_OPS, onchip_buffer

    def get_model_info(self):
        if self.config.get_num_attention_heads(self.model_params) != self.config.get_num_key_value_heads(
            self.model_params
        ):
            GQA = True
        else:
            GQA = False

        info = {"GQA": GQA}  # group query attention
        return info
