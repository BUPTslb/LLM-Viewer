# llama-2-7b
# python3 analyze_cli.py meta-llama/Llama-2-7b-hf Island --batchsize 1 --seqlen 32 --w_bit 8 --kv_bit 8 --a_bit 8
# python3 analyze_cli.py meta-llama/Llama-2-7b-hf Island --batchsize 1 --seqlen 64 --w_bit 8 --kv_bit 8 --a_bit 8
# python3 analyze_cli.py meta-llama/Llama-2-7b-hf Island --batchsize 1 --seqlen 128 --w_bit 8 --kv_bit 8 --a_bit 8
# python3 analyze_cli.py meta-llama/Llama-2-7b-hf Island --batchsize 1 --seqlen 256 --w_bit 8 --kv_bit 8 --a_bit 8
# python3 analyze_cli.py meta-llama/Llama-2-7b-hf Island --batchsize 1 --seqlen 512 --w_bit 8 --kv_bit 8 --a_bit 8
# python3 analyze_cli.py meta-llama/Llama-2-7b-hf Island --batchsize 1 --seqlen 1024 --w_bit 8 --kv_bit 8 --a_bit 8
# python3 analyze_cli.py meta-llama/Llama-2-7b-hf Island --batchsize 1 --seqlen 2048 --w_bit 8 --kv_bit 8 --a_bit 8
# python3 analyze_cli.py meta-llama/Llama-2-7b-hf Island --batchsize 1 --seqlen 4096 --w_bit 8 --kv_bit 8 --a_bit 8
# python3 analyze_cli.py meta-llama/Llama-2-7b-hf Island --batchsize 1 --seqlen 8192 --w_bit 8 --kv_bit 8 --a_bit 8
# python3 analyze_cli.py meta-llama/Llama-2-7b-hf Island --batchsize 1 --seqlen 16384 --w_bit 8 --kv_bit 8 --a_bit 8
# python3 analyze_cli.py meta-llama/Llama-2-7b-hf Island --batchsize 1 --seqlen 32768 --w_bit 8 --kv_bit 8 --a_bit 8
# python3 analyze_cli.py meta-llama/Llama-2-7b-hf Island --batchsize 1 --seqlen 65536 --w_bit 8 --kv_bit 8 --a_bit 8
# python3 analyze_cli.py meta-llama/Llama-2-7b-hf Island --batchsize 1 --seqlen 131072 --w_bit 8 --kv_bit 8 --a_bit 8
# python3 analyze_cli.py meta-llama/Llama-2-7b-hf Island --batchsize 1 --seqlen 262144 --w_bit 8 --kv_bit 8 --a_bit 8

# llama-2-13b
# python3 analyze_cli.py meta-llama/Llama-2-13b-hf Island --batchsize 1 --seqlen 64 --w_bit 8 --kv_bit 8 --a_bit 8
# opt-6.7b
# python3 analyze_cli.py facebook/opt-6.7b Island --batchsize 1 --seqlen 64 --w_bit 8 --kv_bit 8 --a_bit 8
# opt-13b
# python3 analyze_cli.py facebook/opt-13b Island --batchsize 1 --seqlen 64 --w_bit 8 --kv_bit 8 --a_bit 8
# opt-30b
# python3 analyze_cli.py facebook/opt-30b Island --batchsize 1 --seqlen 64 --w_bit 8 --kv_bit 8 --a_bit 8

# llama3-8b
# python3 analyze_cli.py meta-llama/Meta-Llama-3-8B Island --batchsize 1 --seqlen 64 --w_bit 8 --kv_bit 8 --a_bit 8

# mistral-7b
python3 analyze_cli.py "Mistral-7B-Instruct-v0.1" Island --source Mistral --config_file configs/Mistral --batchsize 1 --seqlen 64 --w_bit 8 --kv_bit 8 --a_bit 8
# qwen1.5-1.8b
python3 analyze_cli.py "Qwen1.5-1.8B-Chat" Island --source Qwen --config_file configs/Qwen --batchsize 1 --seqlen 64 --w_bit 8 --kv_bit 8 --a_bit 8
# qwen1.5-4b
python3 analyze_cli.py "Qwen1.5-4B-Chat" Island --source Qwen --config_file configs/Qwen --batchsize 1 --seqlen 64 --w_bit 8 --kv_bit 8 --a_bit 8
# gemma2-2b
python3 analyze_cli.py "gemma-2-2b" Island --source gemma --config_file configs/Llama --batchsize 1 --seqlen 64 --w_bit 8 --kv_bit 8 --a_bit 8
# phi-2-2.7b
python analyze_cli.py "phi-2" Island --source phi --config_file configs/phi --batchsize 1 --seqlen 64 --w_bit 8 --kv_bit 8 --a_bit 8
# falcon-11b
python3 analyze_cli.py "tiiuae/falcon-11b" Island --source falcon --config_file configs/falcon --batchsize 1 --seqlen 64 --w_bit 8 --kv_bit 8 --a_bit 8     
# falcon-40b
python3 analyze_cli.py "tiiuae/falcon-40b" Island --source falcon --config_file configs/falcon --batchsize 1 --seqlen 64 --w_bit 8 --kv_bit 8 --a_bit 8    
# gpt-neox-20b
python3 analyze_cli.py "gpt-neox-20b" Island --source gpt_neox --config_file configs/gpt_neox --batchsize 1 --seqlen 64 --w_bit 8 --kv_bit 8 --a_bit 8
# chatglm2-6b --config_file configs/chatglm3
python3 analyze_cli.py "chatglm2-6b" Island --source chatglm --config_file configs/chatglm3  --batchsize 1 --seqlen 64 --w_bit 8 --kv_bit 8 --a_bit 8