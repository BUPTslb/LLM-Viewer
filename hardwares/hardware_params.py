# the OPS = sparse OPS/2

hardware_params = {
    # NOTICES: For GPU, we use Register File Size as on-chip buffer size
    # https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf
    # NOTICE: V100 not support INT8 in tensor core, so INT8 performance is not good
    "nvidia_V100": {"bandwidth": 900e9, "FP16": 112e12, "INT8": 62e12, "onchip_buffer": 20480e3},
    # https://images.nvidia.com/aem-dam/en-zz/Solutions/technologies/NVIDIA-ADA-GPU-PROVIZ-Architecture-Whitepaper_1.1.pdf
    "nvidia_A6000": {"bandwidth": 768e9, "FP16": 154.8e12, "INT8": 309.7e12, "onchip_buffer": 21504e3},
    # https://images.nvidia.com/aem-dam/en-zz/Solutions/technologies/NVIDIA-ADA-GPU-PROVIZ-Architecture-Whitepaper_1.1.pdf
    "nvidia_A6000_Ada": {"bandwidth": 960e9, "FP16": 364.2e12, "INT8": 728.5e12, "onchip_buffer": 36352e3},
    # https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf
    # Ampere's SM has 256KB RF, max 164KB Shared Mem
    "nvidia_A100": {"bandwidth": 1555e9, "FP16": 312e12, "INT8": 624e12, "onchip_buffer": 27648e3},  # use 40G data
    "nvidia_A100_40G": {"bandwidth": 1555e9, "FP16": 312e12, "INT8": 624e12, "onchip_buffer": 27648e3},
    "nvidia_A100_80G": {"bandwidth": 2039e9, "FP16": 312e12, "INT8": 624e12, "onchip_buffer": 27648e3},
    "nvidia_A800_80G_SXM": {"bandwidth": 2039e9, "FP16": 312e12, "INT8": 624e12, "onchip_buffer": 27648e3},
    "nvidia_A40": {"bandwidth": 696e9, "FP16": 149.7e12, "INT8": 299.3e12, "onchip_buffer": 21504e3},
    # https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper
    "nvidia_H100": {
        "bandwidth": 3072e9,
        "FP16": 1979e12 / 2,
        "INT8": 3958e12 / 2,
        "onchip_buffer": 33792e3,
    },  # use SXM data
    "nvidia_H100_SXM": {"bandwidth": 3072e9, "FP16": 1979e12 / 2, "INT8": 3958e12 / 2, "onchip_buffer": 33792e3},
    "nvidia_H100_PCIe": {"bandwidth": 2048e9, "FP16": 1513e12 / 2, "INT8": 3026e12 / 2, "onchip_buffer": 29184e3},
    # https://images.nvidia.com/aem-dam/Solutions/Data-Center/l4/nvidia-ada-gpu-architecture-whitepaper-v2.1.pdf
    # Ada SM has 256 KB Register File, and 128 KB of L1/Shared Memory
    "nvidia_L40": {"bandwidth": 864e9, "FP16": 181e12, "INT8": 362e12, "onchip_buffer": 36352e3},
    # Intel Skylake-X (Skylake-X, Cascade Lake) Intel Xeon Phi (Knights Landing, Knights Mill) Intel Ice Lake, Tiger Lake and Rocket Lake
    # support AVX-512 & FMA (512-bit), they has throughput of 1 cycle
    # https://www.intel.com/content/www/us/en/products/sku/230496/intel-core-i913900k-processor-36m-cache-up-to-5-80-ghz/specifications.html
    "intel_13900k": {"bandwidth": 89.6e9, "FP16": 8 * 5.4e9 * (512 / 16), "onchip_buffer": 36e6},
    "Snapdragon_8_Gen3": {"bandwidth": 76.6e9, "FP16": 64000e9 , "onchip_buffer": 12e6},
    "3DNF": {"bandwidth": 400e9, "FP16": 4000e9, "onchip_buffer": 6e6},
    "RTX4090": {"bandwidth": 1010e9, "FP16": 82600e9, "onchip_buffer": 16384e3},
    "Island": {"bandwidth": 216e9, "FP16": 4000e9, "INT8": 8e12, "INT4": 16e12, "onchip_buffer": 10e6},

    # Flash-extended hardware model example.
    # NOTE: Do not overwrite existing "bandwidth" for other platforms.
    # Use explicit fields to distinguish DRAM vs Flash bandwidth.
    "Samsung_Exynos_2400_flash": {
        "dram_bandwidth": 200e9,
        "dram_capacity": 16e9,  # bytes
        "flash_bandwidth": 10e9,  # Flash 1 GPU
        "flash_capacity": 512e9,  # bytes
        "FP16": 200e12,
        "onchip_buffer": 10240e3,
    },
    "AIPC_DRAM5600": {
    # 32GB DDR5-5600 Desktop Memory
    # 双通道：2 * DDR5 5600 MT/s, 64-bit ≈ 89.6 GB/s
    # 四通道：2 * DDR5 5600 MT/s, 64-bit ≈ 179.2 GB/s
    "dram_bandwidth": 89.6e9,       # bytes/s
    "dram_capacity": 48e9,         # bytes
    # NPU + iGPU FP16 合计（典型 40~60 TOPS）
    "FP16": 50e12,                 # ops/s
    # NPU SRAM + LLC（估算）
    "onchip_buffer": 32e6,         # bytes
},
    "AIPC_HBF-PCIE": {
   
    # 双通道：2 * DDR5 5600 MT/s, 64-bit ≈ 89.6 GB/s
    # 四通道：2 * DDR5 5600 MT/s, 64-bit ≈ 179.2 GB/s
    "dram_bandwidth": 89.6e9,       # bytes/s
    "dram_capacity": 2048e8,         # bytes
    # 32GB PCIE-HBF，PCIe 5.0 NVMe SSD 16*Lanes
    "flash_bandwidth": 64e9,        # bytes/s
    "flash_capacity": 64e9,        # bytes

    # NPU + iGPU FP16 合计（典型 40~60 TOPS）
    "FP16": 50e12,                 # ops/s

    # NPU SRAM + LLC（估算）
    "onchip_buffer": 32e6,         # bytes
},
    "AIPC_HBF-DDR": {
   
    # 双通道：2 * DDR5 5600 MT/s, 64-bit ≈ 89.6 GB/s
    # 四通道：2 * DDR5 5600 MT/s, 64-bit ≈ 179.2 GB/s
    "dram_bandwidth": 89.6e9,       # bytes/s
    "dram_capacity": 2048e8,         # bytes
    # 32GB PCIE-HBF，DDR
    "flash_bandwidth": 120e9,        # bytes/s (≈ 4ch × 30GB/s) 4-channel LPDDR5
    "flash_capacity": 64e9,        # bytes

    # NPU + iGPU FP16 合计（典型 40~60 TOPS）
    "FP16": 50e12,                 # ops/s

    # NPU SRAM + LLC（估算）
    "onchip_buffer": 32e6,         # bytes
},
    "AIPC_HBF-2.5D": {
   
    # 双通道：2 * DDR5 5600 MT/s, 64-bit ≈ 89.6 GB/s
    # 四通道：2 * DDR5 5600 MT/s, 64-bit ≈ 179.2 GB/s
    "dram_bandwidth": 89.6e9,       # bytes/s
    "dram_capacity": 2048e8,         # bytes 可以更小
    # 32GB PCIE-HBF，2.5D
    "flash_bandwidth":  150e9,        # bytes/s 带宽和容量有关系
    "flash_capacity": 64e9,        # bytes 

    # NPU + iGPU FP16 合计（典型 40~60 TOPS）
    "FP16": 50e12,                 # ops/s

    # NPU SRAM + LLC（估算）
    "onchip_buffer": 32e6,         # bytes
},
}
