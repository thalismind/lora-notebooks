# Shared LoRAs

A collection of Jupyter notebooks for analyzing and modifying LoRA (Low-Rank Adaptation) models. These tools help with understanding LoRA layer importance, blending multiple LoRAs, and optimizing model performance.

## Notebooks

### 1. `blend_anything.ipynb`

**Purpose**: Blend any two LoRA models with memory-efficient streaming processing.

**Key Features**:
- **Memory Efficient**: Uses streaming processing to handle large LoRA models with minimal VRAM usage (typically < 4GB for Flux/Qwen models)
- **Flexible Blending**: Blend LoRAs from the same base model with custom weight ratios (`w1`, `w2`)
- **Rank Conversion**: Resize LoRAs to any target rank during blending (larger or smaller than original)
- **Multiple Factorization Methods**: Supports both SVD and PCA low-rank approximation
- **Device Management**: Configurable compute device and dtype for optimal performance
- **Layer Filtering**: Option to include only specific layers in the blending process
- **Numerical Stability**: Automatic cleanup of near-zero layers and CUDA memory management

**Usage Example**:
```python
blend16 = blend_and_convert_loras_streaming(
    lora1,
    lora2,
    w1=0.5,
    w2=0.75,
    target_rank=16,
    compute_device="cuda:0",
    compute_dtype=torch.float32,
)
```

**Technical Details**:
- Processes one layer at a time to minimize memory usage
- Uses truncated SVD or PCA for rank conversion
- Supports both `.lora_A.weight` and `.lora_B.weight` tensor pairs
- Automatic cleanup and memory management between layers

### 2. `find_important_layers.ipynb`

**Purpose**: Analyze LoRA layers to identify the most important ones using various statistical measures.

**Key Features**:
- **Comprehensive Analysis**: Multiple statistical measures including spectral norms, energy ratios, and distribution statistics
- **SVD-Based Importance**: Uses SVD spectral norms as the primary importance metric
- **Softmax Normalization**: Converts spectral norms to softmax probabilities for better interpretability
- **Block-Level Analysis**: Groups layers by transformer blocks and analyzes block importance
- **Threshold-Based Selection**: Find minimal set of blocks that capture a specified percentage of total importance
- **Memory Efficient**: Handles large models with configurable SVD computation limits
- **Extensible Statistics**: Modular design allows easy addition of new statistical measures

**Statistical Measures**:
- **Basic Stats**: min, max, mean, std, median
- **Sign Analysis**: positive/negative/zero value counts and fractions
- **Norms**: L1, L2, and L∞ norms
- **Percentiles**: 1st, 5th, 25th, 50th, 75th, 95th, 99th percentiles
- **SVD Analysis**: Spectral norm, nuclear norm, energy capture ratios, top singular values

**Usage Examples**:
```python
# Analyze and display results sorted by spectral norm
df = analyze_and_display(lora, device="cpu", dtype=torch.float32,
                         sort_by="dW.svd.spectral_norm", ascending=False)

# Find blocks covering 90% of importance
top90 = top_blocks_by_threshold(df, threshold=0.90)

# Get softmax-normalized importance scores
df["dW.softmax_norm"] = softmax_vals
```

**Key Functions**:
- `analyze_lora()`: Core analysis function that processes all LoRA layers
- `block_importance_from_softmax()`: Groups layers by transformer blocks and computes block importance
- `top_blocks_by_threshold()`: Finds minimal set of blocks for target importance threshold

## Use Cases

### LoRA Optimization
1. **Training Efficiency**: Use `find_important_layers.ipynb` to identify which layers contribute most to model performance
2. **Targeted Training**: Focus future training on the most important layers identified by the analysis
3. **Model Compression**: Remove or reduce rank of less important layers

### LoRA Blending
1. **Style Mixing**: Use `blend_anything.ipynb` to combine different LoRA styles with custom ratios
2. **Rank Optimization**: Convert LoRAs to different ranks during blending for optimal performance
3. **Memory Management**: Process large LoRAs efficiently with streaming approach

### Research and Analysis
1. **Layer Importance Studies**: Understand which parts of the model are most affected by LoRA adaptation
2. **Architecture Analysis**: Compare importance patterns across different model architectures
3. **Training Insights**: Identify patterns in successful vs. unsuccessful LoRA training

## Requirements

- PyTorch
- safetensors
- pandas
- numpy
- tqdm
- Jupyter notebook environment

## Notes

- Both notebooks are designed to work with LoRA models from the same base architecture
- Memory usage is optimized for large models, but performance may vary based on available hardware
- The analysis tools are particularly useful for understanding and optimizing LoRA training strategies
- Results can vary significantly between different models - some show 90%+ importance in a single layer, while others spread across 20+ layers
