# SAE Training Batch Size Explanation

## The Two Types of "Batch Size" in SAE Training

### 1. Transformer Model Batches
- **`model_batch_size = 512`**: Number of text sequences
- **`seq_len = 128`**: Tokens per sequence  
- **Total tokens per model forward pass**: 512 × 128 = 65,536 tokens

### 2. SAE Training Batches  
- **`batch_size = 4096`**: Number of individual activation vectors
- Each activation vector comes from one token position
- SAE sees individual 768-dimensional vectors, not sequences

## Example Training Step

```python
# 1. Model processes sequences
sequences = ["Hello world", "How are you", ...]  # 512 sequences
tokens = tokenize(sequences)  # Shape: (512, 128)

# 2. Extract activations at layer 8
activations = model(tokens)["resid_pre_8"]  # Shape: (512, 128, 768)

# 3. Flatten to individual activation vectors
flat_activations = activations.reshape(-1, 768)  # Shape: (65536, 768)

# 4. Sample batches for SAE training
sae_batch = sample(flat_activations, 4096)  # Shape: (4096, 768)
sae_output = sae(sae_batch)  # Train SAE on these 4096 vectors
```

## Why This Design?

**SAEs learn to reconstruct individual activation vectors**, not sequences. Each 768-dimensional vector represents the model's internal representation at one token position. The SAE doesn't need to know about sequence structure - it just learns: "given this activation vector, what sparse features explain it?"

## Training Budget Calculation

```python
num_tokens = 1e9           # Process 1 billion tokens total
batch_size = 4096          # SAE sees 4096 activations per step  
num_batches = 1e9 // 4096  # = 244,140 SAE training steps
```

This means we'll do ~244k SAE training steps to process 1 billion tokens worth of activations. 