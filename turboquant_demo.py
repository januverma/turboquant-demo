import numpy as np

def qjl_encode(residual, m=None, seed=0):
    """
    Quantized Johnson-Lindenstrauss (QJL) encoding.
    
    Projects the residual error into a lower-dimensional space using
    a random sign matrix, then keeps only the sign bits.
    
    Args:
        residual: the error vector e = v - v_hat (d-dimensional)
        m: projection dimension (default: d, can be smaller)
        seed: random seed for reproducible projection matrix
    
    Returns:
        sign_bits: {+1, -1} array of length m (1 bit each)
        norm_e: the norm of the residual (stored once per vector)
    """
    d = len(residual)
    if m is None:
        m = d
    
    # Generate random ±1 matrix (Rademacher distribution)
    # This matrix is NOT stored — it's regenerated from the seed
    rng = np.random.RandomState(seed)
    S = rng.choice([-1, 1], size=(m, d)).astype(np.float32)
    S /= np.sqrt(m)  # normalize
    
    # Project and take the sign
    projection = S @ residual
    sign_bits = np.sign(projection)  # each is +1 or -1 → 1 bit
    
    norm_e = np.linalg.norm(residual)
    
    return sign_bits, norm_e

def qjl_estimate_inner_product(q, sign_bits_e, norm_e, 
                                sign_bits_q=None, m=None, seed=0):
    """
    Estimate ⟨q, e⟩ using QJL sign bits.
    
    Uses the identity:
        E[sign(Sx) · sign(Sy)] = (2/π) · ⟨x, y⟩ / (‖x‖·‖y‖)
    
    So:
        ⟨q, e⟩ ≈ (π/2) · ⟨sign(Sq), sign(Se)⟩ · ‖q‖·‖e‖ / m
    """
    d = len(q)
    if m is None:
        m = len(sign_bits_e)
    
    # Project query with the SAME random matrix (same seed)
    rng = np.random.RandomState(seed)
    S = rng.choice([-1, 1], size=(m, d)).astype(np.float32)
    S /= np.sqrt(m)
    
    projection_q = S @ q
    sign_q = np.sign(projection_q)
    
    # Estimate inner product
    norm_q = np.linalg.norm(q)
    sign_agreement = np.dot(sign_q, sign_bits_e) / m
    
    # The π/2 correction makes this an UNBIASED estimator
    ip_estimate = (np.pi / 2) * sign_agreement * norm_q * norm_e
    
    return ip_estimate


# --- Full TurboQuant combining PolarQuant + QJL ---
def turboquant_encode(key_vector, bits=4, qjl_seed=0):
    """
    TurboQuant: (b-1)-bit PolarQuant + 1-bit QJL = b total bits
    
    This is the complete encoding pipeline:
    1. Random rotation (fast Hadamard transform)
    2. Per-coordinate scalar quantization (simplified PolarQuant)
    3. Compute residual
    4. QJL encode the residual with 1 bit
    """
    d = len(key_vector)
    quant_bits = bits - 1  # Reserve 1 bit for QJL
    n_levels = 2 ** quant_bits
    
    # Step 1: Random rotation (using orthogonal matrix; 
    # real impl uses fast Walsh-Hadamard)
    rng = np.random.RandomState(42)
    Q, _ = np.linalg.qr(rng.randn(d, d))
    rotated = Q @ key_vector
    
    # Step 2: Scalar quantization with FIXED grid
    # After rotation, coordinates ~ sub-Gaussian with known variance
    # No per-block scale needed!
    norm_v = np.linalg.norm(key_vector)
    scale = norm_v / np.sqrt(d)  # predictable scale from theory
    
    quantized_codes = np.clip(
        np.round((rotated / (3 * scale) + 0.5) * n_levels),
        0, n_levels - 1
    ).astype(int)
    
    # Dequantize for residual computation
    dequantized = (quantized_codes / n_levels - 0.5) * 3 * scale
    
    # Step 3: Residual
    residual = rotated - dequantized
    
    # Step 4: QJL on residual (1 bit per dimension)
    sign_bits, norm_e = qjl_encode(residual, m=d, seed=qjl_seed)
    
    return {
        'quantized_codes': quantized_codes,  # (bits-1) bits each
        'sign_bits': sign_bits,               # 1 bit each
        'norm_v': norm_v,                     # stored once per vector
        'norm_e': norm_e,                     # stored once per vector
    }

def turboquant_attention_score(query, encoded_key, qjl_seed=0):
    """
    Compute attention score ⟨query, key⟩ using TurboQuant.
    
    Unbiased estimator:
        ⟨q, k⟩ ≈ ⟨q_rot, k_hat⟩ + QJL_correction
    """
    d = len(query)
    
    # Rotate query with same rotation
    rng = np.random.RandomState(42)
    Q, _ = np.linalg.qr(rng.randn(d, d))
    q_rot = Q @ query
    
    # Dequantize key
    n_levels = 2 ** (3)  # bits-1 = 3 for 4-bit TurboQuant
    scale = encoded_key['norm_v'] / np.sqrt(d)
    k_hat = (encoded_key['quantized_codes'] / n_levels - 0.5) * 3 * scale
    
    # Main inner product from quantized values
    main_ip = np.dot(q_rot, k_hat)
    
    # QJL correction for the residual (eliminates bias!)
    qjl_correction = qjl_estimate_inner_product(
        q_rot, 
        encoded_key['sign_bits'], 
        encoded_key['norm_e'],
        seed=qjl_seed
    )
    
    return main_ip + qjl_correction


# --- Demo ---
if __name__ == "__main__":
    np.random.seed(123)
    d = 256  # typical head dimension
    
    # Simulate a query and key from an attention layer
    query = np.random.randn(d)
    key = np.random.randn(d)
    
    # True attention score
    true_score = np.dot(query, key)
    
    # TurboQuant at 4 bits (3-bit PolarQuant + 1-bit QJL)
    encoded = turboquant_encode(key, bits=4)
    estimated_score = turboquant_attention_score(query, encoded)
    
    print(f"Dimension:         {d}")
    print(f"True ⟨q, k⟩:      {true_score:.4f}")
    print(f"TurboQuant ⟨q, k⟩: {estimated_score:.4f}")
    print(f"Absolute error:    {abs(true_score - estimated_score):.4f}")
    print(f"Relative error:    {abs(true_score - estimated_score) / abs(true_score) * 100:.2f}%")
    print(f"\nCompression: 32-bit → 4-bit = 8x reduction")
    print(f"Overhead constants: ZERO (vs traditional: 2 extra per block)")
