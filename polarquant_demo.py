import numpy as np

def cartesian_to_polar_pair(x1, x2):
    """Convert a pair of Cartesian coordinates to polar (radius, angle)."""
    r = np.sqrt(x1**2 + x2**2)
    theta = np.arctan2(x2, x1)  # angle in [-pi, pi]
    return r, theta

def random_rotation(v, seed=42):
    """Apply randomized Hadamard transform (simplified as random orthogonal)."""
    rng = np.random.RandomState(seed)
    # In practice, use fast Walsh-Hadamard with random sign flips
    # Here we simulate with a random orthogonal matrix
    d = len(v)
    Q, _ = np.linalg.qr(rng.randn(d, d))
    return Q @ v

def polarquant_encode(v, bits_per_angle=3, seed=42):
    """
    PolarQuant encoding:
    1. Randomly rotate the vector
    2. Pair coordinates → polar (radius, angle)
    3. Recursively pair radii until one final radius remains
    4. Quantize all angles to a fixed grid (no per-block constants!)
    """
    # Step 1: Random rotation spreads energy evenly across coordinates
    v_rot = random_rotation(v, seed)
    
    # Step 2: Recursive polar conversion
    coords = v_rot.copy()
    all_angles = []
    
    while len(coords) > 1:
        new_radii = []
        for i in range(0, len(coords), 2):
            if i + 1 < len(coords):
                r, theta = cartesian_to_polar_pair(coords[i], coords[i+1])
                new_radii.append(r)
                all_angles.append(theta)
            else:
                new_radii.append(coords[i])  # odd element passes through
        coords = np.array(new_radii)
    
    final_radius = coords[0]  # = ||v|| (the vector norm)
    
    # Step 3: Quantize angles to a FIXED grid
    # Key insight: after random rotation, angles follow a known Beta distribution
    # so we can use a universal grid — no per-block scale/zero needed!
    n_levels = 2 ** bits_per_angle
    quantized_angles = []
    for theta in all_angles:
        # Normalize angle to [0, 1] range
        normalized = (theta + np.pi) / (2 * np.pi)
        # Uniform quantization (optimal for the concentrated distribution)
        level = int(np.clip(normalized * n_levels, 0, n_levels - 1))
        quantized_angles.append(level)
    
    return final_radius, quantized_angles, bits_per_angle

def polarquant_decode(final_radius, quantized_angles, bits_per_angle, seed=42):
    """Reconstruct the vector from quantized polar representation."""
    n_levels = 2 ** bits_per_angle
    
    # Dequantize angles
    angles = []
    for level in quantized_angles:
        normalized = (level + 0.5) / n_levels  # midpoint reconstruction
        theta = normalized * 2 * np.pi - np.pi
        angles.append(theta)
    
    # Reverse the recursive polar → Cartesian conversion
    # Start from the final radius and work backwards through the angle tree
    radii = [final_radius]
    angle_idx = len(angles) - 1
    
    while len(radii) < len(angles) + 1:
        new_coords = []
        for r in radii:
            if angle_idx >= 0:
                theta = angles[angle_idx]
                angle_idx -= 1
                x1 = r * np.cos(theta)
                x2 = r * np.sin(theta)
                new_coords.extend([x1, x2])
            else:
                new_coords.append(r)
        radii = new_coords
    
    v_rot_approx = np.array(radii)
    
    # Undo the random rotation
    rng = np.random.RandomState(seed)
    d = len(v_rot_approx)
    Q, _ = np.linalg.qr(rng.randn(d, d))
    return Q.T @ v_rot_approx

# --- Demo ---
if __name__ == "__main__":
    d = 8
    v = np.random.randn(d)
    v = v / np.linalg.norm(v) * 3.0  # scale to norm 3
    
    print(f"Original vector:     {v}")
    print(f"Original norm:       {np.linalg.norm(v):.4f}")
    
    radius, q_angles, bits = polarquant_encode(v, bits_per_angle=4)
    print(f"\nFinal radius:        {radius:.4f}")
    print(f"Quantized angles:    {q_angles}")
    print(f"Bits per angle:      {bits}")
    
    v_hat = polarquant_decode(radius, q_angles, bits)
    print(f"\nReconstructed:       {v_hat}")
    
    mse = np.mean((v - v_hat[:len(v)])**2)
    print(f"MSE:                 {mse:.6f}")
    print(f"Cosine similarity:   {np.dot(v, v_hat[:len(v)]) / (np.linalg.norm(v) * np.linalg.norm(v_hat[:len(v)])):.4f}")
