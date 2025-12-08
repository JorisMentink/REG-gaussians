from Hexplane.regulation import compute_plane_smoothness
import torch
import numpy as np
import torch.nn.functional as F

def plane_regulation_from_grids(multi_res_grids):
    total = 0.0
    for grids in multi_res_grids:
        # grids length 3 -> pure spatial, else spatio-temporal layout used in this repo
        if len(grids) == 3:
            time_grids = []
        else:
            time_grids = [0, 1, 3]
        for grid_id in time_grids:
            total += compute_plane_smoothness(grids[grid_id])
    return total

def time_regulation_from_grids(multi_res_grids):
    total = 0.0
    for grids in multi_res_grids:
        if len(grids) == 3:
            time_grids = []
        else:
            time_grids = [2, 4, 5]
        for grid_id in time_grids:
            total += compute_plane_smoothness(grids[grid_id])
    return total

def l1_regulation_from_grids(multi_res_grids):
    total = 0.0
    for grids in multi_res_grids:
        if len(grids) == 3:
            continue
        else:
            spatiotemporal_grids = [2, 4, 5]
        for grid_id in spatiotemporal_grids:
            total += torch.abs(1.0 - grids[grid_id]).mean()
    return total

def grid_smoothness_loss(deformation_network, time_smoothness_weight, l1_time_planes_weight, plane_tv_weight):
    """
    Returns scalar regularization loss compatible with original repo.
    model must have attribute: model._deformation.deformation_net.grid.grids
    """
    # # safe fallback if model lacks the expected structure
    # if not hasattr(deformation_network.grid, "grids"):
    #     return torch.tensor(0.0, device=next(deformation_network.parameters()).device)

    multi_res_grids = deformation_network.deformation_net.grid.grids
    plane_term = plane_regulation_from_grids(multi_res_grids)
    time_term = time_regulation_from_grids(multi_res_grids)
    l1_term = l1_regulation_from_grids(multi_res_grids)
    return plane_tv_weight * plane_term + time_smoothness_weight * time_term + l1_time_planes_weight * l1_term


def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def geometric_regularization_loss(scales, r_m=1.0, r_ani=3.0, reduction='mean'):
    """
    Compute the geometric regularization loss to prevent excessively large
    or anisotropic Gaussian kernels, as described in Eq. (6).

    Args:
        scales (torch.Tensor): Tensor of shape [N, 3] containing scaling values S_i.
        r_m (float): Maximum allowed scale (default = 1.0).
        r_ani (float): Maximum allowed anisotropy ratio (default = 3.0).
        reduction (str): 'mean' or 'sum' reduction across Gaussians.

    Returns:
        torch.Tensor: Scalar geometric regularization loss.
    """
    # Ensure input is a tensor
    if not torch.is_tensor(scales):
        scales = torch.tensor(scales, dtype=torch.float32)

    # Max and min scaling factors per Gaussian
    max_s, _ = torch.max(scales, dim=1)
    min_s, _ = torch.min(scales, dim=1)

    # Penalize too large scales
    loss_scale = F.relu(max_s - r_m)

    # Penalize anisotropy (ratio between largest and smallest axis)
    loss_aniso = F.relu((max_s / min_s) - r_ani)

    # Combine losses
    loss = loss_scale + loss_aniso

    # Aggregate
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()

    return loss

def velocity_coherence_loss(means3D, velocities, k=5, chunk_size=1024, top_percentage=0.1):
    """
    Compute velocity coherence loss using top X% fastest Gaussians.
    """
    N = means3D.size(0)
    device = means3D.device

    # Step 1: Calculate how many Gaussians to keep (top 30%)
    num_to_keep = max(k * 10, int(N * top_percentage))  # At least 10*k for meaningful neighbors
    num_to_keep = min(num_to_keep, N)  # Can't exceed total Gaussians
    
    if num_to_keep < k:
        print(f"Skipping velocity coherence: only {N} total Gaussians (need at least {k})")
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # Step 2: Get velocity magnitudes and find top performers
    velocity_magnitudes = torch.norm(velocities, dim=-1)
    max_velocity = torch.max(velocity_magnitudes)
    
    # Get indices of top num_to_keep fastest Gaussians
    _, top_indices = torch.topk(velocity_magnitudes, num_to_keep, largest=True)
    
    # Print diagnostics
    min_selected_velocity = velocity_magnitudes[top_indices].min().item()
    print(f"Velocity coherence: Using top {num_to_keep}/{N} ({top_percentage*100:.0f}%) fastest Gaussians")
    print(f"  Max velocity: {max_velocity:.4f}, Min selected: {min_selected_velocity:.4f}")
    
    # Step 3: Filter to top performers
    filtered_means3D = means3D[top_indices]
    filtered_velocities = velocities[top_indices]
    M = filtered_means3D.size(0)
    
    total_loss = 0.0
    total_points = 0

    # Step 4: Process in chunks
    for start in range(0, M, chunk_size):
        end = min(start + chunk_size, M)
        
        chunk = filtered_means3D[start:end]
        pairwise_distances = torch.cdist(chunk, filtered_means3D, p=2)
        
        knn_distances, knn_indices = torch.topk(pairwise_distances, k=k+1, largest=False, dim=-1)
        knn_indices = knn_indices[:, 1:]  # Exclude self
        
        neighbor_velocities = filtered_velocities[knn_indices]
        chunk_velocities = filtered_velocities[start:end].unsqueeze(1)
        
        # Step 5: Normalize velocities for cosine similarity
        chunk_normalized = F.normalize(chunk_velocities, p=2, dim=-1)
        neighbor_normalized = F.normalize(neighbor_velocities, p=2, dim=-1)
        
        # Dot product gives cosine similarity
        cosine_similarity = torch.sum(chunk_normalized * neighbor_normalized, dim=-1)
        
        # Convert to loss: maximize cosine similarity = minimize negative cosine
        cosine_loss = 1.0 - cosine_similarity  # Range [0, 2], 0 = perfect alignment
        
        total_loss += torch.sum(cosine_loss)
        total_points += chunk.size(0) * k

    return total_loss / total_points if total_points > 0 else torch.tensor(0.0, device=device, requires_grad=True)