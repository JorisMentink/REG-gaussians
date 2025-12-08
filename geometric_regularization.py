import torch
import torch.nn.functional as F
from simple_knn._C import distCUDA2

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

def velocity_coherence_loss(means3D, velocities, k=5):
    """
    Compute velocity coherence loss using k-nearest neighbors.

    Args:
        means3D (torch.Tensor): Tensor of shape (N, 3) representing Gaussian positions.
        velocities (torch.Tensor): Tensor of shape (N, 3) representing Gaussian velocities.
        k (int): Number of nearest neighbors to consider.

    Returns:
        torch.Tensor: Scalar loss value.
    """
    # Use the GPU-accelerated k-NN module to find k-nearest neighbors
    distances, indices = distCUDA2(means3D, means3D, k)  # distances: (N, k), indices: (N, k)

    # Gather the velocities of the k-nearest neighbors
    neighbor_velocities = velocities[indices]  # Shape: (N, k, 3)

    # Compute velocity differences
    velocity_diff = neighbor_velocities - velocities.unsqueeze(1)  # Shape: (N, k, 3)

    # Compute the squared norm of velocity differences
    velocity_diff_norm = torch.norm(velocity_diff, dim=-1)  # Shape: (N, k)

    # Compute the loss as the mean of squared differences
    loss = torch.mean(velocity_diff_norm ** 2)
    return loss
