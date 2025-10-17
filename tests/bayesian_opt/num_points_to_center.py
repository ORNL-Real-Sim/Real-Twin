import numpy as np
from scipy.spatial.distance import pdist, squareform

def get_furthest_points_till_center(points, center):
    """
    Add points that are furthest from each other in space, till reaching the center. Use a greedy approach.
    Optimized for large inputs using vectorized operations.
    
    Args:
        points: Array of shape (n, d) where n is the number of points and d is the dimension
        center: Center point in the same dimension as points
        
    Returns:
        Indices of the selected points, total number of points selected
    """
    n = len(points)
    
    if n == 0:
        return [], 0
    
    if len(center) != points.shape[1]:
        raise ValueError("Center point dimension must match points dimension.")
    
    # Calculate all pairwise distances once - this is efficient for large datasets
    distances = squareform(pdist(points))
    
    # Start with the two points that are furthest apart
    i, j = np.unravel_index(np.argmax(distances), distances.shape)
    selected = [i, j]
    selected_set = set(selected)  # Using a set for O(1) lookups
    
    # Calculate center distances once
    center_distances = np.linalg.norm(points - center, axis=1)
    center_threshold = 1e-5
    
    # Check if we've already reached the center with initial points
    if center_distances[i] < center_threshold or center_distances[j] < center_threshold:
        return selected, len(selected)
    
    # Create a mask for unselected points - much faster than checking membership in a loop
    mask = np.ones(n, dtype=bool)
    mask[selected] = False
    
    while len(selected) < n:
        # Compute minimum distances using vectorized operations
        # For each unselected point, find its minimum distance to any selected point
        if len(selected) > 10000:  # For very large selections, use a different approach
            # Calculate distances to newly added point only
            new_distances = distances[selected[-1], mask]
            # Update minimum distances
            if len(selected) == 2:  # First iteration after initial two points
                min_distances = np.minimum(distances[selected[0], mask], new_distances)
            else:
                min_distances = np.minimum(min_distances, new_distances)
        else:
            # For smaller selections, this vectorized approach is faster
            min_distances = np.min(distances[np.ix_(selected, mask)], axis=0)
        
        # Find the point with maximum minimum distance
        max_idx = np.argmax(min_distances)
        next_point = np.arange(n)[mask][max_idx]
        
        # Add the point to selected
        selected.append(next_point)
        selected_set.add(next_point)
        
        # Update the mask
        mask[next_point] = False
        
        # Check if we've reached the center
        if center_distances[next_point] < center_threshold:
            break
    
    return selected, len(selected)

def main():
    # Example usage
    dimension = 3  # Dimension of the points
    #create points in {0, 0.5, 1}^dimension on a grid
    points = np.array(np.meshgrid(*[[0, 0.5, 1]]*dimension)).T.reshape(-1, dimension)
    center = np.array([0.5]*dimension)  # Center point
    print(points.shape)  # Should be (3^dimension, dimension)
    # Get the points that are furthest apart
    selected_indices, num_selected = get_furthest_points_till_center(points, center)
    # print("Selected indices:", selected_indices)
    print("Number of selected points:", num_selected)
    # Print the selected points
    selected_points = points[selected_indices]
    print("Selected points:\n", selected_points)

if __name__ == "__main__":
    main()
    
