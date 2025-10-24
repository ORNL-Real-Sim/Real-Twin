import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import kmeans_plusplus


def generateRandom(x_mins, x_maxs, num_points, rng=None):
    """
    Generate random points with each dimension between corresponding min and max values.

    Args:
        x_mins: Array of minimum values for each dimension
        x_maxs: Array of maximum values for each dimension
        num_points: Number of points to generate
        rng: Optional numpy.random.Generator instance for reproducibility

    Returns:
        Array of shape (num_points, len(x_mins)) with random points
    """
    dims = len(x_mins)

    # Use a Generator for random numbers (recommended over legacy numpy.random)
    if rng is None:
        rng = np.random.default_rng(seed=812)

    # Leverage broadcasting: rng.uniform accepts array-like low/high and will broadcast to the output shape
    x = rng.uniform(x_mins, x_maxs, size=(num_points, dims))

    return x


def get_furthest_points(points, k):
    """
    Return k points that are furthest from each other in space using a greedy approach.
    Optimized for large inputs using vectorized operations.

    Args:
        points: Array of shape (n, d) where n is the number of points and d is the dimension
        k: Number of points to select

    Returns:
        Indices of the k selected points
    """
    n = len(points)

    if k > n:
        raise ValueError(
            "k cannot be greater than the number of points provided.")

    if k == n:
        return list(range(n))

    if k <= 0:
        return []

    if n <= 60000:  # For very large datasets, use a more efficient method

        # Calculate all pairwise distances once
        distances = squareform(pdist(points))

        # Start with the two points that are furthest apart
        i, j = np.unravel_index(np.argmax(distances), distances.shape)

        selected = [i, j]
        selected_set = set(selected)  # For O(1) lookups

        # Create a mask for unselected points
        mask = np.ones(n, dtype=bool)
        mask[selected] = False

        while len(selected) < k:
            # Use different strategies based on selection size for best performance
            if len(selected) > 10000:  # For very large selections
                # Only calculate distances to the newly added point
                new_distances = distances[selected[-1], mask]
                if len(selected) == 2:  # First iteration after initial two points
                    min_distances = np.minimum(
                        distances[selected[0], mask], new_distances)
                else:
                    min_distances = np.minimum(min_distances, new_distances)
            else:
                # For smaller selections, this vectorized approach is faster
                # Calculate min distance from each unselected point to any selected point
                min_distances = np.min(
                    distances[np.ix_(selected, mask)], axis=0)

            # Find the point with maximum minimum distance
            max_idx = np.argmax(min_distances)
            next_point = np.arange(n)[mask][max_idx]

            # Add the point to selected
            selected.append(next_point)
            selected_set.add(next_point)

            # Update the mask
            mask[next_point] = False

        return selected
    else:
        centers, indices = kmeans_plusplus(
            points, n_clusters=k, random_state=0)
        selected = list(indices[:k])
        return selected


def main():
    # Generate random points in 11 dimensions, each between 0 and 1
    num_points = 8100
    dims = 3
    x_mins = np.zeros(dims)
    x_maxs = np.ones(dims)

    # Generate random points
    x_random = generateRandom(x_mins, x_maxs, num_points)

    # Get the points that are furthest apart
    k = 30
    selected_indices = get_furthest_points(x_random, k)
    selected_points = x_random[selected_indices]

    # # Define parameter ranges to map to
    # param_ranges = [
    #     [6, 9.8],      # Dimension 0
    #     [0.6, 12.3],   # Dimension 1
    #     [0.9, 14],     # Dimension 2
    #     [-20, -3],     # Dimension 3
    #     [-20, -3],     # Dimension 4
    #     [-7.38, -2.5], # Dimension 5
    #     [-6, -1.5],    # Dimension 6
    #     [50, 200],     # Dimension 7
    #     [50, 200],     # Dimension 8
    #     [0.55, 1],     # Dimension 9
    #     [1.5, 6]       # Dimension 10
    # ]

    # param_ranges = [
    #     [0, 1],      # Dimension 0
    #     [0, 1],      # Dimension 1
    #     [0, 1],      # Dimension 2
    #     [0, 1],      # Dimension 3
    #     [0, 1],      # Dimension 4
    #     [0, 1],      # Dimension 5
    #     [0, 1],      # Dimension 6
    #     [0, 1],      # Dimension 7
    #     [50, 200],     # Dimension 8
    #     [50, 200],     # Dimension 9
    #     [50, 200],     # Dimension 10
    #     [50, 200]      # Dimension 11
    # ]

    # Define parameter ranges to map to
    param_ranges = [
        [0, 1],      # Dimension 0
        [0, 1],      # Dimension 1
        [0, 1],      # Dimension 2
    ]

    # Map the random points to the parameter ranges
    mapped_points = selected_points.copy()
    for i in range(len(param_ranges)):
        min_val, max_val = param_ranges[i]
        mapped_points[:, i] = selected_points[:, i] * \
            (max_val - min_val) + min_val

    # Print the selected points to a csv file
    np.savetxt("test_low_d.csv", mapped_points, delimiter=",")


if __name__ == "__main__":
    main()
