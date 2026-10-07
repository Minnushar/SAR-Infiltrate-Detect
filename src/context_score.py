import numpy as np


def calculate_context_score(score_grid):
    """
    Calculate how unusual each patch is compared
    with its surrounding patches.

    Input:
        score_grid: 2D array of anomaly scores (8x8)

    Output:
        context_score: 2D array of local-context scores
    """

    height, width = score_grid.shape

    context_score = np.zeros_like(score_grid, dtype=np.float32)

    for y in range(height):
        for x in range(width):

            neighbors = []

            # Check surrounding 8 positions
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:

                    if dy == 0 and dx == 0:
                        continue

                    ny = y + dy
                    nx = x + dx

                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                    ):
                        neighbors.append(
                            score_grid[ny, nx]
                        )

            # Average neighboring anomaly score
            neighborhood_mean = np.mean(neighbors)

            # Difference from local neighborhood
            context_score[y, x] = abs(
                score_grid[y, x] - neighborhood_mean
            )

    return context_score


if __name__ == "__main__":

    # Example 8x8 anomaly score grid
    test_grid = np.random.rand(8, 8).astype(np.float32)

    context = calculate_context_score(test_grid)

    print("Input score grid shape:", test_grid.shape)
    print("Context score grid shape:", context.shape)
    print("Minimum context score:", context.min())
    print("Maximum context score:", context.max())