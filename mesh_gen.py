import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay, distance
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon, Point


def distribute_uniform_boundary_points(boundary, n_points):
    """
    Distributes points uniformly along the boundary of a shape.
    """
    boundary = np.vstack([boundary, boundary[0]])  # Close the contour
    total_length = np.sum(np.sqrt(np.sum(np.diff(boundary, axis=0) ** 2, axis=1)))
    spacing = total_length / n_points
    boundary_points = []
    current_length = 0
    i = 0

    while len(boundary_points) < n_points:
        start = boundary[i]
        end = boundary[i + 1]
        segment_length = np.linalg.norm(end - start)

        while current_length + segment_length >= spacing:
            t = (spacing - current_length) / segment_length
            new_point = start + t * (end - start)
            boundary_points.append(new_point)
            current_length = 0
            start = new_point
            segment_length = np.linalg.norm(end - start)

        current_length += segment_length
        i += 1
        if i >= len(boundary) - 1:
            break

    if np.linalg.norm(boundary_points[0] - boundary_points[-1]) > spacing / 2:
        boundary_points.append(boundary_points[0])  # Ensure closure
    return np.array(boundary_points)


def poisson_disk_sampling(boundary, boundary_points, min_dist, num_samples, min_dist_to_boundary):
    """
    Generates interior points while maintaining a minimum distance from boundaries.
    """
    polygon = Polygon(boundary)
    bbox = polygon.bounds
    points = []

    while len(points) < num_samples:
        x = np.random.uniform(bbox[0], bbox[2])
        y = np.random.uniform(bbox[1], bbox[3])
        candidate = Point(x, y)

        if polygon.contains(candidate):
            if len(points) == 0 or np.min(cdist([candidate.coords[0]], points)) > min_dist:
                if np.min(cdist([candidate.coords[0]], boundary_points)) > min_dist_to_boundary:
                    points.append([x, y])

    return np.array(points)


def find_neighbors(point_index, triangles):
    """
    Finds neighboring points for a given index in a triangulation.
    """
    neighbors = set()
    for tri in triangles:
        if point_index in tri:
            neighbors.update(tri)

    neighbors.discard(point_index)  # Remove self
    return list(neighbors)


def laplacian_smoothing(points, triangles, boundary_points, iterations=10):
    """
    Smooths the mesh using Laplacian smoothing while preserving boundary points.
    """
    for _ in range(iterations):
        new_points = np.copy(points)

        for i in range(len(points)):
            if i < len(boundary_points):  # Preserve boundary points
                continue
            neighbors = find_neighbors(i, triangles)
            if neighbors:
                new_points[i] = np.mean(points[neighbors], axis=0)

        points[:] = new_points
    return points


def process_markers(image_path, n_boundary_points, n_interior_points, min_distance, min_dist_to_boundary):
    """
    Processes an image to extract contours, generate points, perform triangulation, and visualize the results.
    """
    image = cv2.imread(image_path)

    # Display the original image
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    # Convert image to grayscale and binarize
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary_image = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    # Extract contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        boundary = contour.reshape(-1, 2)

        # Generate boundary and interior points
        boundary_points = distribute_uniform_boundary_points(boundary, n_boundary_points)
        interior_points = poisson_disk_sampling(boundary, boundary_points, min_distance, n_interior_points, min_dist_to_boundary)

        # Combine points and generate Delaunay triangulation
        all_points = np.vstack([boundary_points, interior_points])
        tri = Delaunay(all_points)

        # Create polygon and filter valid triangles
        polygon = Polygon(boundary)
        valid_triangles = [simplex for simplex in tri.simplices if polygon.contains(Point(all_points[simplex].mean(axis=0)))]

        # Apply smoothing
        all_points = laplacian_smoothing(all_points, valid_triangles, boundary_points)

        # Plot the final mesh
        plt.figure(figsize=(8, 8))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.plot(boundary_points[:, 0], boundary_points[:, 1], 'ro-', label="Boundary Points")
        plt.triplot(all_points[:, 0], all_points[:, 1], valid_triangles, color='blue')
        plt.scatter(all_points[:, 0], all_points[:, 1], color='k', s=5)
        plt.axis('off')
        plt.legend()
        plt.show()


# Image paths
image_paths = [
    ("forma1.png", 20, 15, 10, 15),
    ("forma2.png", 35, 25, 10, 10),
    ("forma3.png", 30, 20, 10, 10),
    ("forma4.png", 35, 15, 10, 10),
    ("forma5.png", 40, 15, 10, 10),
    ("forma6.png", 40, 20, 10, 10),
]

# Process each image
for img_path, n_boundary, n_interior, min_dist, min_dist_boundary in image_paths:
    process_markers(img_path, n_boundary, n_interior, min_dist, min_dist_boundary)