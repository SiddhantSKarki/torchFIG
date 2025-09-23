import cv2
import torch

import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import line


def graph_to_2d_image_from_dataset(adj, features, mask, scale=30, edge_width=1):
    """
    Convert a padded graph from SuperpixelGraphDataset into a 2D image
    exactly like superpixels_to_2d_image does.
    """
    if isinstance(adj, torch.Tensor):
        adj = adj.numpy()
    if isinstance(features, torch.Tensor):
        features = features.numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()

    # Keep only actual nodes
    valid_nodes = mask.squeeze() > 0
    adj = adj[valid_nodes][:, valid_nodes]
    features = features[valid_nodes]
    num_nodes = len(features)

    # Scale positions
    pos = (features * scale).astype(int)
    image = np.zeros((scale * 26, scale * 26, 1), dtype=np.uint8)

    # Draw nodes as rectangles
    for x, y in pos:
        x0, y0 = int(x), int(y)
        x1, y1 = x0 - scale, y0 - scale
        color = 255  # pure white node
        cv2.rectangle(image, (x0, y0), (x1, y1), color, -1)

    # Draw edges
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adj[i, j] > 0:
                x0, y0 = pos[i]
                x1, y1 = pos[j]
                x0 -= scale // 2
                y0 -= scale // 2
                x1 -= scale // 2
                y1 -= scale // 2
                cv2.line(image, (x0, y0), (x1, y1), 125, edge_width)

    return image

def visualize_custom_dataset(
    dataset,
    image_name: str,
    examples_per_class: int = 10,
    classes: tuple[int, ...] = tuple(range(10)),
    figsize: tuple[int,int] = (50,50),
    edge_width: int = 1
):
    """
    Visualize a SuperpixelGraphDataset just like the original visualize function.
    """
    class_to_examples = {class_ix: [] for class_ix in classes}

    for adj, mask, features_pos, _, features , label in dataset:
        class_ix = int(label)

        if class_ix not in class_to_examples:
            continue

        if len(class_to_examples[class_ix]) >= examples_per_class:
            continue

        # Convert graph to 2D image
        img = graph_to_2d_image_from_dataset(adj, features, mask, 45, edge_width=edge_width)
        class_to_examples[class_ix].append(img)

        # Stop if we have enough examples for all classes
        enough = all(len(examples) >= examples_per_class for examples in class_to_examples.values())
        if enough:
            break

    # Plot grid
    plt.figure(figsize=figsize)
    subplot_ix = 1
    for class_ix in classes:
        for example in class_to_examples[class_ix]:
            plt.subplot(len(classes), examples_per_class, subplot_ix)
            subplot_ix += 1
            plt.imshow(example, cmap=plt.cm.binary)
            plt.axis('off')
    plt.savefig(image_name)
