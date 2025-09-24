import torch
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.utils import to_networkx
import networkx as nx
from skimage.draw import line
from torch_geometric.data import Data

@torch.datasets.register("SuperPixelGraph")
class SuperpixelGraphDataset(Dataset):
    """
    Converts MNISTSuperpixels into padded graph format:
    - Each superpixel = node
    - Edges exist only if present in the ground truth graph
    - Pads adjacency and features to MAX_NODES
    """
    def __init__(self, root="./data", max_nodes=256, num_features_per_node=2, train=True):
        super().__init__()
        self.dataset = MNISTSuperpixels(root=root, train=train)
        self.MAX_NODES = max_nodes
        self.NUM_FEATURES_PER_NODE = num_features_per_node

    def __len__(self):
        return len(self.dataset)

    def graph_to_image(self, adj, mask, pixel_pos, img_size=256):
        # Convert safely to numpy
        adj = adj.detach().cpu().numpy()
        mask = mask.detach().cpu().numpy().squeeze()
        pixel_pos = pixel_pos.detach().cpu().numpy()

        # Create blank image
        img = np.zeros((img_size, img_size), dtype=np.float32)

        # Get valid nodes
        valid_idx = np.where(mask > 0)[0]
        if len(valid_idx) == 0:
            return torch.from_numpy(img), torch.from_numpy(pixel_pos)

        coords = pixel_pos[valid_idx]

        # Normalize to [0,1]
        coords = coords / (coords.max() + 1e-6)

        # Scale to image space
        coords = (coords * (img_size - 1)).astype(int)

        # --- Draw edges ---
        for i in valid_idx:
            for j in valid_idx:
                if adj[i, j] > 0:
                    x0, y0 = coords[np.where(valid_idx == i)[0][0]]
                    x1, y1 = coords[np.where(valid_idx == j)[0][0]]
                    rr, cc = line(x0, y0, x1, y1)
                    img[rr, cc] = 1.0

        return torch.from_numpy(img), torch.from_numpy(coords)

    def __getitem__(self, idx):
        pyg_data = self.dataset[idx]

        # Mask surrogate
        mask_surrogate = (pyg_data.x > 0).int()

        # <------ Building graph -------->
        G = to_networkx(pyg_data, node_attrs=["pos"])
        num_nodes = mask_surrogate.sum().item()

        # <--- Building mask --->
        mask = np.zeros((self.MAX_NODES, 1), dtype=np.float32)
        mask[:num_nodes] = mask_surrogate[:num_nodes].cpu().numpy()

        # --- Build adjacency ---
        node_list = list(G.nodes())[:num_nodes]
        node_idx_map = {node: i for i, node in enumerate(node_list)}
        adj_no_pad = np.zeros((num_nodes, num_nodes), dtype=np.float32)

        for u, v in G.edges():
            if u in node_idx_map and v in node_idx_map:
                i, j = node_idx_map[u], node_idx_map[v]
                adj_no_pad[i, j] = 1
                adj_no_pad[j, i] = 1

        padded_adj = np.zeros((self.MAX_NODES, self.MAX_NODES), dtype=np.float32)
        padded_adj[:num_nodes, :num_nodes] = adj_no_pad

        # --- Node positions ---
        node_pos_matrix = np.array([G.nodes[n]["pos"] for n in node_list])
        pixel_pos_padded = np.zeros((self.MAX_NODES, self.NUM_FEATURES_PER_NODE), dtype=np.float32)
        pixel_pos_padded[:num_nodes] = node_pos_matrix

        default_features = pixel_pos_padded.copy()

        # Convert everything to tensors before passing to graph_to_image
        image_tensor, pixel_pos_scaled = self.graph_to_image(
            torch.from_numpy(padded_adj),
            torch.from_numpy(mask),
            torch.from_numpy(pixel_pos_padded),
        )

        pixel_pos_padded[:num_nodes] = pixel_pos_scaled.numpy() + 1  # +1 for position encoding

        # Label
        label = pyg_data.y.clone().detach().long()

        return (
            torch.from_numpy(padded_adj) * torch.from_numpy(mask).int(),
            torch.from_numpy(mask).int(),
            torch.from_numpy(pixel_pos_padded).int(),
            image_tensor.float(),
            default_features,
            label
        )
    
@torch.datasets.register("SuperPixelGraphDefault")
class SuperpixelDefaultDataset(Dataset):
    """
    Converts MNISTSuperpixels into padded graph format:
    - Each superpixel = node
    - Edges exist only if present in the ground truth graph
    - Pads adjacency and features to MAX_NODES
    Also returns the original MNIST 28x28 image.
    """
    def __init__(self, root="./data", max_nodes=256, num_features_per_node=2, train=True, transform=None):
        super().__init__()
        self.dataset = MNISTSuperpixels(root=root, train=train, transform=transform)
        self.MAX_NODES = max_nodes
        self.NUM_FEATURES_PER_NODE = num_features_per_node

    def __len__(self):
        return len(self.dataset)

    def graph_to_image(self, adj, mask, pixel_pos, img_size=28):
        adj = adj.detach().cpu().numpy()
        mask = mask.detach().cpu().numpy().squeeze()
        pixel_pos = pixel_pos.detach().cpu().numpy()

        img = np.zeros((img_size, img_size), dtype=np.float32)

        valid_idx = np.where(mask > 0)[0]
        if len(valid_idx) == 0:
            return torch.from_numpy(img), torch.from_numpy(pixel_pos)

        coords = pixel_pos[valid_idx]

        coords = coords / (coords.max() + 1e-6)
        coords = (coords * (img_size - 1)).astype(int)

        for i in valid_idx:
            for j in valid_idx:
                if adj[i, j] > 0:
                    x0, y0 = coords[np.where(valid_idx == i)[0][0]]
                    x1, y1 = coords[np.where(valid_idx == j)[0][0]]
                    rr, cc = line(x0, y0, x1, y1)
                    img[rr, cc] = 1.0

        return torch.from_numpy(img), torch.from_numpy(coords)

    def __getitem__(self, idx):
        pyg_data = self.dataset[idx]
    
        # keep node positions as features
        x = pyg_data.pos.float()  # [num_nodes, 2]
        edge_index = pyg_data.edge_index  # already exists
        y = pyg_data.y
    
        data = Data(x=x, edge_index=edge_index, y=y)
        return data