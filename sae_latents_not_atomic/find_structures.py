#%%% 
import torch
import wandb 
import os
import sys
sys.path.append("/workspace/SAELens")
from sae_lens.training.sparse_autoencoder import SparseAutoencoder
import json
import urllib.parse
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import tqdm




def load_legacy_saes(start=1, end=8):
    saes = []
    api = wandb.Api()
    for i in range(start, end + 1):
        wandb_link = f"jbloom/mats_sae_training_gpt2_feature_splitting_experiment/sparse_autoencoder_gpt2-small_blocks.8.hook_resid_pre_{768 * 2**(i-1)}:v9"

        artifact = api.artifact(
            wandb_link,
            type="model",
        )
        artifact_dir = artifact.download()
        file = os.listdir(artifact_dir)[0]
        sae = SparseAutoencoder.load_from_pretrained_legacy(
            os.path.join(artifact_dir, file)
        )
        saes.append(sae.to("cuda"))
    return saes


def load_sae(artifact_name, sae_class):
    # Initialize wandb
    api = wandb.Api()

    # Download the artifact
    artifact = api.artifact(artifact_name)
    artifact_dir = artifact.download()

    # Load the configuration
    config_path = os.path.join(artifact_dir, "config.json")
    with open(config_path, "r") as f:
        cfg = json.load(f)

    # Convert string representations back to torch.dtype
    if "dtype" in cfg:
        cfg["dtype"] = getattr(torch, cfg["dtype"].split(".")[-1])

    sae = sae_class(cfg)

    # Load the state dict
    state_dict_path = os.path.join(artifact_dir, "sae.pt")
    state_dict = torch.load(state_dict_path, map_location=cfg["device"])
    sae.load_state_dict(state_dict)

    return sae, cfg


josephs_saes = load_legacy_saes(start=1, end=7)
base_sae = josephs_saes[-1]  # You need to implement this function to load your base SAE
print(base_sae.W_dec.shape)
# %%
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.linalg import svd
import tqdm

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances

def custom_ransac_pca(X, min_samples, max_trials, threshold):
    best_inliers = []
    best_pca = None
    
    # Randomly select a subset of data points
    inliers = np.random.choice(X.shape[0], 10, replace=False)
    previous_inliers = None

    for _ in range(max_trials):
        subset = X[inliers]
        
        # Perform PCA on the subset
        pca = PCA(n_components=2)
        pca.fit(subset)
        
        # Reconstruct the data points using PCA
        reconstructed = pca.inverse_transform(pca.transform(X))
        
        # Calculate the cosine distances between the reconstructed points and the original points
        distances = []
        for i in tqdm.tqdm(range(X.shape[0])):
            distances.append(cosine_distances(X[i].reshape(1, -1), reconstructed[i].reshape(1, -1)))
        
        # Find the inlier points within e threshold
        distances = np.array(distances).flatten()
        inliers = np.where(distances < threshold)[0]
        inlier_distances = np.mean(distances[inliers])
        
        print(f"Number of inliers: {len(inliers)}, distances: {np.mean(inlier_distances)}, threshold: {threshold}")
        if len(inliers) > 20:
            threshold *= 0.8
        if len(inliers) < 10:
            threshold *= 1.2


        # Check if inliers are the same as in the previous iteration
        if previous_inliers is not None and np.array_equal(inliers, previous_inliers):
            print("Inliers are the same as in the previous iteration. Breaking the loop.")
            break    

        previous_inliers = inliers  
    return inliers, best_pca


# Example usage
X = base_sae.W_dec.detach().cpu().numpy()
min_samples = 10
max_trials = 100
threshold = 0.5

selected_indices, best_pca = custom_ransac_pca(X, min_samples, max_trials, threshold)
inlier_datapoints = X[selected_indices]

print(f"Number of inlier datapoints: {len(inlier_datapoints)}")


# %%

import requests
import json

def get_max_activating_token(model_id, layer, index):
    url = f"https://www.neuronpedia.org/api/feature/{model_id}/{layer}/{index}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data['activations'] and len(data['activations']) > 0:
            max_token = data['activations'][0]['tokens'][data['activations'][0]['maxValueTokenIndex']]
            return max_token
    return f"Feature {index}"



# Get the selected points
selected_points = X[selected_indices]

# Perform PCA on the selected points
pca = PCA(n_components=2)
selected_points_pca = pca.fit_transform(selected_points)

# Fetch max activating tokens
max_activating_tokens = []
for index in tqdm.tqdm(selected_indices):
    token = get_max_activating_token("gpt2-small", "8-res_fs49152-jb", str(index))
    max_activating_tokens.append(token)

# Create a scatter plot
plt.figure(figsize=(10, 8))
plt.scatter(selected_points_pca[:, 0], selected_points_pca[:, 1], color='red', s=100)

# Add labels to the points
for i, (x, y) in enumerate(selected_points_pca):
    plt.annotate(max_activating_tokens[i], (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8)

plt.title('PCA Visualization of Selected Points with Max Activating Tokens')
plt.xlabel(f'First Principal Component (Explained Variance: {pca.explained_variance_ratio_[0]:.4f})')
plt.ylabel(f'Second Principal Component (Explained Variance: {pca.explained_variance_ratio_[1]:.4f})')

# Add explained variance information
total_variance = sum(pca.explained_variance_ratio_)
plt.text(0.02, 0.98, f'Total Explained Variance (2 PCs): {total_variance:.4f}', 
         transform=plt.gca().transAxes, verticalalignment='top')

plt.tight_layout()

# Save the plot as an SVG
plt.savefig('pca_visualization_max_activating_tokens.svg')
print("Plot saved as 'pca_visualization_max_activating_tokens.svg'")

# Display some statistics
print("\nSelected point indices:", selected_indices)
print("\nCoordinates of selected points in PCA space with max activating tokens:")
for i, (x, y) in enumerate(selected_points_pca):
    print(f"Point {i+1}: ({x:.4f}, {y:.4f}) - Token: {max_activating_tokens[i]}")

print(f"\nTotal explained variance (2 PCs): {total_variance:.4f}")

# Calculate and print the full explained variance ratios
pca_full = PCA()
pca_full.fit(selected_points)
print("\nExplained variance ratios for all components:")
for i, ratio in enumerate(pca_full.explained_variance_ratio_, 1):
    print(f"PC{i}: {ratio:.4f}")

# Order the selected_indices and max_activating_tokens by first PCA component
sorted_data = sorted(zip(selected_points_pca[:, 0], selected_indices, max_activating_tokens))
sorted_indices, sorted_tokens = zip(*[(index, token) for _, index, token in sorted_data])

LIST_NAME = 'PCA'
LIST_FEATURES = []
for feature in selected_indices:
    LIST_FEATURES.append({"modelId": "gpt2-small", "layer": "8-res_fs49152-jb", "index": str(feature)})
url = "https://neuronpedia.org/quick-list/"
name = urllib.parse.quote(LIST_NAME)
url = url + "?name=" + name
url = url + "&features=" + urllib.parse.quote(json.dumps(LIST_FEATURES))
print(url)
# %%
from scipy.linalg import eigh
def spca(X, n_components=2):
    # Ensure X is normalized
    X = X / np.linalg.norm(X, axis=1)[:, np.newaxis]
    
    # Compute the weighted covariance matrix
    C = np.dot(X.T, X) / X.shape[0]
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = eigh(C)
    
    # Sort eigenvectors by descending eigenvalues
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Return the top n_components
    return eigenvectors[:, :n_components], eigenvalues


def custom_ransac_spca(X, min_samples, max_trials, threshold):

    
    # Randomly select a subset of data points
    inliers = np.random.choice(X.shape[0], min_samples, replace=False)
    previous_inliers = None

    for i in range(max_trials):
        subset = X[inliers]
        
        # Perform Spherical PCA on the subset
        principal_directions, _ = spca(subset, n_components=1)
        
        # Project data onto the principal directions
        projected = np.dot(X, principal_directions)
        
        # Reconstruct the data points
        reconstructed = np.dot(projected, principal_directions.T)
        
        # Normalize reconstructed points
        reconstructed = reconstructed / np.linalg.norm(reconstructed, axis=1)[:, np.newaxis]
        
        # Calculate the cosine distances
        distances = 1 - np.sum(X * reconstructed, axis=1)
        
        # Find the inlier points within threshold
        inliers = np.where(distances < threshold)[0]
        inlier_distances = np.mean(distances[inliers])
        
        print(f"Number of inliers: {len(inliers)}, distances: {inlier_distances}, threshold: {threshold}")
        threshold *= 0.9

        if len(inliers) < 15 and i > 3:
            print("Too few inliers. Breaking the loop.")
            break

        # Check if inliers are the same as in the previous iteration
        if previous_inliers is not None and np.array_equal(inliers, previous_inliers):
            print("Inliers are the same as in the previous iteration. Breaking the loop.")
            break    

        previous_inliers = inliers  
    
    return inliers, principal_directions


# Example usage
X = base_sae.W_dec.detach().cpu().numpy()
min_samples = 10
max_trials = 100
threshold = 0.7

selected_indices, best_pca = custom_ransac_spca(X, min_samples, max_trials, threshold)
inlier_datapoints = X[selected_indices]

print(f"Number of inlier datapoints: {len(inlier_datapoints)}")

# Assuming X and selected_indices are already defined
# Get the selected points
selected_points = X[selected_indices]

# Perform Spherical PCA on the selected points
spca_components, eigenvalues = spca(selected_points, n_components=2)
selected_points_spca = np.dot(selected_points, spca_components)

# Fetch max activating tokens
max_activating_tokens = []
for index in tqdm.tqdm(selected_indices):
    token = get_max_activating_token("gpt2-small", "8-res_fs49152-jb", str(index))
    max_activating_tokens.append(token)

# Create a scatter plot
plt.figure(figsize=(10, 8))
plt.scatter(selected_points_spca[:, 0], selected_points_spca[:, 1], color='red', s=100)

# Add labels to the points
for i, (x, y) in enumerate(selected_points_spca):
    plt.annotate(max_activating_tokens[i], (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8)

plt.title('Spherical PCA Visualization of Selected Points with Max Activating Tokens')
plt.xlabel(f'First Principal Arc')
plt.ylabel(f'Second Principal Arc')

# Calculate explained variance ratios
total_variance = sum(eigenvalues)
explained_variance_ratio = eigenvalues / total_variance

# Add explained variance information
plt.text(0.02, 0.98, f'Total Explained Variance (2 PCs): {sum(explained_variance_ratio[:2]):.4f}', 
         transform=plt.gca().transAxes, verticalalignment='top')

plt.tight_layout()

# Save the plot as an SVG
plt.savefig('spca_visualization_max_activating_tokens.svg')
print("Plot saved as 'spca_visualization_max_activating_tokens.svg'")

# Display some statistics
print("\nSelected point indices:", selected_indices)
print("\nCoordinates of selected points in SPCA space with max activating tokens:")
for i, (x, y) in enumerate(selected_points_spca):
    print(f"Point {i+1}: ({x:.4f}, {y:.4f}) - Token: {max_activating_tokens[i]}")

print(f"\nTotal explained variance (2 PCs): {sum(explained_variance_ratio[:2]):.4f}")

# Print the full explained variance ratios
print("\nExplained variance ratios for all components:")
for i, ratio in enumerate(explained_variance_ratio, 1):
    print(f"PC{i}: {ratio:.4f}")

# Order the selected_indices and max_activating_tokens by first SPCA component
sorted_data = sorted(zip(selected_points_spca[:, 0], selected_indices, max_activating_tokens))
sorted_indices, sorted_tokens = zip(*[(index, token) for _, index, token in sorted_data])

LIST_NAME = 'SPCA'
LIST_FEATURES = []
for feature in selected_indices:
    LIST_FEATURES.append({"modelId": "gpt2-small", "layer": "8-res_fs49152-jb", "index": str(feature)})
url = "https://neuronpedia.org/quick-list/"
name = urllib.parse.quote(LIST_NAME)
url = url + "?name=" + name
url = url + "&features=" + urllib.parse.quote(json.dumps(LIST_FEATURES))
print(url)
# %%
