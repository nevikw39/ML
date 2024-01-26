import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from Data_preprocess import *


"""
### Steps of PCA:

1. **Centeralize the Data:**
   - Centeralize the dataset by subtracting the mean
   $$X_{\text{cent}} = X - \bar{X}$$

2. **Calculate the Covariance Matrix:**
   - Calculate the covariance matrix, which represents the relationships between different features.
   $$ \text{Cov}(X, Y) = \sum_{i=1}^{n}(x_i - \bar{X})(y_i - \bar{Y})\ $$

3. **Compute Eigenvectors and Eigenvalues:**
   - Calculate the eigenvectors and eigenvalues of the covariance matrix.
   $$ \text{Covariance Matrix} \times \text{Eigenvector} = \text{Eigenvalue} \times \text{Eigenvector} $$

4. **Sort Eigenvectors by Eigenvalues:**
   - Sort the eigenvectors in descending order based on their corresponding eigenvalues.

5. **Select Principal Components:**
   - Choose the top $k$ eigenvectors (principal components) corresponding to the $k$ largest eigenvalues to form the projection matrix $W$.
   $$ W = \begin{bmatrix} \text{eigenvector}_1 & \text{eigenvector}_2 & \ldots & \text{eigenvector}_k \end{bmatrix} $$

6. **Transform the Data:**
   - Project the original data onto the new feature subspace using the projection matrix $W$.
   $$ \text{Transformed Data} = \text{Original Data} \times W $$

### Formula Notes:
- $n$ is the number of data points.
- $X$ and $Y$ are two variables.
- $\bar{X}$ and $\bar{Y}$ are the means of $X$ and $Y$, respectively.
- $\text{Eigenvector}$ and $\text{Eigenvalue}$ are obtained from the covariance matrix.
"""


class MY_PCA:
    """
    Please the PCA function here

    fit_transform -- Fit the model with input data and apply the dimensionality reduction on it.
    transform -- Apply dimensionality reduction to input data
    components_remain_ratio -- Calculate the minimum number of components needed to retain a specific proportion of the original data's information.
    reconstructData -- Reconstruct the data from top k eigenvectors

    """

    def __init__(self, n_components=2):
        self.n_components = n_components
        self.data = None
        self.eigen_vec = None
        self.eigen_val = None
        self.principal_components = None
        self.covariance_matrix = None

    @staticmethod
    def PCA_visualization(data_pca, label, text=False, n_components=2, tag=None):
        plt.figure(figsize=(12, 4))

        if len(label.shape) > 1:
            label = np.argmax(label, axis=1)
        #         Plot the data in the reduced dimensional space
        if text:
            x_min, x_max = np.min(data_pca, 0), np.max(data_pca, 0)
            data_pca = (data_pca - x_min) / (x_max - x_min)
            plt.figure(figsize=(10, 6))
            for i in range(data_pca.shape[0]):
                if n_components == 2:
                    plt.text(
                        data_pca[i, 0],
                        data_pca[i, 1],
                        str(label[i]),
                        color=plt.cm.Set1(label[i]),
                        fontdict={"size": 15},
                    )
                elif n_components == 1:
                    plt.text(
                        data_pca[i],
                        np.zeros(data_pca.shape[0]),
                        str(label[i]),
                        color=plt.cm.Set1(label[i]),
                        fontdict={"size": 15},
                    )

            plt.xticks([]), plt.yticks([]), plt.ylim([-0.1, 1.1]), plt.xlim([-0.1, 1.1])

        else:
            if n_components == 2:
                plt.scatter(
                    data_pca[:, 0], data_pca[:, 1], c=label, cmap="viridis", alpha=0.7
                )
            elif n_components == 1:
                plt.scatter(
                    data_pca,
                    np.zeros(data_pca.shape[0]),
                    c=label,
                    cmap="viridis",
                    alpha=0.7,
                )

            plt.colorbar(label="Label")
            plt.tight_layout()

        plt.title(f"{tag} Data in Reduced Dimensional Space (PCA) with Colored Labels")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.savefig(f"PCA_{tag}.png")
        plt.show()
        plt.close()

        return

    def _COVARIANCE_MATRIX_COMPUTATION(self, centerized_data):
        # GRADED CODE: PCA COVARIANCE MATRIX COMPUTATION
        ### !!! In this part, you can only use the common matrix operation in numpy. !!! ###
        ### !!! Please Do Not use another functions or libraries in this part. !!! ###

        ### START CODE HERE ###
        self.covariance_matrix = (
            centerized_data.T @ centerized_data / centerized_data.shape[0]
        )
        ### END CODE HERE ###
        return self.covariance_matrix

    def _COMPUTE_THE_EIGENVECTORS_AND_EIGENVALUES(self, cov_mat):
        # GRADED CODE: PCA EIGENVECTORS AND EIGENVALUES COMPUTATION
        ### !!! In this part, you can use any numpy functions. !!! ###
        ### !!! If you are using the numpy library, it is recommended to use np.linalg.eigh(). !!!###

        ### START CODE HERE ###
        eigen_val, eigen_vec = np.linalg.eig(cov_mat)
        sorted_indices = np.argsort(eigen_val)[::-1]
        self.eigen_val = eigen_val[sorted_indices].astype(float)
        self.eigen_vec = eigen_vec[:, sorted_indices].astype(float)
        ### END CODE HERE ###

        return

    def components_remain_ratio(self, ratio):
        # GRADED CODE: RECONSTRUCT DATA
        ### START CODE HERE ###
        total_var = self.eigen_val.sum()
        var_ratio = self.eigen_val / total_var
        cumulative_var = np.cumsum(var_ratio)
        num_components = np.searchsorted(cumulative_var, ratio, "right") + 1
        return num_components, var_ratio
        ### END CODE HERE ###

    def reconstructData(self, data_x, mean_data, k):
        """
        reconstruct_data -- the data reconstructed by top k eigenvector
        z -- the list of coefficients for top k eigenvector
        """
        # GRADED CODE: RECONSTRUCT DATA
        ### START CODE HERE ###
        z = (data_x - mean_data) @ self.eigen_vec[:, :k]
        reconstruct_data = z @ self.eigen_vec[:, :k].T + mean_data
        ### END CODE HERE ###

        return reconstruct_data, z

    def transform(self, data_X):
        # GRADED CODE: PCA TRANSFORM FUNCTION
        ### START CODE HERE ###
        data_pca = data_X @ self.principal_components
        ### END CODE HERE ###

        return data_pca

    def fit_transform(self, data_X):
        # GRADED CODE: PCA FITTING FUNCTION
        ### START CODE HERE ###
        self.data = data_X
        covariance_matrix = self._COVARIANCE_MATRIX_COMPUTATION(data_X)
        self._COMPUTE_THE_EIGENVECTORS_AND_EIGENVALUES(covariance_matrix)
        self.principal_components = self.eigen_vec[:, : self.n_components]
        data_pca = self.transform(data_X)
        ### END CODE HERE ###

        return data_pca


"""
### Steps of Sparse PCA:

1. **Centeralize the Data:**
   - Centeralize the dataset by subtracting the mean
   $$X_{\text{cent}} = X - \bar{X}$$

2. **Initialize Components:**
   - - Initialize the loading matrix $V$ using the transposed loading matrix from SVD.
   $$ V \text{ (Initialize)} $$

3. **Iterative Thresholding:**
   - Perform iterative thresholding to enforce sparsity on the loading matrix $V$.
      $$ V_{\text{new}} = \text{soft\_threshold}(V_{\text{old}}, \alpha) $$
      where $$ \text{soft\_threshold}(x, \alpha) = \text{sign}(x) \cdot \max(|x| - \alpha, 0) $$

4. **Normalize Components:**
   - Normalize the loading matrix $V$ to maintain unit length.
     $$ V = \frac{V}{\|V\|_2} $$

5. **Repeat Iterations:**
   - Repeat steps 3-4 for a specified number of iterations or until convergence based on the change in sum of squared differences (SSD).

6. **Transform the Data:**
   - Project the original data onto the sparse principal components $V$.
      $$ \text{Transformed Data} = \text{Original Data} \cdot V $$

### Formula Notes:
- $X$ is the original data matrix.
- $X_{\text{cent}}$ is the centerized data matrix.
- $V$ is the matrix of sparse principal components.
- $\alpha$ is the sparsity-inducing parameter.
- $\text{soft\_threshold}(x, \alpha)$ is the soft thresholding function.
- $\text{sign}(x)$ returns the sign of $x$.
- $\|\cdot\|_2$ denotes the L2 norm.
"""


class MY_SparsePCA:
    def __init__(self, n_components, alpha, max_iter=1000, tol=1e-6):
        self.alpha = alpha
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.components_ = None
        self.error_ = None
        self.S = None
        self.Vt = None

    @staticmethod
    def soft_threshold(x, alpha):
        # GRADED CODE: Iterative Thresholding
        ### START CODE HERE ###
        # Soft threshold function used for SPCA.
        rt = np.sign(x) * np.maximum(np.abs(x) - alpha, 0)
        ### END CODE HERE ###

        return rt

    def fit_transform(self, x):
        # GRADED CODE: SPARSE PCA FITTINT FUNCTION
        ### START CODE HERE ###
        # Initialize Components:
        _, S, Vt = np.linalg.svd(x, full_matrices=False)
        self.components_ = Vt[: self.n_components].T
        self.S = S
        self.Vt = Vt

        for iteration in tqdm(range(self.max_iter)):
            # Compute the transform in this iteration
            x_projected = self.transform(x)

            # Iterative Thresholding:
            self.components_ = self.soft_threshold(self.components_, self.alpha)

            # Normalize Components:
            norm = np.linalg.norm(self.components_, axis=0)
            norm[norm < 1e-10] = 1.0
            self.components_ /= norm

            # SSD Check
            ssd = ((x - x_projected @ self.components_.T) ** 2).sum()
            if ssd < self.tol:
                break
        # Transform the data
        x_transformed = self.transform(x)

        ### END CODE HERE###

        return x_transformed

    def fit(self, x):
        self.fit_transform(x)
        return self

    def transform(self, x):
        return x @ self.components_
