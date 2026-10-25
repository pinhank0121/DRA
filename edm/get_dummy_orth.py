import torch
import torch.nn as nn


class DummyFeatureExtractor(nn.Module):
    """
    Dummy feature extractor that returns fixed orthogonal vectors based on class labels.
    """

    def __init__(
        self,
        num_classes=10,
        feature_dims=(640, 1, 1),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        cfg=False,
    ):
        super(DummyFeatureExtractor, self).__init__()

        self.num_classes = num_classes
        self.feature_dims = feature_dims
        self.device = device
        self.cfg = cfg

        # Calculate total feature dimension
        feat_size = feature_dims[0] * feature_dims[1] * feature_dims[2]

        # Create orthogonal vectors
        # Start with random vectors
        vectors = torch.randn(num_classes, feat_size, device=device)

        # Apply Gram-Schmidt orthogonalization
        for i in range(1, num_classes):
            for j in range(i):
                vectors[i] -= torch.dot(vectors[i], vectors[j]) * vectors[j] / torch.dot(vectors[j], vectors[j])

        # Normalize vectors to unit length
        for i in range(num_classes):
            vectors[i] = vectors[i] / torch.norm(vectors[i])

        # Reshape to match feature dimensions
        self.class_features = vectors.view(num_classes, *feature_dims)

        print(f"Initialized dummy feature extractor with {num_classes} orthogonal vectors of shape {feature_dims}")

    def forward(self, x, class_labels, return_embed=False):
        """
        Args:
            x: Input tensor (not used, just for compatibility)
            class_labels: Class labels tensor of shape (batch_size,) or one-hot encoded (batch_size, num_classes)

        Returns:
            Features corresponding to the class labels
        """
        batch_size = x.shape[0]

        # Convert one-hot to indices if needed
        if class_labels is not None and class_labels.dim() > 1:
            class_indices = torch.argmax(class_labels, dim=1)
        else:
            class_indices = class_labels

        # Select the corresponding feature vector for each example
        features = self.class_features[class_indices]

        if self.cfg:
            # For classifier-free guidance, return both conditional and unconditional features
            # Unconditional features are all zeros in this dummy implementation
            uncond_features = torch.zeros_like(features)
            return features, uncond_features
        if return_embed:
            return [features], None

        return [features]


def get_dummy_feature_model(
    num_classes=10,
    feature_dims=(640, 1, 1),
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    cfg=False,
):
    model = DummyFeatureExtractor(num_classes, feature_dims, device, cfg)
    return model


if __name__ == "__main__":
    model = get_dummy_feature_model(cfg=False)
    print(model)

    bs = 4
    input_tensor = torch.randn(bs, 3, 32, 32).cuda()
    class_labels = torch.randint(0, 10, (bs,)).cuda()

    features = model(input_tensor, class_labels)

    if isinstance(features, tuple):
        print(f"Conditional features shape: {features[0].shape}")
        print(f"Unconditional features shape: {features[1].shape}")
    else:
        print(f"Features shape: {features.shape}")

    # Verify orthogonality
    vectors = model.class_features.view(model.num_classes, -1)
    sim_matrix = torch.mm(vectors, vectors.t())
    print("Similarity matrix (diagonal should be 1, off-diagonal ~0):")
    print(sim_matrix)
