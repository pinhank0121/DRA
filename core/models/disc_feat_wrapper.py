import torch

class DiscFeatExtractorWrapper(torch.nn.Module):
    def __init__(self, model, path, device):
        super(DiscFeatExtractorWrapper, self).__init__()
        self.model = model
        self.path = path
        checkpoint = torch.load(path)
        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")
        # unwrap from DataParallel
        if isinstance(self.model, torch.nn.DataParallel):
            self.model = self.model.module
        self.model.eval()
        self.model.to(device)
        # freeze the model
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x, class_labels=None, return_embed=False):
        if return_embed:
            raise ValueError("return_embed is not supported for disc feature extractor wrapper.")
        with torch.no_grad():
            return [self.model.forward_features(x)], None # list wrapper to match expected output format