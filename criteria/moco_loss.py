import torch
from torch import nn
import torch.nn.functional as F
from configs.paths_config import model_paths
from torchvision import transforms

class MocoLoss(nn.Module):

    def __init__(self):
        super(MocoLoss, self).__init__()
        print("Loading MOCO model from path: {}".format(model_paths["moco"]))
        self.model = self.__load_model()
        self.model.cuda()
        self.model.eval()
        self.resize = transforms.Resize([256, 256])
    @staticmethod
    def __load_model():
        import torchvision.models as models
        model = models.__dict__["resnet50"]()
        # freeze all layers but the last fc
        for name, param in model.named_parameters():
            if name not in ['fc.weight', 'fc.bias']:
                param.requires_grad = False
        checkpoint = torch.load(model_paths['moco'], map_location="cpu")
        state_dict = checkpoint['state_dict']
        # rename moco pre-trained keys
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                # remove prefix
                state_dict[k[len("module.encoder_q."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
        msg = model.load_state_dict(state_dict, strict=False)
        assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
        # remove output layer
        model = nn.Sequential(*list(model.children())[:-1]).cuda()
        return model

    def extract_feats(self, x):
        x = F.interpolate(x, size=224)
        x_feats = self.model(x)
        x_feats = nn.functional.normalize(x_feats, dim=1)
        x_feats = x_feats.squeeze()
        return x_feats

    def forward(self, y_hat, y):
        if y_hat.size(3) != 256:
            y_hat = self.resize(y_hat)
        if y.size(3) != 256:
            y = self.resize(y)
        n_samples = y.shape[0]
        y_feats = self.extract_feats(y)  # Otherwise use the feature from there
        y_hat_feats = self.extract_feats(y_hat)
        y_feats = y_feats.detach()
        loss = 0
        id_logs = []
        count = 0
        for i in range(n_samples):
            diff_target = y_hat_feats[i].dot(y_feats[i])
            id_logs.append({'diff_target': float(diff_target),
                            'diff_input': float(diff_target),
                            'diff_views': float(diff_target)})
            loss += 1 - diff_target
            count += 1
        return loss / count, id_logs
