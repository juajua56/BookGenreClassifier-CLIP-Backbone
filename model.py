import torch
import torch.nn as nn
import clip

class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim, clip_model, device, args):
        super(Classifier, self).__init__()
        self.model = clip_model.to(torch.float32)
        self.ei = nn.Linear(input_dim, input_dim, bias=False)
        self.et = nn.Linear(input_dim, input_dim, bias=False)
        
        self.MLP = nn.Sequential(nn.Linear(input_dim, 3072),
                                 nn.GELU(),
                                 nn.Dropout(args.dropout),
                                 nn.Linear(3072, output_dim),
                                 nn.Dropout(args.dropout)
                                 )
        self.device = device

    def forward(self, text, image):
        text_token = clip.tokenize(text, truncate=True).to(self.device)
        text_feature = self.model.encode_text(text_token).to(torch.float32)
        image_feature = self.model.encode_image(image).to(torch.float32)

        ET = self.ei(text_feature)
        EI = self.et(image_feature)
        k = torch.exp(ET) / (torch.exp(EI) + torch.exp(ET))
        c = k * text_feature + (1 - k) * image_feature

        x = self.MLP(c)

        return x

def load_model(device, args):
    model, preprocess = clip.load("ViT-L/14@336px", device=device)
    input_dim = args.input_dim
    output_dim = args.output_dim
    classifier = Classifier(input_dim, output_dim, model, device, args).to(device)
    return classifier, preprocess
