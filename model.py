import torch
import torch.nn as nn
import clip

class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim, clip_model, device):
        super(Classifier, self).__init__()
        self.model = clip_model.to(torch.float32)
        self.fc1 = nn.Linear(input_dim, 3072)
        self.fc2 = nn.Linear(3072, output_dim)
        self.gelu = nn.GELU()
        self.drop = nn.Dropout(0.8)
        self.ei = nn.Linear(input_dim, input_dim, bias=False)
        self.et = nn.Linear(input_dim, input_dim, bias=False)
        self.device = device

    def forward(self, text, image):
        text_token = clip.tokenize(text, truncate=True).to(self.device)
        text_feature = self.model.encode_text(text_token).to(torch.float32)
        image_feature = self.model.encode_image(image).to(torch.float32)

        ET = self.ei(text_feature)
        EI = self.et(image_feature)
        k = torch.exp(ET) / (torch.exp(EI) + torch.exp(ET))
        c = k * text_feature + (1 - k) * image_feature

        x = self.fc1(c)
        x = self.gelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x

def load_model(device):
    model, preprocess = clip.load("ViT-L/14@336px", device=device)
    input_dim = 768
    output_dim = 24
    classifier = Classifier(input_dim, output_dim, model, device).to(device)
    return classifier, preprocess
