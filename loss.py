import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        vgg16 = torchvision.models.vgg16(pretrained=True)
        features = vgg16.features.eval()
        self.blocks = nn.ModuleList(
            [
                features[:4],
                features[4:9],
                features[9:16],
                features[16:23],
            ]
        )
        for bl in self.blocks:
            for param in bl.parameters():
                param.requires_grad = False

        self.resize = resize

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        device = input.device

        self.mean = self.mean.to(device)
        self.std = self.std.to(device)

        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)

        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std

        if self.resize:
            input = F.interpolate(
                input, mode="bilinear", size=(224, 224), align_corners=False
            )
            target = F.interpolate(
                target, mode="bilinear", size=(224, 224), align_corners=False
            )

        loss = 0.0
        for i, block in enumerate(self.blocks):
            input = block(input)
            target = block(target)
            if i in feature_layers:
                loss += F.l1_loss(input, target)
            if i in style_layers:
                act_input = input.view(input.shape[0], input.shape[1], -1)
                act_target = target.view(target.shape[0], target.shape[1], -1)
                gram_input = act_input @ act_input.permute(0, 2, 1)
                gram_target = act_target @ act_target.permute(0, 2, 1)
                loss += F.l1_loss(gram_input, gram_target)

        return loss


class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        self.register_buffer(
            "sobel_x",
            torch.tensor(
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
            ).view(1, 1, 3, 3),
        )
        self.register_buffer(
            "sobel_y",
            torch.tensor(
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
            ).view(1, 1, 3, 3),
        )

    def forward(self, output, target):
        edge_loss = 0.0
        for c in range(3):
            output_channel = output[:, c : c + 1, :, :]
            target_channel = target[:, c : c + 1, :, :]
            edge_x_output = F.conv2d(output_channel, self.sobel_x, padding=1)
            edge_y_output = F.conv2d(output_channel, self.sobel_y, padding=1)
            edge_x_target = F.conv2d(target_channel, self.sobel_x, padding=1)
            edge_y_target = F.conv2d(target_channel, self.sobel_y, padding=1)
            edge_loss += torch.mean(
                torch.abs(
                    torch.sqrt(edge_x_output**2 + edge_y_output**2 + 1e-6)
                    - torch.sqrt(edge_x_target**2 + edge_y_target**2 + 1e-6)
                )
            )

        return edge_loss / 3.0


class CombinedLoss(nn.Module):
    def __init__(self, weight_l1=10, weight_edge=2, weight_perceptual=3):
        super(CombinedLoss, self).__init__()
        self.weight_l1 = weight_l1
        self.weight_edge = weight_edge
        self.weight_perceptual = weight_perceptual
        self.l1_loss = nn.L1Loss()
        self.edge_loss = EdgeLoss()
        self.perceptual_loss = VGGPerceptualLoss()

    def forward(self, output, target):
        loss_l1 = self.l1_loss(output, target)
        loss_edge = self.edge_loss(output, target)
        loss_perceptual = self.perceptual_loss(output, target)
        total_loss = (
            self.weight_l1 * loss_l1
            + self.weight_edge * loss_edge
            + self.weight_perceptual * loss_perceptual
        )
        return total_loss
