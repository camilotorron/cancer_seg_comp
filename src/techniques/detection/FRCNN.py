import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn.modules.container import Sequential


class FRCNN:
    backbone: Sequential = None
    anchor_sizes: tuple = None
    device = "cpu"
    model: FasterRCNN = None
    anchor_generator: AnchorGenerator = None

    def __init__(self, n_classes: int = 1):
        self.backbone = torchvision.models.mobilenet_v2(
            weights="MobileNet_V2_Weights.IMAGENET1K_V1"
        ).features
        self.backbone.out_channels = 1280
        self.anchor_sizes = ((32, 64, 128, 256, 512),)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.anchor_generator = AnchorGenerator(sizes=self.anchor_sizes)

        self.model = FasterRCNN(
            self.backbone,
            num_classes=n_classes,
            rpn_anchor_generator=self.anchor_generator,
        )
        self.model.to(self.device)

    def prepare_data(self, root, annFile, transform=None):
        # Load a dataset in COCO format
        if transform is not None:
            dataset = CocoDetection(root, annFile, transform=transform)

        dataset = CocoDetection(root, annFile)

        # Create a DataLoader
        data_loader = DataLoader(dataset, batch_size=2, shuffle=True)

        return data_loader

    def fine_tune(self, model, data_loader, num_epochs=3):
        # Define optimizer and scheduler
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=3, gamma=0.1
        )

        for epoch in range(num_epochs):
            # Training loop
            model.train()
            for images, targets in data_loader:
                images = list(image.to(self.device) for image in images)
                targets = [
                    {k: v.to(self.device) for k, v in t.items()} for t in targets
                ]

                # Forward pass
                loss_dict = model(images, targets)

                # Compute total loss
                losses = sum(loss for loss in loss_dict.values())

                # Backward pass
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

            # Update learning rate
            lr_scheduler.step()

            print(f"Epoch {epoch+1} completed")

    def evaluate(self, model, data_loader):
        # Set the model to evaluation mode
        model.eval()

        # Your evaluation code here
        pass
