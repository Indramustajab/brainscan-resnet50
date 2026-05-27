import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import torch.nn as nn

device = torch.device("cpu")

model = models.resnet50()
model.fc = nn.Linear(model.fc.in_features, 2)

model.load_state_dict(torch.load("stroke_model.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

classes = ["Normal","Stroke"]

def predict_image(img_path):

    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():

        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)

        confidence, pred = torch.max(probs,1)

    label = classes[pred.item()]
    conf = confidence.item()*100

    prob_values = probs.numpy()[0]
    diff = abs(prob_values[0] - prob_values[1])

    if conf < 90 or diff < 0.2:
        label = "Gambar bukan CT Scan Otak"

    return label, conf