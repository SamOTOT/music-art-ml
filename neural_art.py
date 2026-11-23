"""Neural style transfer: repaint a photo in the style of another image (ticket 13)."""
import time
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torchvision.models import vgg19, VGG19_Weights

# ---------- settings ----------
PHOTO = "photo.jpg"
STYLE_IMAGE = "painting.png"
OUTPUT = "neural_art.png"
SIZE = 400                 # longest side in pixels (bigger = slower)
STYLE_WEIGHT = 1e5         # higher = more painting, lower = more photo
STEPS = 300
LEARNING_RATE = 0.03

STYLE_LAYERS = {0: "conv1_1", 5: "conv2_1", 10: "conv3_1", 19: "conv4_1", 28: "conv5_1"}
CONTENT_LAYER = 21         # conv4_2
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- the network: pretrained VGG19, frozen ----------
vgg = vgg19(weights=VGG19_Weights.DEFAULT).features.eval().to(device)
for p in vgg.parameters():
    p.requires_grad_(False)
for i, layer in enumerate(vgg):
    if isinstance(layer, torch.nn.ReLU):
        vgg[i] = torch.nn.ReLU(inplace=False)     # so stored layer outputs aren't overwritten

# ---------- images ----------
to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

def load_image(path, size):
    img = Image.open(path).convert("RGB")
    scale = size / max(img.size)
    img = img.resize((round(img.width * scale), round(img.height * scale)))
    return to_tensor(img).unsqueeze(0).to(device)

def load_style(path, shape):
    """Crop the style image to the photo's shape (from the middle), then resize evenly."""
    img = Image.open(path).convert("RGB")
    w, h = shape
    target = w / h
    if img.width / img.height > target:
        new_w = round(img.height * target)
        left = (img.width - new_w) // 2
        img = img.crop((left, 0, left + new_w, img.height))
    else:
        new_h = round(img.width / target)
        upper = (img.height - new_h) // 2
        img = img.crop((0, upper, img.width, upper + new_h))
    img = img.resize((w, h))
    return to_tensor(img).unsqueeze(0).to(device)

def to_picture(tensor):
    img = tensor.detach().cpu().squeeze(0)
    img = img * torch.tensor(STD).view(3, 1, 1) + torch.tensor(MEAN).view(3, 1, 1)
    return img.clamp(0, 1).permute(1, 2, 0).numpy()

# ---------- what VGG19 sees ----------
def get_features(image):
    features = {}
    x = image
    for i, layer in enumerate(vgg):
        x = layer(x)
        if i in STYLE_LAYERS or i == CONTENT_LAYER:
            features[i] = x
    return features

def gram(feature_map):
    _, c, h, w = feature_map.shape
    f = feature_map.view(c, h * w)
    return f @ f.t() / (c * h * w)

# ---------- paint ----------
if __name__ == "__main__":
    content = load_image(PHOTO, SIZE)
    style = load_style(STYLE_IMAGE, (content.shape[3], content.shape[2]))

    target_content = get_features(content)[CONTENT_LAYER].detach()
    style_features = get_features(style)
    target_grams = {i: gram(style_features[i]).detach() for i in STYLE_LAYERS}

    generated = content.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([generated], lr=LEARNING_RATE)

    print(f"Painting {PHOTO} in the style of {STYLE_IMAGE} on {device}...")
    start = time.time()
    for step in range(1, STEPS + 1):
        optimizer.zero_grad()
        features = get_features(generated)
        content_loss = torch.mean((features[CONTENT_LAYER] - target_content) ** 2)
        style_loss = 0
        for i in STYLE_LAYERS:
            style_loss = style_loss + torch.mean((gram(features[i]) - target_grams[i]) ** 2)
        loss = content_loss + STYLE_WEIGHT * style_loss
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            print(f"step {step}: content {content_loss.item():.1f}, "
                  f"style {STYLE_WEIGHT * style_loss.item():.1f}, {time.time() - start:.0f}s")

    plt.imsave(OUTPUT, to_picture(generated))
    print("saved", OUTPUT)
    plt.imshow(to_picture(generated))
    plt.axis("off")
    plt.show()