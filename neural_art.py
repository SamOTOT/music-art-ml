import torch
from torchvision.models import vgg19, VGG19_Weights
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on:", device)

vgg = vgg19(weights=VGG19_Weights.DEFAULT).features.eval().to(device)
for p in vgg.parameters():
    p.requires_grad_(False)          # freeze: the network never changes

SIZE = 400                                   # longest side in pixels: small = fast on a CPU
MEAN = [0.485, 0.456, 0.406]                 # ImageNet colour averages VGG19 was trained with
STD = [0.229, 0.224, 0.225]

to_tensor = transforms.Compose([
    transforms.ToTensor(),                   # picture → numbers between 0 and 1
    transforms.Normalize(MEAN, STD),         # adjust colours the way VGG19 expects
])

def load_image(path, size=None, shape=None):
    img = Image.open(path).convert("RGB")
    if shape is not None:
        img = img.resize(shape)              # force an exact (width, height)
    elif size is not None:
        scale = size / max(img.size)         # shrink so the longest side = size
        img = img.resize((round(img.width * scale), round(img.height * scale)))
    return to_tensor(img).unsqueeze(0).to(device)   # add the "batch of 1" dimension

def to_picture(tensor):
    img = tensor.detach().cpu().squeeze(0)               # remove the batch dimension
    img = img * torch.tensor(STD).view(3, 1, 1) + torch.tensor(MEAN).view(3, 1, 1)   # undo Normalize
    return img.clamp(0, 1).permute(1, 2, 0).numpy()      # back to height × width × 3 for Matplotlib

content = load_image("photo.jpg", size=SIZE)
style = load_image("painting.png", shape=(content.shape[3], content.shape[2]))
print("content:", content.shape)
print("style:  ", style.shape)

# VGG19's ReLUs change values in place to save memory, which would overwrite the outputs we store.
for i, layer in enumerate(vgg):
    if isinstance(layer, torch.nn.ReLU):
        vgg[i] = torch.nn.ReLU(inplace=False)

STYLE_LAYERS = {0: "conv1_1", 5: "conv2_1", 10: "conv3_1", 19: "conv4_1", 28: "conv5_1"}
CONTENT_LAYER = 21                                    # conv4_2

def get_features(image):
    features = {}
    x = image
    for i, layer in enumerate(vgg):
        x = layer(x)                                  # pass through one layer at a time
        if i in STYLE_LAYERS or i == CONTENT_LAYER:
            features[i] = x                           # keep the layers we need
    return features

def gram(feature_map):
    _, c, h, w = feature_map.shape                    # c filters, each an h × w grid
    f = feature_map.view(c, h * w)                    # each filter's grid flattened into one row
    return f @ f.t() / (c * h * w)                    # every pair of rows: how much they fire together

content_features = get_features(content)
style_features = get_features(style)
style_grams = {i: gram(style_features[i]) for i in STYLE_LAYERS}

for i, name in STYLE_LAYERS.items():
    print(name, "features", tuple(style_features[i].shape), "-> Gram", tuple(style_grams[i].shape))
print("content conv4_2 features", tuple(content_features[CONTENT_LAYER].shape))

generated = content.clone().requires_grad_(True)     # start as a copy of the photo, and make it trainable
optimizer = torch.optim.Adam([generated], lr=0.03)   # Adam will nudge the image's numbers

STYLE_WEIGHT = 1e6       # higher = more painting, lower = more photo
STEPS = 300

target_content = content_features[CONTENT_LAYER].detach()
target_grams = {i: g.detach() for i, g in style_grams.items()}

start = time.time()
for step in range(1, STEPS + 1):
    optimizer.zero_grad()                            # clear last round's gradients
    features = get_features(generated)

    content_loss = torch.mean((features[CONTENT_LAYER] - target_content) ** 2)
    style_loss = 0
    for i in STYLE_LAYERS:
        style_loss = style_loss + torch.mean((gram(features[i]) - target_grams[i]) ** 2)

    loss = content_loss + STYLE_WEIGHT * style_loss
    loss.backward()                                  # work out which way to nudge every number
    optimizer.step()                                 # nudge them

    if step % 50 == 0:
        print(f"step {step}: content {content_loss.item():.3f}, "
              f"style {STYLE_WEIGHT * style_loss.item():.3f}, {time.time() - start:.0f}s")

plt.imsave("neural_art.png", to_picture(generated))
plt.imshow(to_picture(generated))
plt.axis("off")
plt.show()