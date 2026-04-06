from torchvision import tv_tensors
from torchvision.transforms import functional as F
from torchvision.transforms.v2 import Transform


class Pad(Transform):
    def __init__(self, target_size=224, fill=0):
        super().__init__()
        self.target_size = target_size
        self.fill = fill

    def forward(self, img):
        # Handle both PIL and tv_tensors.Image
        if isinstance(img, tv_tensors.Image):
            img = F.to_pil_image(img)

        # Resize with aspect ratio preserved
        img = F.resize(img, self.target_size, antialias=True)

        # Padding to square
        pad = [0, 0, 0, 0]
        w, h = img.size
        if w == h:
            padded = img
        elif w > h:
            pad = (0, (w - h) // 2, 0, (w - h + 1) // 2)  # (left, top, right, bottom)
        else:
            pad = ((h - w) // 2, 0, (h - w + 1) // 2, 0) # (left, top, right, bottom)
            
        padded = F.pad(img, pad, fill=self.fill)

        # Final resize (just to be safe)
        padded = F.resize(padded, (self.target_size, self.target_size), antialias=True)

        return tv_tensors.Image(padded)
