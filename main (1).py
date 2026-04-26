import os
import warnings
from pathlib import Path

# ✅ FIX 1: correct project root (your script is inside u-net)
PROJECT_ROOT = Path(__file__).resolve().parent

# ✅ FIX 2: correct dataset path (THIS is your real data location)
os.environ["CELEBA_DATA_DIR"] = str(PROJECT_ROOT / "data" / "processed")

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageFile, UnidentifiedImageError
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
ImageFile.LOAD_TRUNCATED_IMAGES = True


def _directory_has_images(path: Path) -> bool:
    return path.is_dir() and any(
        file_path.is_file() and file_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        for file_path in path.iterdir()
    )


def resolve_data_dir(preferred_dir: str | None = None) -> Path:
    candidate_dirs = []

    if preferred_dir:
        candidate_dirs.append(Path(preferred_dir))

    # ✅ FIX 3: correct search paths (ALL relative to your real project root)
    candidate_dirs.extend([
        PROJECT_ROOT / "data" / "processed",
        PROJECT_ROOT / "data" / "processed" / "train",
        PROJECT_ROOT / "archive" / "img_align_celeba" / "img_align_celeba",
    ])

    for candidate_dir in candidate_dirs:
        if _directory_has_images(candidate_dir):
            return candidate_dir

    searched_dirs = "\n".join(f"- {candidate_dir}" for candidate_dir in candidate_dirs)
    raise FileNotFoundError(
        "No image dataset was found. Checked these directories:\n"
        f"{searched_dirs}\n\n"
        "Set the CELEBA_DATA_DIR environment variable if your images live elsewhere."
    )


# ─────────────────────────────────────────────
#  MASK GENERATOR (UNCHANGED)
# ─────────────────────────────────────────────
class MaskGenerator:
    def __init__(self, mask_ratio=0.25):
        self.mask_ratio = mask_ratio

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        _, H, W = img.shape
        mask = torch.ones(1, H, W)

        hole_h = max(1, int(H * self.mask_ratio))
        hole_w = max(1, int(W * self.mask_ratio))
        top  = torch.randint(0, max(1, H - hole_h + 1), (1,)).item()
        left = torch.randint(0, max(1, W - hole_w + 1), (1,)).item()

        mask[:, top:top + hole_h, left:left + hole_w] = 0
        return mask


# ─────────────────────────────────────────────
#  DATASET (ONLY PATH FIXED, LOGIC SAME)
# ─────────────────────────────────────────────
class CelebaDataset(Dataset):
    def __init__(self, data_dir: str, mask_generator: MaskGenerator, img_size: int = 128):
        self.data_dir = Path(data_dir)
        self.mask_generator = mask_generator
        self._bad_image_paths = set()

        # ✅ FIX 4: recursive scan so NOTHING is missed
        candidate_paths = sorted([
            p for p in self.data_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ])

        empty_image_paths = [p for p in candidate_paths if p.stat().st_size == 0]
        self.image_paths = [p for p in candidate_paths if p.stat().st_size > 0]

        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No images found in: {self.data_dir}")

        if empty_image_paths:
            warnings.warn(
                f"Skipped {len(empty_image_paths)} empty image file(s).",
                stacklevel=2,
            )

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def _load_image_tensor(self, image_path: Path) -> torch.Tensor:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            return self.transform(img)

    def __getitem__(self, idx):
        dataset_size = len(self.image_paths)

        for offset in range(dataset_size):
            current_idx = (idx + offset) % dataset_size
            image_path = self.image_paths[current_idx]

            if image_path in self._bad_image_paths:
                continue

            try:
                y = self._load_image_tensor(image_path)
                mask = self.mask_generator(y)
                x = y * mask
                return x, y
            except (UnidentifiedImageError, OSError, ValueError):
                self._bad_image_paths.add(image_path)

        raise RuntimeError("No readable images available.")


# ─────────────────────────────────────────────
# EVERYTHING BELOW = UNCHANGED (MODEL + TRAINING)
# ─────────────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        skip = self.double_conv(x)
        return self.pool(skip), skip


class Decoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, 2)
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, blocks=4, start_ch=8):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        self.encoders.append(Encoder(in_ch, start_ch))
        ch = start_ch

        for _ in range(blocks - 1):
            self.encoders.append(Encoder(ch, ch * 2))
            ch *= 2

        self.bottleneck = nn.Sequential(
            nn.Conv2d(ch, ch * 2, 3, padding=1),
            nn.BatchNorm2d(ch * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch * 2, ch * 2, 3, padding=1),
            nn.BatchNorm2d(ch * 2),
            nn.ReLU(inplace=True),
        )
        ch *= 2

        for _ in range(blocks):
            self.decoders.append(Decoder(ch, ch // 2))
            ch //= 2

        self.output_conv = nn.Sequential(
            nn.Conv2d(ch, out_ch, 1),
            nn.Tanh()
        )

    def forward(self, x):
        skips = []
        for enc in self.encoders:
            x, skip = enc(x)
            skips.append(skip)

        x = self.bottleneck(x)

        for dec in self.decoders:
            x = dec(x, skips.pop())

        return self.output_conv(x)


class InpaintingModel(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.unet = UNet()
        self.loss = nn.L1Loss()
        self.lr = lr

    def forward(self, x):
        return self.unet(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss(self(x), y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss(self(x), y)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


# ─────────────────────────────────────────────
# ENTRY (UNCHANGED LOGIC)
# ─────────────────────────────────────────────
if __name__ == "__main__":

    DATA_DIR = resolve_data_dir(os.environ.get("CELEBA_DATA_DIR"))

    mask_gen = MaskGenerator()
    dataset = CelebaDataset(DATA_DIR, mask_gen)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)

    model = InpaintingModel()

    trainer = pl.Trainer(max_epochs=10, accelerator="auto")
    trainer.fit(model, train_loader, val_loader)
    trainer.save_checkpoint("unet.ckpt")