# train_multistage.py
# DOPE Multi-Stage Training (IsaacSim JSON 기반)

import os
import json
import warnings
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from torchvision import models

warnings.filterwarnings("ignore")

# 0. 사용자 설정
DATA_PATH   = r"C:\Isaac_Data\Pallet_Dataset_Total"  # IsaacSim에서 생성한 png/json 폴더
OUTPUT_PATH = r"C:\DOPE_Training\train_test_final_251207"
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)
EPOCHS      = 50
BATCH_SIZE  = 8
LEARNING_RATE = 1e-4

IMAGE_SIZE  = 512        # 입력 이미지 크기
OUTPUT_SIZE = 32         # heatmap 크기 (VGG stride 16 기준 512 / 16 = 32)

NUM_KEYPOINTS = 9        # LUF, RUF, RDF, LDF, LUB, RUB, RDB, LDB, Center
NUM_AFFINITY  = 16       # 임의 설정
NUM_STAGES    = 3        # Multi-stage 수 (3~6 정도, 3부터 시작 추천)

PATIENCE      = 5       # Early Stopping patience

# 1. Util: Gaussian Heatmap 생성
def make_gaussian(size, sigma=1.0, center=None):
    x = np.arange(0, size, 1, float)
    y = np.arange(0, size, 1, float)
    x, y = np.meshgrid(x, y)

    if center is None:
        x0 = size // 2
        y0 = size // 2
    else:
        x0, y0 = center

    g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    return g

# 2. IsaacSim JSON에서 9개 키포인트 읽기
def get_keypoints_from_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    kp_9 = np.array(
        data["objects"][0]["projected_cuboid"],
        dtype=np.float32
    )  # (9, 2)

    if kp_9.shape[0] != NUM_KEYPOINTS:
        warnings.warn(f"{json_path} has {kp_9.shape[0]} keypoints (expected 9).")
    return kp_9[:NUM_KEYPOINTS]

# 3. DOPE Multi-Stage Network
class DopeMultiStageNet(nn.Module):
    # - VGG-19 backbone (conv5_3까지, stride 16)
    # - Stage 1: backbone feature -> belief, affinity
    # - Stage 2~N: [feature, prev_belief, prev_affinity] concat -> belief, affinity


    def __init__(
        self,
        num_keypoints: int,
        num_affinity: int,
        num_stages: int = 3,
        pretrained_backbone: bool = True,
    ):
        super().__init__()

        self.num_keypoints = num_keypoints
        self.num_affinity = num_affinity
        self.num_stages = num_stages

        # 1) VGG-19 backbone
        vgg = models.vgg19(pretrained=pretrained_backbone)
        # conv5_3 직전까지 사용 (stride 16, output: 512x32x32)
        self.backbone = vgg.features[:30]

        # 2) Stage 1
        self.stage1_belief = self._make_stage_block(
            in_channels=512, out_channels=num_keypoints
        )
        self.stage1_affinity = self._make_stage_block(
            in_channels=512, out_channels=num_affinity
        )

        # 3) Stage 2~N
        in_ch_later = 512 + num_keypoints + num_affinity
        self.belief_blocks = nn.ModuleList()
        self.aff_blocks = nn.ModuleList()

        for _ in range(num_stages - 1):
            self.belief_blocks.append(
                self._make_stage_block(
                    in_channels=in_ch_later,
                    out_channels=num_keypoints
                )
            )
            self.aff_blocks.append(
                self._make_stage_block(
                    in_channels=in_ch_later,
                    out_channels=num_affinity
                )
            )

    def _make_stage_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        layers = [
            nn.Conv2d(in_channels, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 512, 1), nn.ReLU(inplace=True),
            nn.Conv2d(512, out_channels, 1)
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        feat = self.backbone(x)  # (B, 512, 32, 32)

        beliefs = []
        affinities = []

        # Stage 1
        b1 = self.stage1_belief(feat)
        a1 = self.stage1_affinity(feat)
        beliefs.append(b1)
        affinities.append(a1)

        # Stage 2~N
        for i in range(self.num_stages - 1):
            x_in = torch.cat([feat, beliefs[-1], affinities[-1]], dim=1)
            b = self.belief_blocks[i](x_in)
            a = self.aff_blocks[i](x_in)
            beliefs.append(b)
            affinities.append(a)

        return beliefs, affinities

# 4. Dataset
class DopeIsaacDataset(Dataset):
    def __init__(self, data_root, transform=None):
        self.data_root = data_root
        self.transform = transform
        self.samples = []

        img_files = sorted([f for f in os.listdir(data_root)
                            if f.lower().endswith(".png")])

        for img_file in img_files:
            base = os.path.splitext(img_file)[0]
            json_file = base + ".json"
            json_path = os.path.join(data_root, json_file)
            if os.path.exists(json_path):
                self.samples.append((os.path.join(data_root, img_file), json_path))
            else:
                warnings.warn(f"Missing JSON for {img_file}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, json_path = self.samples[idx]

        # 이미지 로드 및 리사이즈
        img = Image.open(img_path).convert("RGB")
        if img.size != (IMAGE_SIZE, IMAGE_SIZE):
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE))

        if self.transform:
            img_t = self.transform(img)
        else:
            img_t = transforms.ToTensor()(img)

        # 키포인트 로드
        kp_9 = get_keypoints_from_json(json_path)  # (9,2)

        # 512 -> 32 scaling
        scale_factor = OUTPUT_SIZE / IMAGE_SIZE
        kp_scaled = kp_9 * scale_factor

        # belief map 생성
        belief_maps = []
        for i in range(NUM_KEYPOINTS):
            cx, cy = kp_scaled[i]
            if 0 <= cx < OUTPUT_SIZE and 0 <= cy < OUTPUT_SIZE:
                heatmap = make_gaussian(
                    OUTPUT_SIZE,
                    sigma=1.0,
                    center=(cx, cy)
                )
            else:
                heatmap = np.zeros((OUTPUT_SIZE, OUTPUT_SIZE), dtype=np.float32)

            belief_maps.append(heatmap)

        beliefs_t = torch.tensor(
            np.stack(belief_maps, axis=0),
            dtype=torch.float32
        )  # (9,32,32)

        return {"img": img_t, "beliefs": beliefs_t}

# 5. Training
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # output dir
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # transform (VGG용 정규화)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # dataset & split
    full_dataset = DopeIsaacDataset(DATA_PATH, transform=transform)
    dataset_len = len(full_dataset)
    train_size = int(0.7 * dataset_len)
    val_size = dataset_len - train_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Windows에서는 num_workers=0 권장 (spawn 이슈 방지)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # model
    model = DopeMultiStageNet(
        num_keypoints=NUM_KEYPOINTS,
        num_affinity=NUM_AFFINITY,
        num_stages=NUM_STAGES,
        pretrained_backbone=True
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss().to(device)

    best_val_loss = float("inf")
    counter = 0

    print(f"Total dataset: {dataset_len} (Train: {train_size}, Val: {val_size})")
    print("Start Training...")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for batch_idx, data in enumerate(train_loader):
            imgs = data["img"].to(device)
            targets = data["beliefs"].to(device)  # (B,9,32,32)

            optimizer.zero_grad()

            pred_beliefs_list, _ = model(imgs)

            loss = 0.0
            for pb in pred_beliefs_list:
                loss = loss + criterion(pb, targets)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(
                    f"  [Epoch {epoch}] Batch {batch_idx}/{len(train_loader)} "
                    f"| Loss: {loss.item():.6f}"
                )

        avg_train_loss = total_loss / max(len(train_loader), 1)
        print(f"Epoch {epoch} Train Loss: {avg_train_loss:.6f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in val_loader:
                imgs = data["img"].to(device)
                targets = data["beliefs"].to(device)

                pred_beliefs_list, _ = model(imgs)

                loss_val = 0.0
                for pb in pred_beliefs_list:
                    loss_val += criterion(pb, targets)

                val_loss += loss_val.item()

        avg_val_loss = val_loss / max(len(val_loader), 1)
        print(f"[Epoch {epoch}] Validation Loss: {avg_val_loss:.6f}")

        # Early Stopping & Save
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            torch.save(
                model.state_dict(),
                os.path.join(OUTPUT_PATH, "best_model_multistage.pth")
            )
            print("Best Model Updated!")
        else:
            counter += 1
            print(f"   Validation not improved ({counter}/{PATIENCE})")
            if counter >= PATIENCE:
                print("Early Stopping triggered.")
                break

        # 주기 저장
        if epoch % 10 == 0:
            torch.save(
                model.state_dict(),
                os.path.join(OUTPUT_PATH, f"model_epoch_{epoch}.pth")
            )
            print(f"Epoch {epoch} model saved.")

    print("Training Complete.")

# Entry
if __name__ == "__main__":
    main()
