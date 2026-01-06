import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision

class _ResNet50Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
            base = torchvision.models.resnet50(weights=weights)
        except Exception:
            base = torchvision.models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(base.children())[:-1])

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        return x

class ReIDExtractor:
    def __init__(self, model_path: str | None = None, device: str | None = None, input_size: tuple[int, int] = (256, 128)):
        self.device = device if device in ("cpu", "cuda") else ("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.model = None
        self.feat_dim = None
        if model_path and os.path.isfile(model_path):
            try:
                m = torch.jit.load(model_path, map_location=self.device)
                m.eval()
                self.model = m
            except Exception:
                self.model = _ResNet50Backbone()
        else:
            self.model = _ResNet50Backbone()
        self.model.to(self.device)
        with torch.no_grad():
            d = torch.zeros(1, 3, self.input_size[0], self.input_size[1], device=self.device)
            o = self.model(d)
            if o.ndim == 2:
                self.feat_dim = int(o.shape[1])
            else:
                self.feat_dim = int(np.prod(o.shape[1:]))

    def _clip_bbox(self, h: int, w: int, bbox):
        x1, y1, x2, y2 = bbox
        x1 = max(0, int(np.floor(x1)))
        y1 = max(0, int(np.floor(y1)))
        x2 = min(w - 1, int(np.ceil(x2)))
        y2 = min(h - 1, int(np.ceil(y2)))
        if x2 <= x1:
            x2 = min(w - 1, x1 + 1)
        if y2 <= y1:
            y2 = min(h - 1, y1 + 1)
        return x1, y1, x2, y2

    def _preprocess(self, frame: np.ndarray, bbox):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = self._clip_bbox(h, w, bbox)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            roi = frame
        resized = cv2.resize(roi, (self.input_size[1], self.input_size[0]))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img = rgb.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        chw = np.transpose(img, (0, 1, 2)).transpose(2, 0, 1)
        t = torch.from_numpy(chw).unsqueeze(0).to(self.device)
        return t

    def extract(self, frame: np.ndarray, bbox) -> np.ndarray:
        t = self._preprocess(frame, bbox)
        with torch.no_grad():
            f = self.model(t)
            if f.ndim > 2:
                f = torch.flatten(f, 1)
            f = f.squeeze(0)
            n = torch.norm(f, p=2)
            if float(n) > 0:
                f = f / n
        return f.detach().cpu().numpy()