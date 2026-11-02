"""K-fold CV training + OOF export for the paper's extra runs, reusing Approach 11's pipeline unchanged.

    python train_cv.py --name base_eegnet_cv5   --model eegnet   --set segments.use_events=false
    python train_cv.py --name base_resnet1d_cv5 --model resnet1d --set segments.use_events=false
    python train_cv.py --name sanf_net_cv5_r1   --model sanf     --cv-seed 1      # repeated CV

Same data store, folds (StratifiedKFold with --cv-seed), training loop, loss, schedule, EMA and
augmentation as SANF-Net; only the network differs. Runs go to Approach 11's artifacts/runs/<name>
(new folders only), and predictions to analysis/oof/<name>.csv via export_oof.py. Resumable.

Baselines (EEG only; recording prediction = mean of per-segment logits over the bag):
  eegnet    EEGNet-8,2 (Lawhern et al. 2018), temporal kernel 50 samples (0.5 s at 100 Hz)
  resnet1d  1D ResNet on raw 19-channel EEG with age concatenated before the classifier, in the
            spirit of CEEDNet (Kim et al. 2023); not the authors' exact code or training recipe
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
A11 = HERE.parents[1] / "Approach11_StateAwareNormative"
sys.path.insert(0, str(A11))

import torch                                                                # noqa: E402
import torch.nn as nn                                                       # noqa: E402
import torch.nn.functional as F                                             # noqa: E402
import yaml                                                                 # noqa: E402

from src.common.config import load_config                                   # noqa: E402
from src.common.logging_utils import log                                    # noqa: E402
from src.training import engine                                             # noqa: E402
from src.training.pipeline import labels, load_bundle, make_dataset         # noqa: E402

import export_oof                                                           # noqa: E402


class BagWrapper(nn.Module):
    """Segment-level network -> recording prediction (mean of segment logits over the valid bag)."""

    def __init__(self, seg_net, use_age=False):
        super().__init__()
        self.seg_net, self.use_age = seg_net, use_age
        self.modalities = ["eeg", "age"] if use_age else ["eeg"]

    def forward(self, batch):
        x = batch["x"]
        B, K, C, T = x.shape
        age = batch["age"].repeat_interleave(K, 0) if self.use_age else None
        logits = self.seg_net(x.reshape(B * K, C, T), age).view(B, K, -1).float()
        m = batch["mask"].float().unsqueeze(-1)
        logits = (logits * m).sum(1) / m.sum(1).clamp_min(1)
        gates = torch.ones(B, len(self.modalities), device=x.device) / len(self.modalities)
        return {"logits": logits, "probs": F.softmax(logits, 1), "gates": gates}


class EEGNet(nn.Module):
    """EEGNet-8,2 (Lawhern et al. 2018) for 19 x 1000 inputs at 100 Hz."""

    def __init__(self, C=19, n_classes=3, F1=8, D=2, F2=16, k=50, p=0.25):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(1, F1, (1, k), padding=(0, k // 2), bias=False), nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (C, 1), groups=F1, bias=False), nn.BatchNorm2d(F1 * D), nn.ELU(),
            nn.AvgPool2d((1, 4)), nn.Dropout(p),
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, 1, bias=False), nn.BatchNorm2d(F2), nn.ELU(),
            nn.AvgPool2d((1, 8)), nn.Dropout(p), nn.Flatten())
        with torch.no_grad():
            n = self.f(torch.zeros(1, 1, C, 1000)).shape[1]
        self.fc = nn.Linear(n, n_classes)

    def forward(self, x, age=None):
        return self.fc(self.f(x.unsqueeze(1)))


class Res1D(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv1d(cin, cout, 9, stride, 4, bias=False)
        self.b1 = nn.BatchNorm1d(cout)
        self.c2 = nn.Conv1d(cout, cout, 9, 1, 4, bias=False)
        self.b2 = nn.BatchNorm1d(cout)
        self.skip = (nn.Sequential(nn.Conv1d(cin, cout, 1, stride, bias=False), nn.BatchNorm1d(cout))
                     if stride != 1 or cin != cout else nn.Identity())

    def forward(self, x):
        return F.relu(self.b2(self.c2(F.relu(self.b1(self.c1(x))))) + self.skip(x))


class ResNet1D(nn.Module):
    """1D ResNet (4 stages x 2 blocks) on raw EEG; age joins before the classifier (CEEDNet-style)."""

    def __init__(self, C=19, n_classes=3, w=(64, 96, 128, 192), p=0.3):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(C, w[0], 15, 2, 7, bias=False), nn.BatchNorm1d(w[0]), nn.ReLU(),
                                  nn.MaxPool1d(3, 2, 1))
        layers, cin = [], w[0]
        for i, c in enumerate(w):
            layers += [Res1D(cin, c, 1 if i == 0 else 2), Res1D(c, c, 1)]
            cin = c
        self.body = nn.Sequential(*layers)
        self.age = nn.Sequential(nn.Linear(1, 16), nn.ReLU())
        self.fc = nn.Sequential(nn.Dropout(p), nn.Linear(cin + 16, n_classes))

    def forward(self, x, age):
        h = self.body(self.stem(x)).mean(-1)
        return self.fc(torch.cat([h, self.age(age)], 1))


def builder(kind):
    if kind == "sanf":
        return engine.build_model
    if kind == "eegnet":
        return lambda bundle: BagWrapper(EEGNet())
    if kind == "resnet1d":
        return lambda bundle: BagWrapper(ResNet1D(), use_age=True)
    raise SystemExit(f"unknown model {kind}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", default="sanf", choices=["sanf", "eegnet", "resnet1d"])
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--cv-seed", type=int, default=0)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(None, args.set)
    cfg["cv"] = {"n_folds": args.cv, "seed": args.cv_seed}
    cfg["paper_model"] = args.model
    run_dir = Path(cfg["paths"]["runs"]) / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    snap = {k: v for k, v in cfg.items() if not k.startswith("_")}
    (run_dir / "config_used.yaml").write_text(yaml.safe_dump(snap, sort_keys=False), encoding="utf-8")

    engine.build_model = builder(args.model)            # fit_seed calls engine.build_model
    export_oof.build_model = engine.build_model         # export_oof imported it by name
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"train_cv: '{args.name}' model={args.model} cv={args.cv} seed={args.cv_seed} overrides {args.set or 'none'}")
    for k in range(args.cv):
        b = load_bundle(cfg, fold=k)
        val_loader = engine._loader(make_dataset(b, "val", train=False), cfg, shuffle=False)
        engine.fit_seed(b, k, run_dir, dev, val_loader, labels(b, "val"), member=f"fold{k}")
    export_oof.export(run_dir)


if __name__ == "__main__":
    main()
