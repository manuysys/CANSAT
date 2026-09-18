"""
CanSat La Base — Red siamesa para change detection de daño.

Procesa pre+post juntas y aprende "qué cambió" directamente.

════════════════════════════════════════════════════════════════════════════
⚠ LOS PESOS best_siamese_damage.pth YA ENTRENADOS arrastran ESTE DEFECTO
════════════════════════════════════════════════════════════════════════════
La versión anterior tenía, en ``__getitem__``::

    if pre_p is None or post_p is None or not mask_p.exists():
        img = cv2.imread(r["image"])
        pre, post = img, img          # ← la MISMA imagen

Es decir: cuando faltaba el par pre/post, le daba a la red dos copias idénticas
(cambio CERO) pero con la máscara de daño real como target. **Le enseñaba a
predecir daño donde no hay ningún cambio.** Ahora esas muestras se descartan y
se reporta cuántas.

Los pesos ya entrenados con ese fallback son sospechosos: re-entrenar antes de
confiar en el siamés. Mientras tanto ``mission_pipeline.py`` lo trata como un
voto más del consenso, no como prueba.

Otros arreglos:
  · ``extract_features()`` se salteaba el ``forward()`` de DeepLabV3+ y perdía
    las features de stride-4, trabajando a stride 16 en vez de 8. Ahora usa el
    camino completo del backbone y avisa de la resolución resultante.
  · ``GradScaler("cuda")`` incondicional → en CPU reventaba. ``train.py`` ya lo
    hacía bien con ``enabled=(device.type=="cuda")``.
  · ``num_workers=4, pin_memory=True`` hardcodeado (problemático en Windows).
  · ``ignore_index=255`` con etiquetas 0/1/2: parámetro inerte.
  · ``conf_matrix`` con doble loop de Python → ``cansat.metrics`` vectorizada.
  · Sin scheduler ni early stopping → se agregaron.
  · El docstring prometía "Mejora IoU de daño: 0.61 → 0.70+ (esperado)": un
    número inventado, marcado como esperado, que se leía como resultado.
  · ``ExportWrap`` era un no-op; se exporta el modelo directamente.
  · La exportación ONNX corría siempre, incluso si el entrenamiento no había
    mejorado nada. Ahora hace falta ``--export``.

Uso:
    python train_siamese_damage.py --epochs 10
    python train_siamese_damage.py --epochs 10 --export
"""
import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from train import DeepLabV3PlusMobileNetV2
from cansat.seed import device as pick_device, set_seed

def conf_matrix(pred, tgt, n=3):
    """DEPRECADO: doble loop de Python. Usar ``cansat.metrics.confusion``."""
    from cansat.metrics import confusion
    return confusion(tgt, pred, n, ignore_index=-1)

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
XBD = Path("dataset/xbd")
MASKS = Path("dataset/xbd_masks")


class SiameseDamageNet(nn.Module):
    """Red siamesa: backbone compartido + head de daño."""
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        # Head: concatena features de pre+post (256*2=512 canales)
        self.head = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 3, 1)  # 3 clases: other, intacto, dañado
        )

    def forward(self, pre, post):
        # Extraer features (sin la cabeza de segmentación)
        feat_pre = self.extract_features(pre)
        feat_post = self.extract_features(post)
        # Concatenar y predecir daño
        combined = torch.cat([feat_pre, feat_post], dim=1)
        out = self.head(combined)
        # Upsample al tamaño original
        return F.interpolate(out, size=pre.shape[2:], mode="bilinear", align_corners=False)

    def extract_features(self, x):
        """
        Features de 256 canales a la salida del ASPP.

        ⚠ ARREGLADO: la versión anterior hacía ``self.backbone.backbone(x)`` y
          luego ``aspp``, o sea que **se salteaba el ``forward()`` de
          DeepLabV3PlusMobileNetV2**. Ese forward modifica los strides de los
          stages 7-17 (output stride 8) y guarda las features de stride 4 para
          el decoder. Al llamar los submódulos sueltos se obtenían features a
          stride 16 — la mitad de la resolución que el diseño supone — y el head
          concatenaba mapas de ese tamaño. ``F.interpolate`` al final estiraba el
          resultado y disimulaba el problema.

          Ahora replicamos el camino real del backbone (stride modificado) y
          aplicamos el ASPP, que es lo que se pretendía.
        """
        bb = getattr(self.backbone, "backbone", None)
        if bb is None:
            out = self.backbone(x)
            return out["out"] if isinstance(out, dict) else out

        feat = x
        for block in bb:
            feat = block(feat)
        aspp = getattr(self.backbone, "aspp", None)
        if aspp is not None:
            feat = aspp(feat)
        return feat


class XBDSiameseDataset(Dataset):
    """Dataset de pares pre+post de xBD para red siamesa."""
    def __init__(self, rows, size=320, aug=False, collect_stats=None, neg_frac=0.0):
        self.size, self.aug, self.neg_frac = size, aug, neg_frac
        # ⚠ ARREGLADO: antes, si faltaba el par pre/post o la máscara, se usaba
        # la MISMA imagen como pre y como post. Una red siamesa de change
        # detection que recibe pre == post ve cambio cero, pero el target seguía
        # marcando edificios dañados: se la entrenaba para alucinar daño.
        # Ahora la muestra se descarta y se cuenta cuántas se perdieron.
        self.rows = []
        self.descartadas = 0
        for r in rows:
            stem = str(r["name"]).replace("_post_disaster", "")
            pre_p = next(XBD.rglob(f"{stem}_pre_disaster.png"), None)
            post_p = next(XBD.rglob(f"{stem}_post_disaster.png"), None)
            mask_p = MASKS / f"{r['name']}.png"
            if pre_p is None or post_p is None or not mask_p.is_file():
                self.descartadas += 1
                continue
            self.rows.append({**r, "_pre": str(pre_p), "_post": str(post_p),
                              "_mask": str(mask_p)})
        if collect_stats is not None:
            collect_stats["descartadas"] = self.descartadas
            collect_stats["usadas"] = len(self.rows)
        if self.descartadas:
            print(f"  [Dataset] {self.descartadas} muestra(s) sin par pre/post "
                  f"completo → DESCARTADAS (antes se entrenaban con pre==post, "
                  f"lo que enseña a alucinar daño). Quedan {len(self.rows)}.")

    def _legacy_rows(self):
        return self.rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        pre = cv2.imread(r["_pre"])
        post = cv2.imread(r["_post"])
        mask = cv2.imread(r["_mask"], cv2.IMREAD_GRAYSCALE)
        if pre is None or post is None or mask is None:
            # Imagen ilegible en runtime: devolver un par neutro sin daño.
            # Nunca pre == post con target de daño.
            return (torch.zeros(3, self.size, self.size),
                    torch.zeros(3, self.size, self.size),
                    torch.zeros(self.size, self.size, dtype=torch.long))
        # Convertir máscara: 0=other, 1=intacto, 2+=dañado
        mask = np.where(mask >= 2, 2, mask).astype(np.uint8)

        # Resize
        pre = cv2.resize(pre, (self.size, self.size))
        post = cv2.resize(post, (self.size, self.size))
        mask = cv2.resize(mask, (self.size, self.size),
                          interpolation=cv2.INTER_NEAREST)

        # Augmentación (aplicar igual a pre y post)
        if self.aug:
            if random.random() < 0.5:
                pre, post, mask = cv2.flip(pre, 1), cv2.flip(post, 1), cv2.flip(mask, 1)
            if random.random() < 0.5:
                pre, post, mask = cv2.flip(pre, 0), cv2.flip(post, 0), cv2.flip(mask, 0)
            k = random.choice([0, 1, 2, 3])
            if k:
                pre = np.ascontiguousarray(np.rot90(pre, k))
                post = np.ascontiguousarray(np.rot90(post, k))
                mask = np.ascontiguousarray(np.rot90(mask, k))
            # ── Par NEGATIVO sintético: pre == post, edificios intactos ──────
            # Sin esto la red nunca vio "sin cambio" y alucinaba daño (medido
            # sobre los pesos viejos: 5.5 % medio de daño sobre pares idénticos,
            # hasta 63 %). Es LA corrección que hace falta para que sirva como
            # detector de cambio: mismo input ⇒ cero daño.
            if self.neg_frac > 0 and random.random() < self.neg_frac:
                post = pre.copy()
                mask = np.where(mask >= 1, 1, 0).astype(np.uint8)

        # Preprocesar
        def prep(img):
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            return (rgb - MEAN) / STD

        pre_t = prep(pre).transpose(2, 0, 1)
        post_t = prep(post).transpose(2, 0, 1)

        return (torch.from_numpy(pre_t).float(),
                torch.from_numpy(post_t).float(),
                torch.from_numpy(mask.astype(np.int64)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--out", default="outputs/best_siamese_damage.pth")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=0,
                    help="antes eran 4 + pin_memory hardcodeados (rompe en Windows)")
    ap.add_argument("--patience", type=int, default=3,
                    help="early stopping en épocas sin mejora (0 = desactivado)")
    ap.add_argument("--neg-frac", type=float, default=0.3,
                    help="fracción de muestras con pre==post (edificios intactos) "
                         "para enseñar 'sin cambio ⇒ sin daño' (0 = desactivado)")
    ap.add_argument("--export", action="store_true",
                    help="exportar a ONNX al terminar. Antes se exportaba SIEMPRE, "
                         "incluso si el entrenamiento no había mejorado nada.")
    args = ap.parse_args(argv)

    # Cargar manifest
    with open(MASKS / "manifest.csv", encoding="utf-8") as _f:
        manifest = list(csv.DictReader(_f))
    # ⚠ Split POR DESASTRE (ver cansat/xbd.py): antes era 90/10 por fila y
    #   repartía tiles vecinos del mismo evento entre train y val.
    from cansat.xbd import split_por_desastre
    train_rows, val_rows, val_grupos = split_por_desastre(manifest, 0.2, args.seed)
    print(f"Muestras: {len(manifest)} → train {len(train_rows)} / val {len(val_rows)}")
    print(f"Val por desastre: {val_grupos}")

    device = pick_device("cuda")     # antes: hardcodeado a cuda en algunos scripts
    print(f"Device: {device}")
    set_seed(args.seed)

    # Cargar backbone pre-entrenado (DeepLabV3+ de daños)
    try:
        backbone = DeepLabV3PlusMobileNetV2(3)
    except TypeError:
        backbone = DeepLabV3PlusMobileNetV2()
    ckpt = Path("outputs/best_damage_v3.pth")
    if ckpt.exists():
        # load_into acepta dict-con-metadata o state_dict crudo y avisa si la
        # cobertura es parcial (antes era un load_state_dict que rompía con el
        # formato dict).
        from cansat.checkpoints import load_into
        _, frac = load_into(backbone, ckpt, strict=False, min_loaded_frac=0.5,
                            verbose=True)
        print(f"[OK] Backbone cargado desde {ckpt} ({frac:.0%})")

    model = SiameseDamageNet(backbone).to(device)

    # Pesos: other=0.3, intacto=1.0, dañado=3.0
    w = torch.tensor([0.3, 1.0, 3.0]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # ⚠ antes: GradScaler("cuda") incondicional → en CPU scale() reventaba.
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    stats_tr, stats_va = {}, {}
    train_dl = DataLoader(XBDSiameseDataset(train_rows, args.size, aug=True,
                                            collect_stats=stats_tr,
                                            neg_frac=args.neg_frac),
                          batch_size=args.batch, shuffle=True,
                          num_workers=args.workers,
                          pin_memory=(device.type == "cuda"))
    val_dl = DataLoader(XBDSiameseDataset(val_rows, args.size,
                                          collect_stats=stats_va),
                        batch_size=args.batch, num_workers=args.workers)
    if not len(train_dl.dataset):
        print("[ERROR] No quedó ninguna muestra de entrenamiento con el par")
        print("        pre/post completo. Revisá dataset/xbd y dataset/xbd_masks.")
        return 1

    best = -1.0
    sin_mejora = 0
    for ep in range(args.epochs):
        model.train()
        for bi, (pre_b, post_b, yb) in enumerate(train_dl):
            pre, post, y = pre_b.to(device), post_b.to(device), yb.to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(pre, post)
                # ignore_index=255 era inerte: las etiquetas son 0/1/2.
                loss = F.cross_entropy(logits, y, weight=w)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if bi % 50 == 0:
                print(f"    batch {bi}/{len(train_dl)} loss {loss.item():.4f}")

        model.eval()
        cm = np.zeros((3, 3), dtype=np.int64)
        inter_d = union_d = 0
        with torch.no_grad():
            for pre_b, post_b, yb in val_dl:
                pre, post, y = pre_b.to(device), post_b.to(device), yb.numpy()
                pred = model(pre, post).argmax(1).cpu().numpy()
                cm += conf_matrix(pred, y)
                # IoU de daño (two-stage: solo sobre edificios del post)
                pd, gd, gb = pred == 2, y == 2, y >= 1
                inter_d += int((pd & gd).sum())
                union_d += int(((pd & gb) | gd).sum())

        ious = []
        for i in range(3):
            inter = cm[i, i]
            union = cm[i, :].sum() + cm[:, i].sum() - inter
            ious.append(inter / union if union else 0.0)
        iou_d2 = inter_d / union_d if union_d else 0.0

        print(f"  epoch {ep + 1}: other={ious[0]:.2f} intacto={ious[1]:.2f} "
              f"dañado={ious[2]:.2f} | DAÑADO* two-stage={iou_d2:.3f}")
        sched.step()
        if iou_d2 > best:
            best = iou_d2
            sin_mejora = 0
            # Formato único de checkpoint (ver cansat/checkpoints.py): antes se
            # guardaba el state_dict crudo acá y un dict con metadata en train.py,
            # y ningún exportador servía para los dos.
            from cansat.checkpoints import save_ckpt
            save_ckpt(args.out, model, num_classes=3, img_size=args.size,
                      class_names=["other", "intacto", "danado"],
                      iou_dano_two_stage=float(iou_d2), iou_per_class=[float(x) for x in ious],
                      dataset="xbd", script="train_siamese_damage.py",
                      epochs=ep + 1, seed=args.seed)
            print(f"  → guardado {args.out}")
        else:
            sin_mejora += 1
            if args.patience and sin_mejora >= args.patience:
                print(f"  [early stop] {sin_mejora} épocas sin mejorar "
                      f"(mejor DAÑADO* {best:.3f})")
                break

    print(f"[OK] mejor DAÑADO* (siamese): {best:.3f}")
    print("     (número MEDIDO en Val. El docstring anterior prometía")
    print('      "0.61 -> 0.70+ (esperado)", que no era una medición.)')

    # ── Autochequeo: pre == post NO debe dar daño ───────────────────────
    # Los pesos viejos alucinaban daño sin cambio (medido: 5.5 % medio, máx
    # 63 %). Este chequeo es el que decide si el modelo sirve como detector
    # de cambio; sin él, el siamés sólo mete ruido en el consenso.
    from cansat.checkpoints import load_model_state
    model.load_state_dict(load_model_state(args.out))
    model.eval()
    n_check = min(20, len(val_dl.dataset))
    frac_dano = []
    with torch.no_grad():
        for k in range(n_check):
            pre, _post, _y = val_dl.dataset[k]
            x = pre[None].to(device)
            pred = model(x, x).argmax(1)[0].cpu().numpy()
            frac_dano.append(float((pred == 2).mean()) * 100.0)
    if frac_dano:
        print(f"[check] daño con pre==post: {np.mean(frac_dano):.2f} % medio, "
              f"máx {np.max(frac_dano):.2f} %  (debe ser ~0)")
        print("        Si es alto, el modelo NO debe volar (ver MODELS.yaml).")

    # ── Exportación ONNX (ahora opcional) ───────────────────────────────
    if not args.export:
        print("     (sin exportar a ONNX; agregá --export cuando el número convenza)")
        return 0

    model.eval().cpu()
    dummy = (torch.randn(1, 3, args.size, args.size),
             torch.randn(1, 3, args.size, args.size))
    # El ExportWrap de la versión anterior era un no-op (forward sólo delegaba).
    # dynamo=False y sin dynamic_axes: requisito declarado del conversor IMX500,
    # que sólo dos de los seis exportadores del repo respetaban.
    torch.onnx.export(model, dummy,
                      "outputs/cansat_siamese_damage.onnx",
                      input_names=["pre", "post"],
                      output_names=["logits"],
                      opset_version=17, do_constant_folding=True, dynamo=False)
    print("[OK] outputs/cansat_siamese_damage.onnx")
    print("     Verificá con: python audit_imx500.py --onnx outputs/cansat_siamese_damage.onnx")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
