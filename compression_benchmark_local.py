"""
Benchmark de Compressão de Modelos — ResNet-18 × PlantVillage
Grupo 10 — Projeto Transformador II — PUCPR

Correções aplicadas em relação à versão original:
  [6B] Structured Pruning: reconstrução REAL da arquitetura (MACs/fp32_MB/latência caem de verdade)
  [6B] Structured Pruning: variantes com e sem fine-tune para todos os níveis (simetria com unstructured)
  [7A] PTQ Dinâmica: remove Conv2d do alvo (PyTorch ignora silenciosamente — só Linear funciona)
  [7B] PTQ Estática: troca eager mode por FX-graph (suporte a skip connections da ResNet)
  [7B] PTQ Estática: calibração ampliada de 16 → 200 batches
  [7C] QAT: troca prepare_qat por prepare_qat_fx + congelamento de BN nas últimas épocas
  [7*] Todas as métricas de quantização medidas in-process (elimina bug de reload com strict=False)
  [7E] Novo combo: Structured 60% + QAT (motivado por Guerra et al. 2020 — arXiv:2002.00523)

Como usar:
  1. pip install -r requirements.txt
  2. Coloque kaggle.json em ~/.kaggle/kaggle.json
  3. Coloque best_model.pth na pasta do script
  4. python compression_benchmark_local.py
"""

import os, copy, time, random, glob, gzip, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models.resnet import BasicBlock
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import warnings
warnings.filterwarnings('ignore')

# ── Reproducibilidade ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CPU    = torch.device('cpu')
print(f'Device: {DEVICE}')
if DEVICE.type == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}')

# ── Hiperparâmetros de quantização (ajuste aqui para testes rápidos) ──────────
QAT_EPOCHS    = 10   # reduza para 3 em CPU sem GPU
CALIB_BATCHES = 200  # batches de calibração para PTQ estática


# ══════════════════════════════════════════════════════════════════════════════
# 2. Dataset PlantVillage
# ══════════════════════════════════════════════════════════════════════════════
DATA_DIR = 'PlantVillage'

if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f'Diretório "{DATA_DIR}" não encontrado.')

classes_found = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
print(f'Classes encontradas: {len(classes_found)}')


# ══════════════════════════════════════════════════════════════════════════════
# 3. Transforms e DataLoaders
# ══════════════════════════════════════════════════════════════════════════════
IMG_SIZE    = 224
BATCH_SIZE  = 64
NUM_WORKERS = min(4, os.cpu_count() or 2)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
eval_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

class TransformSubset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices, transform):
        self.dataset   = dataset
        self.indices   = indices
        self.transform = transform
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, i):
        img, label = self.dataset[self.indices[i]]
        return self.transform(img), label


CLASSES_15 = [
    'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
    'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy',
]

full_dataset = datasets.ImageFolder(DATA_DIR)
all_classes  = full_dataset.classes

valid_class_idx = {all_classes.index(c): new_i
                   for new_i, c in enumerate(CLASSES_15)
                   if c in all_classes}

filtered_samples = [(path, valid_class_idx[lbl])
                    for path, lbl in full_dataset.samples
                    if lbl in valid_class_idx]

full_dataset.samples      = filtered_samples
full_dataset.targets      = [lbl for _, lbl in filtered_samples]
full_dataset.classes      = CLASSES_15
full_dataset.class_to_idx = {c: i for i, c in enumerate(CLASSES_15)}

CLASSES     = CLASSES_15
NUM_CLASSES = len(CLASSES)
labels      = np.array(full_dataset.targets)
print(f'Total de imagens (15 classes): {len(full_dataset)} | Classes: {NUM_CLASSES}')

sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
trainval_idx, test_idx = next(sss1.split(np.zeros(len(labels)), labels))
sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.15/0.85, random_state=SEED)
train_idx, val_idx = next(sss2.split(np.zeros(len(trainval_idx)), labels[trainval_idx]))
train_idx = trainval_idx[train_idx]
val_idx   = trainval_idx[val_idx]
print(f'Split → train: {len(train_idx)} | val: {len(val_idx)} | test: {len(test_idx)}')

train_ds = TransformSubset(full_dataset, train_idx, train_tf)
val_ds   = TransformSubset(full_dataset, val_idx,   eval_tf)
test_ds  = TransformSubset(full_dataset, test_idx,  eval_tf)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == 'cuda'))
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == 'cuda'))
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == 'cuda'))

print('DataLoaders prontos!')


# ══════════════════════════════════════════════════════════════════════════════
# 4. Funções utilitárias
# ══════════════════════════════════════════════════════════════════════════════
def build_model(num_classes):
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


@torch.no_grad()
def evaluate(model, loader, device=DEVICE):
    criterion = nn.CrossEntropyLoss()
    model.eval().to(device)
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        logits = model(imgs)
        loss   = criterion(logits, lbls)
        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(1)
        correct += (preds == lbls).sum().item()
        total   += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(lbls.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_labels


def save_sparse(model, path):
    sparse_state = {}
    for k, v in model.state_dict().items():
        if (isinstance(v, torch.Tensor)
                and v.dtype == torch.float32
                and v.dim() >= 2):
            sparse_state[k] = v.to_sparse().coalesce()
        else:
            sparse_state[k] = v
    with gzip.open(path, 'wb', compresslevel=9) as f:
        pickle.dump(sparse_state, f, protocol=4)


def load_sparse(model, path):
    with gzip.open(path, 'rb') as f:
        sparse_state = pickle.load(f)
    dense_state = {k: v.to_dense() if v.is_sparse else v for k, v in sparse_state.items()}
    model.load_state_dict(dense_state)
    return model


def model_size_mb(model, tmp_path='/tmp/_size_check.pth.gz'):
    save_sparse(model, tmp_path)
    size = os.path.getsize(tmp_path)
    os.remove(tmp_path)
    return size / 1e6


def count_nonzero_params(model):
    total   = sum(p.nelement() for p in model.parameters())
    nonzero = sum(p.nonzero().size(0) for p in model.parameters())
    return total, nonzero


def count_macs(model, input_size=(1, 3, 224, 224), device=CPU):
    """Conta MACs reais via hooks — reflete a arquitetura reconstruída."""
    total_macs = [0]

    def conv_hook(module, inp, out):
        b, c_out, h_out, w_out = out.shape
        c_in   = module.in_channels
        kH, kW = module.kernel_size
        groups = module.groups
        total_macs[0] += b * c_out * h_out * w_out * (c_in // groups) * kH * kW

    def linear_hook(module, inp, out):
        b = inp[0].shape[0]
        total_macs[0] += b * module.in_features * module.out_features

    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))

    model.eval().to(device)
    dummy = torch.randn(*input_size).to(device)
    with torch.no_grad():
        model(dummy)
    for h in hooks:
        h.remove()
    return total_macs[0] / 1e6


def measure_latency(model, device=DEVICE, n_runs=200, batch_size=1):
    model.eval().to(device)
    dummy = torch.randn(batch_size, 3, IMG_SIZE, IMG_SIZE).to(device)
    for _ in range(20):
        with torch.no_grad():
            _ = model(dummy)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_runs):
        with torch.no_grad():
            _ = model(dummy)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / n_runs * 1000


def print_metrics(name, acc, size_mb, latency_ms, total_params, nonzero_params,
                  macs_m=None):
    sparsity = 100 * (1 - nonzero_params / total_params) if total_params > 0 else 0
    print(f'\n{"="*55}')
    print(f'  {name}')
    print(f'{"="*55}')
    print(f'  Acurácia (Top-1):  {acc*100:.2f}%')
    print(f'  Tamanho em disco:  {size_mb:.2f} MB')
    print(f'  Latência (1 img):  {latency_ms:.2f} ms')
    if macs_m is not None:
        print(f'  MACs reais:        {macs_m:.1f} M')
    print(f'  Parâmetros totais: {total_params:,}')
    print(f'  Parâmetros ativos: {nonzero_params:,}')
    print(f'  Sparsidade:        {sparsity:.1f}%')


RESULTS = {}

def save_result(name, acc, size_mb, latency_ms, total_params, nonzero_params,
                macs_m=None):
    RESULTS[name] = dict(
        accuracy=round(acc * 100, 2),
        size_mb=round(size_mb, 2),
        latency_ms=round(latency_ms, 2),
        macs_M=round(macs_m, 2) if macs_m is not None else float('nan'),
        total_params=total_params,
        nonzero_params=nonzero_params,
        sparsity=round(100 * (1 - nonzero_params / total_params), 1) if total_params > 0 else 0,
    )

print('Utilitários carregados!')


# ══════════════════════════════════════════════════════════════════════════════
# 5. Baseline
# ══════════════════════════════════════════════════════════════════════════════
PTH_PATH = 'best_model.pth'

if not os.path.exists(PTH_PATH):
    raise FileNotFoundError(
        f'Arquivo "{PTH_PATH}" não encontrado.\n'
        'Coloque o best_model.pth na mesma pasta deste script.'
    )

baseline_model = build_model(NUM_CLASSES)
state = torch.load(PTH_PATH, map_location='cpu', weights_only=False)
baseline_model.load_state_dict(state)
baseline_model = baseline_model.to(DEVICE)
print('Modelo base carregado com sucesso!')

_, base_acc, base_preds, base_labels = evaluate(baseline_model, test_loader)
base_size    = model_size_mb(baseline_model)
base_latency = measure_latency(baseline_model)
base_macs    = count_macs(baseline_model)
base_total, base_nz = count_nonzero_params(baseline_model)

print_metrics('BASELINE', base_acc, base_size, base_latency,
              base_total, base_nz, base_macs)
save_result('Baseline', base_acc, base_size, base_latency,
            base_total, base_nz, base_macs)

save_sparse(baseline_model, 'baseline.pth.gz')
print('Salvo: baseline.pth.gz')


# ══════════════════════════════════════════════════════════════════════════════
# 6. PRUNING
# ══════════════════════════════════════════════════════════════════════════════

# ── Utilitários de pruning (compartilhados) ───────────────────────────────────
def apply_unstructured_pruning(model, amount=0.4):
    model = copy.deepcopy(model)
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.l1_unstructured(module, name='weight', amount=amount)
    return model


def remove_pruning_masks(model):
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            try:
                prune.remove(module, 'weight')
            except ValueError:
                pass
    return model


def finetune(model, train_loader, val_loader, epochs, lr, device=DEVICE):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    best_acc, best_state = 0.0, None

    for epoch in range(1, epochs + 1):
        model.train().to(device)
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            criterion(model(imgs), lbls).backward()
            optimizer.step()
        model.eval()
        _, val_acc, _, _ = evaluate(model, val_loader, device)
        print(f'    Época {epoch}/{epochs}  val_acc={val_acc*100:.2f}%')
        if val_acc > best_acc:
            best_acc   = val_acc
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model


# ── 6A. Unstructured Pruning (sem fine-tune) ──────────────────────────────────
UNSTRUCTURED_AMOUNTS = [0.30, 0.40, 0.50]

for amount in UNSTRUCTURED_AMOUNTS:
    pruned = apply_unstructured_pruning(baseline_model, amount=amount)
    pruned = remove_pruning_masks(pruned)
    _, acc, _, _ = evaluate(pruned, test_loader)
    size_mb  = model_size_mb(pruned)
    latency  = measure_latency(pruned)
    macs_m   = count_macs(pruned)
    tot, nz  = count_nonzero_params(pruned)
    label = f'Unstructured Pruning {int(amount*100)}%'
    print_metrics(label, acc, size_mb, latency, tot, nz, macs_m)
    save_result(label, acc, size_mb, latency, tot, nz, macs_m)
    fname = f'pruned_unstructured_{int(amount*100)}.pth.gz'
    save_sparse(pruned, fname)
    print(f'  Salvo: {fname}')


# # ── 6B. Structured Pruning — CORRIGIDO ───────────────────────────────────────
# #
# # PROBLEMA ORIGINAL:
# #   apply_structured_pruning() retornava (model, safe_names) sem podar nada.
# #   prune.ln_structured aplicava máscaras (zeros), mas o tensor mantinha o mesmo
# #   shape → MACs, fp32_MB e latência idênticos ao baseline em todos os níveis.
# #
# # CORREÇÃO:
# #   Reconstrução real da arquitetura: Conv2d e BatchNorm são substituídos por
# #   versões menores, copiando apenas os pesos dos filtros sobreviventes (seleção
# #   por L1-norm, mesmo critério do ln_structured).
# #   Resultado: MACs, params_total, fp32_MB e latência realmente diminuem.
# #
# # REGRA DOS SKIP CONNECTIONS:
# #   Apenas conv1 de cada BasicBlock é podada (canais intermediários diminuem).
# #   conv2 mantém os canais de saída originais para não quebrar a compatibilidade
# #   dimensional com o bloco seguinte nem com o downsample shortcut.

# def get_surviving_indices(conv, amount):
#     """Retorna índices dos filtros com maior L1-norm (sobreviventes à poda)."""
#     with torch.no_grad():
#         norms = conv.weight.data.abs().sum(dim=(1, 2, 3))
#     n_keep = max(1, int(norms.shape[0] * (1 - amount)))
#     _, indices = torch.topk(norms, n_keep)
#     return indices.sort().values


# def rebuild_conv(old_conv, out_indices, in_indices=None):
#     """Reconstrói Conv2d mantendo só os filtros/canais selecionados."""
#     in_ch  = len(in_indices) if in_indices is not None else old_conv.in_channels
#     out_ch = len(out_indices)
#     new_conv = nn.Conv2d(
#         in_ch, out_ch,
#         kernel_size=old_conv.kernel_size,
#         stride=old_conv.stride,
#         padding=old_conv.padding,
#         bias=(old_conv.bias is not None),
#         groups=old_conv.groups,
#     )
#     with torch.no_grad():
#         w = old_conv.weight.data[out_indices]
#         if in_indices is not None:
#             w = w[:, in_indices]
#         new_conv.weight.data.copy_(w)
#         if old_conv.bias is not None:
#             new_conv.bias.data.copy_(old_conv.bias.data[out_indices])
#     return new_conv


# def rebuild_bn(old_bn, indices):
#     """Reconstrói BatchNorm2d mantendo só os canais selecionados."""
#     new_bn = nn.BatchNorm2d(
#         len(indices), eps=old_bn.eps,
#         momentum=old_bn.momentum, affine=old_bn.affine,
#     )
#     if old_bn.affine:
#         with torch.no_grad():
#             new_bn.weight.data.copy_(old_bn.weight.data[indices])
#             new_bn.bias.data.copy_(old_bn.bias.data[indices])
#             new_bn.running_mean.copy_(old_bn.running_mean[indices])
#             new_bn.running_var.copy_(old_bn.running_var[indices])
#     return new_bn


# def rebuild_basicblock(block, amount):
#     """
#     Reconstrói BasicBlock podando conv1 (canais intermediários) e ajustando
#     a entrada de conv2 de acordo. Saída de conv2 é mantida para preservar
#     compatibilidade com o bloco seguinte e o skip connection.
#     """
#     surviving = get_surviving_indices(block.conv1, amount)
#     out_ch    = block.conv2.out_channels

#     new_block = BasicBlock.__new__(BasicBlock)
#     nn.Module.__init__(new_block)

#     new_block.conv1 = rebuild_conv(block.conv1, surviving)
#     new_block.bn1   = rebuild_bn(block.bn1, surviving)
#     new_block.relu  = nn.ReLU(inplace=True)
#     new_block.conv2 = rebuild_conv(block.conv2, torch.arange(out_ch), surviving)
#     new_block.bn2   = rebuild_bn(block.bn2, torch.arange(out_ch))
#     new_block.downsample = block.downsample
#     new_block.stride     = block.stride

#     return new_block


# def rebuild_resnet18_structured(model, amount):
#     """
#     Reconstrói a ResNet-18 com arquitetura real menor.
#     Camadas preservadas: conv1 global, fc, downsample shortcuts.
#     Camadas podadas: conv1 de cada BasicBlock nas layers 1-4.
#     """
#     new_model = copy.deepcopy(model)
#     for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
#         layer      = getattr(new_model, layer_name)
#         new_blocks = [rebuild_basicblock(block, amount) for block in layer]
#         setattr(new_model, layer_name, nn.Sequential(*new_blocks))
#     return new_model


# # Loop: variantes SEM e COM fine-tune para cada nível (simetria com unstructured)
# STRUCTURED_AMOUNTS = [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70]
# STRUCT_FT_EPOCHS   = 5
# STRUCT_FT_LR       = 5e-5

# for amount in STRUCTURED_AMOUNTS:
#     pct = int(amount * 100)
#     print(f'\n{"="*60}')
#     print(f'  Structured Pruning {pct}%')
#     print(f'{"="*60}')

#     # Sem fine-tune
#     pruned = rebuild_resnet18_structured(baseline_model, amount)
#     _, acc, _, _ = evaluate(pruned, test_loader)
#     size_mb  = model_size_mb(pruned)
#     latency  = measure_latency(pruned)
#     macs_m   = count_macs(pruned)
#     tot, nz  = count_nonzero_params(pruned)
#     label = f'Structured Pruning {pct}%'
#     print_metrics(label, acc, size_mb, latency, tot, nz, macs_m)
#     save_result(label, acc, size_mb, latency, tot, nz, macs_m)
#     save_sparse(pruned, f'pruned_structured_{pct}.pth.gz')
#     print(f'  Salvo: pruned_structured_{pct}.pth.gz')

#     # Com fine-tune
#     print(f'\n  Fine-tuning ({STRUCT_FT_EPOCHS} épocas) ...')
#     pruned_ft = finetune(copy.deepcopy(pruned), train_loader, val_loader,
#                          epochs=STRUCT_FT_EPOCHS, lr=STRUCT_FT_LR)
#     _, acc_ft, _, _ = evaluate(pruned_ft, test_loader)
#     size_ft  = model_size_mb(pruned_ft)
#     lat_ft   = measure_latency(pruned_ft)
#     macs_ft  = count_macs(pruned_ft)
#     tot_ft, nz_ft = count_nonzero_params(pruned_ft)
#     label_ft = f'Structured Pruning {pct}% + Finetune'
#     print_metrics(label_ft, acc_ft, size_ft, lat_ft, tot_ft, nz_ft, macs_ft)
#     save_result(label_ft, acc_ft, size_ft, lat_ft, tot_ft, nz_ft, macs_ft)
#     save_sparse(pruned_ft, f'pruned_structured_{pct}_finetuned.pth.gz')
#     print(f'  Salvo: pruned_structured_{pct}_finetuned.pth.gz')


# ── 6D. Iterative Pruning ─────────────────────────────────────────────────────
def iterative_pruning(model, target_amount=0.90, rounds=5, finetune_epochs=3,
                      lr=1e-4, device=DEVICE):
    model = copy.deepcopy(model).to(device)
    amounts = [1 - (1 - target_amount) ** ((r + 1) / rounds) for r in range(rounds)]
    for r, amount in enumerate(amounts):
        print(f'\n--- Round {r+1}/{rounds} | Sparsidade alvo: {amount*100:.1f}% ---')
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                prune.l1_unstructured(module, name='weight', amount=amount)
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        for epoch in range(finetune_epochs):
            for imgs, lbls in train_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                optimizer.zero_grad()
                criterion(model(imgs), lbls).backward()
                optimizer.step()
        _, val_acc, _, _ = evaluate(model, val_loader, device)
        tot, nz = count_nonzero_params(model)
        print(f'  val_acc={val_acc*100:.2f}%  |  params ativos: {nz:,}')
    model = remove_pruning_masks(model)
    return model


for target in [0.50]:
    print(f'\n{"="*60}')
    print(f'  Pruning Iterativo — alvo {int(target*100)}%')
    print(f'{"="*60}')
    iter_model = iterative_pruning(baseline_model, target_amount=target,
                                   rounds=5, finetune_epochs=3)
    _, acc, _, _ = evaluate(iter_model, test_loader)
    size_mb  = model_size_mb(iter_model)
    latency  = measure_latency(iter_model)
    macs_m   = count_macs(iter_model)
    tot, nz  = count_nonzero_params(iter_model)
    label = f'Iterative Pruning {int(target*100)}%'
    print_metrics(label, acc, size_mb, latency, tot, nz, macs_m)
    save_result(label, acc, size_mb, latency, tot, nz, macs_m)
    save_sparse(iter_model, f'pruned_iterative_{int(target*100)}.pth.gz')
    print(f'  Salvo: pruned_iterative_{int(target*100)}.pth.gz')


# ── 6E. Unstructured Agressivo + Finetune ────────────────────────────────────
for amount in [0.70, 0.75, 0.80, 0.85]:
    pruned = apply_unstructured_pruning(baseline_model, amount=amount)
    print(f'\nFine-tuning após Unstructured {int(amount*100)}% ...')
    pruned = finetune(pruned, train_loader, val_loader, epochs=5, lr=1e-4)
    pruned = remove_pruning_masks(pruned)
    _, acc, _, _ = evaluate(pruned, test_loader)
    size_mb  = model_size_mb(pruned)
    latency  = measure_latency(pruned)
    macs_m   = count_macs(pruned)
    tot, nz  = count_nonzero_params(pruned)
    label = f'Unstructured Pruning {int(amount*100)}% + Finetune'
    print_metrics(label, acc, size_mb, latency, tot, nz, macs_m)
    save_result(label, acc, size_mb, latency, tot, nz, macs_m)
    save_sparse(pruned, f'pruned_unstructured_{int(amount*100)}_finetuned.pth.gz')
    print(f'  Salvo: pruned_unstructured_{int(amount*100)}_finetuned.pth.gz')


# # ══════════════════════════════════════════════════════════════════════════════
# # 7. QUANTIZATION — Corrigida
# # ══════════════════════════════════════════════════════════════════════════════

# # ── 7A. PTQ Dinâmica — corrigida ──────────────────────────────────────────────
# #
# # PROBLEMA ORIGINAL: passava {nn.Linear, nn.Conv2d} mas PyTorch silenciosamente
# #   ignora Conv2d no quantize_dynamic → modelo ficava FP32 por inteiro (42 MB).
# #
# # CORREÇÃO: alvo apenas nn.Linear (único suportado de fato). Para CNNs o ganho
# #   é pequeno (só a cabeça é quantizada), mas a acurácia é preservada e o
# #   resultado é honesto. Ganho real em Conv2d vem do QAT (7C).
# #
# # Acurácia medida IN-PROCESS (antes de qualquer save/load) — elimina bug de
# # reload com strict=False que afetava a versão anterior.

# def ptq_dynamic(model):
#     model_cpu = copy.deepcopy(model).to(CPU)
#     model_cpu.eval()
#     return torch.quantization.quantize_dynamic(
#         model_cpu,
#         {nn.Linear},   # Conv2d removido — ineficaz no quantize_dynamic
#         dtype=torch.qint8,
#     )


# print('\n' + '='*60)
# print('  7A. PTQ Dinâmica (INT8) — corrigida')
# print('='*60)
# ptq_dyn_model = ptq_dynamic(baseline_model)

# _, acc_dyn, _, _ = evaluate(ptq_dyn_model, test_loader, device=CPU)
# size_dyn    = model_size_mb(ptq_dyn_model)
# latency_dyn = measure_latency(ptq_dyn_model, device=CPU)
# tot_dyn, nz_dyn = count_nonzero_params(baseline_model)

# label = 'PTQ Dynamic (INT8)'
# print_metrics(label, acc_dyn, size_dyn, latency_dyn, tot_dyn, nz_dyn)
# print('  ℹ️  PTQ dinâmica quantiza só Linear → ganho de tamanho pequeno em CNNs.')
# save_result(label, acc_dyn, size_dyn, latency_dyn, tot_dyn, nz_dyn)

# with gzip.open('ptq_dynamic_int8.pth.gz', 'wb', compresslevel=9) as f:
#     pickle.dump(ptq_dyn_model.state_dict(), f, protocol=4)
# print('Salvo: ptq_dynamic_int8.pth.gz')


# # ── 7B. PTQ Estática — corrigida via FX-graph mode ───────────────────────────
# #
# # PROBLEMA ORIGINAL: prepare/convert clássico (eager mode) não insere observers
# #   nos skip connections da ResNet → escalas INT8 erradas → colapso (11% = acaso).
# #   Versão anterior registrava N/A sem tentar a API correta.
# #   Calibração com 16 batches era insuficiente para estimar min/max das ativações.
# #
# # CORREÇÃO: usa quantize_fx (FX-graph mode) que traça o grafo completo e insere
# #   observers em todas as operações, incluindo as adições residuais.
# #   Calibração ampliada para CALIB_BATCHES (200) batches.
# #   Acurácia medida in-process antes de qualquer serialização.

# print('\n' + '='*60)
# print('  7B. PTQ Estática (INT8) — FX-graph mode, corrigida')
# print('='*60)

# try:
#     from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
#     from torch.ao.quantization import QConfigMapping

#     model_cpu = copy.deepcopy(baseline_model).to(CPU)
#     model_cpu.eval()

#     qconfig_mapping = QConfigMapping().set_global(
#         torch.ao.quantization.get_default_qconfig('fbgemm')
#     )

#     example_input  = torch.randn(1, 3, 224, 224)
#     model_prepared = prepare_fx(model_cpu, qconfig_mapping, example_input)

#     print(f'  Calibrando com {CALIB_BATCHES} batches ...')
#     batches_done = 0
#     with torch.no_grad():
#         for imgs, _ in train_loader:
#             if batches_done >= CALIB_BATCHES:
#                 break
#             model_prepared(imgs.to(CPU))
#             batches_done += 1
#             if batches_done % 50 == 0:
#                 print(f'    {batches_done}/{CALIB_BATCHES} batches')

#     ptq_static_model = convert_fx(model_prepared)

#     # Acurácia IN-PROCESS — antes de qualquer save/load
#     _, acc_static, _, _ = evaluate(ptq_static_model, test_loader, device=CPU)
#     print(f'\n  ✅ Acurácia in-process: {acc_static*100:.2f}%')

#     size_static    = model_size_mb(ptq_static_model)
#     latency_static = measure_latency(ptq_static_model, device=CPU)
#     tot_s, nz_s    = count_nonzero_params(baseline_model)

#     label = 'PTQ Static FX (INT8)'
#     print_metrics(label, acc_static, size_static, latency_static, tot_s, nz_s)
#     save_result(label, acc_static, size_static, latency_static, tot_s, nz_s)

#     with gzip.open('ptq_static_fx_int8.pth.gz', 'wb', compresslevel=9) as f:
#         pickle.dump(ptq_static_model.state_dict(), f, protocol=4)
#     print('Salvo: ptq_static_fx_int8.pth.gz')

# except Exception as e:
#     print(f'\n  ⚠️  PTQ Estática falhou: {e}')
#     print('     Backend fbgemm ausente (ARM/macOS) ou PyTorch < 1.13. Registrando N/A.')
#     RESULTS['PTQ Static FX (INT8)'] = dict(
#         accuracy=float('nan'), size_mb=float('nan'), latency_ms=float('nan'),
#         macs_M=float('nan'), total_params=0, nonzero_params=0,
#         sparsity=float('nan'), note=str(e)[:120],
#     )


# # ── 7C. QAT — corrigido ───────────────────────────────────────────────────────
# #
# # PROBLEMAS ORIGINAIS:
# #   1. prepare_qat clássico (eager) quebrava nos residuals → mesmo bug da 7B.
# #   2. Poucas épocas em CPU → convergência insuficiente → 75% (esperado ~99%).
# #   3. Acurácia medida após reload com strict=False → potencialmente corrompida.
# #
# # CORREÇÕES:
# #   1. prepare_qat_fx — lida com residuals via FX graph.
# #   2. QAT_EPOCHS = 10 (configurável no topo do arquivo).
# #   3. Congelamento de BN e observers nas últimas 3 épocas (padrão moderno).
# #   4. Acurácia medida in-process antes de qualquer save.

# def qat_train_fx(model, train_loader, val_loader, epochs=QAT_EPOCHS, lr=1e-4):
#     from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx
#     from torch.ao.quantization import QConfigMapping

#     model_cpu = copy.deepcopy(model).to(CPU)
#     model_cpu.train()

#     qconfig_mapping = QConfigMapping().set_global(
#         torch.ao.quantization.get_default_qat_qconfig('fbgemm')
#     )

#     example_input  = torch.randn(1, 3, 224, 224)
#     model_prepared = prepare_qat_fx(model_cpu, qconfig_mapping, example_input)

#     optimizer = optim.Adam(model_prepared.parameters(), lr=lr, weight_decay=1e-4)
#     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
#     criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
#     best_acc, best_state = 0.0, None
#     freeze_epoch = max(1, epochs - 3)

#     for epoch in range(1, epochs + 1):
#         if epoch == freeze_epoch:
#             print(f'  [Época {epoch}] Congelando BatchNorm e observers ...')
#             model_prepared.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
#             model_prepared.apply(torch.ao.quantization.disable_observer)

#         model_prepared.train()
#         for imgs, lbls in train_loader:
#             imgs, lbls = imgs.to(CPU), lbls.to(CPU)
#             optimizer.zero_grad()
#             criterion(model_prepared(imgs), lbls).backward()
#             optimizer.step()
#         scheduler.step()

#         model_prepared.eval()
#         _, val_acc, _, _ = evaluate(model_prepared, val_loader, device=CPU)
#         print(f'  Época {epoch:02d}/{epochs}  val_acc={val_acc*100:.2f}%')

#         if val_acc > best_acc:
#             best_acc   = val_acc
#             best_state = copy.deepcopy(model_prepared.state_dict())

#     model_prepared.load_state_dict(best_state)
#     model_prepared.eval()
#     return convert_fx(model_prepared)


# print('\n' + '='*60)
# print(f'  7C. QAT (INT8) — FX-graph mode, {QAT_EPOCHS} épocas')
# print('='*60)

# try:
#     qat_model = qat_train_fx(baseline_model, train_loader, val_loader,
#                              epochs=QAT_EPOCHS)

#     _, acc_qat, _, _ = evaluate(qat_model, test_loader, device=CPU)
#     print(f'\n  ✅ Acurácia in-process: {acc_qat*100:.2f}%')

#     size_qat    = model_size_mb(qat_model)
#     latency_qat = measure_latency(qat_model, device=CPU)
#     tot_q, nz_q = count_nonzero_params(baseline_model)

#     label = 'QAT (INT8)'
#     print_metrics(label, acc_qat, size_qat, latency_qat, tot_q, nz_q)
#     save_result(label, acc_qat, size_qat, latency_qat, tot_q, nz_q)

#     with gzip.open('qat_int8.pth.gz', 'wb', compresslevel=9) as f:
#         pickle.dump(qat_model.state_dict(), f, protocol=4)
#     print('Salvo: qat_int8.pth.gz')

# except Exception as e:
#     print(f'\n  ⚠️  QAT falhou: {e}')
#     RESULTS['QAT (INT8)'] = dict(
#         accuracy=float('nan'), size_mb=float('nan'), latency_ms=float('nan'),
#         macs_M=float('nan'), total_params=0, nonzero_params=0,
#         sparsity=float('nan'), note=str(e)[:120],
#     )


# # ── 7D. Combo: Unstructured 70% + PTQ Dinâmica ───────────────────────────────
# print('\n' + '='*60)
# print('  7D. Combo: Unstructured 70% + PTQ Dinâmica (INT8)')
# print('='*60)

# pruned_70 = apply_unstructured_pruning(baseline_model, amount=0.70)
# pruned_70 = remove_pruning_masks(pruned_70)
# print('  Fine-tuning pós-poda (3 épocas) ...')
# pruned_70 = finetune(pruned_70, train_loader, val_loader, epochs=3, lr=1e-4)
# combo_ptq = ptq_dynamic(pruned_70)

# _, acc_combo, _, _ = evaluate(combo_ptq, test_loader, device=CPU)
# size_combo    = model_size_mb(combo_ptq)
# latency_combo = measure_latency(combo_ptq, device=CPU)
# tot_c, nz_c   = count_nonzero_params(pruned_70)

# label = 'Unstructured 70% + PTQ INT8'
# print_metrics(label, acc_combo, size_combo, latency_combo, tot_c, nz_c)
# save_result(label, acc_combo, size_combo, latency_combo, tot_c, nz_c)

# with gzip.open('combo_unstruct70_ptq.pth.gz', 'wb', compresslevel=9) as f:
#     pickle.dump(combo_ptq.state_dict(), f, protocol=4)
# print('Salvo: combo_unstruct70_ptq.pth.gz')


# # ── 7E. [NOVO] Combo: Structured 60% + QAT ───────────────────────────────────
# #
# # Motivado por Guerra et al. (2020) — arXiv:2002.00523:
# #   Combinar poda ESTRUTURADA (reduz MACs reais) com QAT (recupera acurácia via
# #   treino consciente da quantização) produz os melhores trade-offs de compressão.
# #   Para ResNet-18, os autores mostram redução > 26% do tamanho com < 3pp de perda.
# #
# # Fluxo: baseline → structured 60% (arquitetura real menor) → fine-tune
# #        intercalado → QAT por QAT_EPOCHS épocas → INT8 real.

# print('\n' + '='*60)
# print('  7E. [NOVO] Combo: Structured 60% + QAT (INT8)')
# print('      Baseado em Guerra et al. 2020 — melhor trade-off teórico')
# print('='*60)

# try:
#     pruned_struct = rebuild_resnet18_structured(baseline_model, amount=0.60)
#     print('  Fine-tuning pós-poda estruturada (5 épocas) ...')
#     pruned_struct = finetune(pruned_struct, train_loader, val_loader,
#                              epochs=5, lr=STRUCT_FT_LR)

#     print('\n  Iniciando QAT no modelo podado ...')
#     combo_qat = qat_train_fx(pruned_struct, train_loader, val_loader,
#                              epochs=QAT_EPOCHS, lr=5e-5)

#     _, acc_sq, _, _ = evaluate(combo_qat, test_loader, device=CPU)
#     print(f'\n  ✅ Acurácia in-process: {acc_sq*100:.2f}%')

#     size_sq    = model_size_mb(combo_qat)
#     latency_sq = measure_latency(combo_qat, device=CPU)
#     macs_sq    = count_macs(pruned_struct)   # MACs da arquitetura podada
#     tot_sq, nz_sq = count_nonzero_params(pruned_struct)

#     label = 'Structured 60% + QAT INT8'
#     print_metrics(label, acc_sq, size_sq, latency_sq, tot_sq, nz_sq, macs_sq)
#     save_result(label, acc_sq, size_sq, latency_sq, tot_sq, nz_sq, macs_sq)

#     with gzip.open('combo_struct60_qat.pth.gz', 'wb', compresslevel=9) as f:
#         pickle.dump(combo_qat.state_dict(), f, protocol=4)
#     print('Salvo: combo_struct60_qat.pth.gz')

# except Exception as e:
#     print(f'\n  ⚠️  Combo Structured+QAT falhou: {e}')
#     RESULTS['Structured 60% + QAT INT8'] = dict(
#         accuracy=float('nan'), size_mb=float('nan'), latency_ms=float('nan'),
#         macs_M=float('nan'), total_params=0, nonzero_params=0,
#         sparsity=float('nan'), note=str(e)[:120],
#     )


# ══════════════════════════════════════════════════════════════════════════════
# 8. Tabela comparativa
# ══════════════════════════════════════════════════════════════════════════════
import pandas as pd

df = pd.DataFrame(RESULTS).T.reset_index()
df.columns = ['Modelo', 'Acurácia (%)', 'Tamanho (MB)', 'Latência (ms)',
              'MACs (M)', 'Params Totais', 'Params Ativos', 'Sparsidade (%)']
df = df.sort_values('Acurácia (%)', ascending=False).reset_index(drop=True)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 140)
print('\n')
print(df.to_string(index=False))

df.to_csv('benchmark_results.csv', index=False)
print('\nSalvo: benchmark_results.csv')


# ══════════════════════════════════════════════════════════════════════════════
# 9. Curva de Pareto — Acurácia vs. Tamanho e Acurácia vs. Latência
# ══════════════════════════════════════════════════════════════════════════════
try:
    from adjustText import adjust_text
    HAS_ADJUST = True
except ImportError:
    HAS_ADJUST = False
    print('adjustText não instalado — pip install adjustText para labels sem sobreposição')

df2 = df.copy()
for col in ['Acurácia (%)', 'Sparsidade (%)', 'Latência (ms)', 'Tamanho (MB)', 'MACs (M)']:
    df2[col] = pd.to_numeric(df2[col], errors='coerce')
df2 = df2.dropna(subset=['Acurácia (%)'])

def categorize(nome):
    if 'Baseline'     in nome: return 'Baseline'
    if 'Iterative'    in nome: return 'Iterativo'
    if 'Unstructured' in nome and '+' not in nome: return 'Unstructured'
    if 'Structured'   in nome and '+' not in nome: return 'Structured'
    if any(x in nome for x in ['PTQ', 'QAT', 'INT8']): return 'Quantização'
    if '+' in nome: return 'Combo'
    return 'Outro'

CORES = {
    'Baseline':    '#5F5E5A', 'Unstructured': '#378ADD',
    'Iterativo':   '#7F77DD', 'Structured':   '#1D9E75',
    'Quantização': '#BA7517', 'Combo':        '#D85A30',
    'Outro':       '#888780',
}
MARKERS = {
    'Baseline': 'D', 'Unstructured': 'o', 'Iterativo': 's',
    'Structured': '^', 'Quantização': 'P', 'Combo': '*', 'Outro': 'X',
}

df2['Cat']    = df2['Modelo'].apply(categorize)
df2['Cor']    = df2['Cat'].map(CORES)
df2['Marker'] = df2['Cat'].map(MARKERS)

_base    = df2.loc[df2['Modelo'] == 'Baseline', 'Acurácia (%)'].values
BASE_ACC = float(_base[0]) if len(_base) else None

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor('#FAFAFA')

def style_ax(ax):
    ax.set_facecolor('#FFFFFF')
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#DDDDDD')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.6, zorder=0)
    ax.tick_params(colors='#888780', labelsize=10)
    if BASE_ACC is not None:
        ax.axhline(BASE_ACC * 0.95, color='#E24B4A', linestyle=':', linewidth=1.3, alpha=0.8, zorder=1)
        ax.axhline(BASE_ACC * 0.90, color='#D85A30', linestyle=':', linewidth=1.3, alpha=0.6, zorder=1)

def plot_scatter(ax, x_col, xlabel):
    style_ax(ax)
    texts = []
    for cat in df2['Cat'].unique():
        sub = df2[df2['Cat'] == cat]
        ax.scatter(sub[x_col], sub['Acurácia (%)'],
                   c=sub['Cor'], marker=MARKERS[cat],
                   s=160, zorder=5, edgecolors='white', linewidths=0.8, alpha=0.92)
    for _, row in df2.iterrows():
        if pd.isna(row['Acurácia (%)']) or pd.isna(row[x_col]):
            continue
        nome = (row['Modelo']
                .replace(' + Finetune', ' +FT')
                .replace(' (INT8)', '')
                .replace('Pruning ', ''))
        t = ax.text(row[x_col], row['Acurácia (%)'], nome,
                    fontsize=8.5, color='#2C2C2A', alpha=0.9, va='bottom', ha='left')
        texts.append(t)
    if HAS_ADJUST:
        adjust_text(texts, ax=ax,
                    arrowprops=dict(arrowstyle='-', color='#AAAAAA', lw=0.6),
                    expand_points=(1.8, 2.2), force_points=(0.4, 0.6), force_text=(0.5, 0.8))
    ax.set_xlabel(xlabel, fontsize=11, color='#444441', labelpad=8)
    ax.set_ylabel('Acurácia Top-1 (%)', fontsize=11, color='#444441', labelpad=8)
    ax.set_ylim(-5, 108)

plot_scatter(axes[0], 'Sparsidade (%)', 'Sparsidade (%)')
axes[0].set_title('Acurácia × Sparsidade', fontsize=13, fontweight='medium', color='#2C2C2A', pad=14)
axes[0].set_xlim(-3, 105)

plot_scatter(axes[1], 'Latência (ms)', 'Latência de inferência (ms) — batch=1')
axes[1].set_title('Acurácia × Latência', fontsize=13, fontweight='medium', color='#2C2C2A', pad=14)

cats_presentes = [c for c in CORES if c in df2['Cat'].values]
legend_handles = [
    mlines.Line2D([], [], color=CORES[cat], marker=MARKERS[cat],
                  linestyle='None', markersize=9, markeredgecolor='white',
                  markeredgewidth=0.6, label=cat)
    for cat in cats_presentes
]
ref_95 = mlines.Line2D([], [], color='#E24B4A', linestyle=':', linewidth=1.4, label='−5% acurácia baseline')
ref_90 = mlines.Line2D([], [], color='#D85A30', linestyle=':', linewidth=1.4, label='−10% acurácia baseline')

fig.legend(handles=legend_handles + [ref_95, ref_90],
           loc='lower center', ncol=6, fontsize=9.5,
           frameon=True, framealpha=0.95, edgecolor='#DDDDDD',
           bbox_to_anchor=(0.5, -0.05), handletextpad=0.5, columnspacing=1.2)

plt.suptitle('Pareto — ResNet-18 × PlantVillage (15 classes)',
             fontsize=14, fontweight='medium', y=1.02, color='#2C2C2A')
plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig('pareto_curves.png', dpi=180, bbox_inches='tight', facecolor='#FAFAFA')
plt.close()
print('Salvo: pareto_curves.png')

print('\n✅ Benchmark concluído! Arquivos gerados:')
for f in sorted(glob.glob('*.pth.gz') + ['benchmark_results.csv', 'pareto_curves.png']):
    if os.path.exists(f):
        print(f'  {f}')