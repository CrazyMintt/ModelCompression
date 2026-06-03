"""
Benchmark de Compressão de Modelos — ResNet-18 × PlantVillage
Grupo 10 — Projeto Transformador II — PUCPR
Adaptado para rodar LOCAL (sem Google Colab)

Como usar:
  1. Instale as dependências:        pip install -r requirements.txt
  2. Configure o Kaggle (uma vez):   coloque kaggle.json em ~/.kaggle/kaggle.json
  3. Coloque best_model.pth na pasta do script
  4. Execute:                        python compression_benchmark_local.py
"""

import os, copy, time, random, glob, gzip, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib
matplotlib.use('Agg')   # sem display gráfico — salva em arquivo
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
print(f'Device: {DEVICE}')
if DEVICE.type == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}')

# ══════════════════════════════════════════════════════════════════════════════
# 2. Download do dataset PlantVillage via kaggle CLI
# ══════════════════════════════════════════════════════════════════════════════
DATA_DIR = 'PlantVillage'

if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f'Diretório do dataset "{DATA_DIR}" não encontrado.')

classes_found = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
print(f'Classes encontradas: {len(classes_found)}')


# ══════════════════════════════════════════════════════════════════════════════
# 3. Transforms e DataLoaders
# ══════════════════════════════════════════════════════════════════════════════
IMG_SIZE    = 224
BATCH_SIZE  = 64
NUM_WORKERS = min(4, os.cpu_count() or 2)  # local pode usar mais cores

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
    'Pepper__bell___Bacterial_spot',
    'Pepper__bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus',
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
        # Tensores quantizados (qint8, quint8) não suportam to_sparse — salva denso
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


def print_metrics(name, acc, size_mb, latency_ms, total_params, nonzero_params):
    sparsity = 100 * (1 - nonzero_params / total_params)
    print(f'\n{"="*55}')
    print(f'  {name}')
    print(f'{"="*55}')
    print(f'  Acurácia (Top-1):  {acc*100:.2f}%')
    print(f'  Tamanho em disco:  {size_mb:.2f} MB')
    print(f'  Latência (1 img):  {latency_ms:.2f} ms')
    print(f'  Parâmetros totais: {total_params:,}')
    print(f'  Parâmetros ativos: {nonzero_params:,}')
    print(f'  Sparsidade:        {sparsity:.1f}%')


RESULTS = {}

def save_result(name, acc, size_mb, latency_ms, total_params, nonzero_params):
    RESULTS[name] = dict(
        accuracy=round(acc*100, 2),
        size_mb=round(size_mb, 2),
        latency_ms=round(latency_ms, 2),
        total_params=total_params,
        nonzero_params=nonzero_params,
        sparsity=round(100*(1-nonzero_params/total_params), 1),
    )

print('Utilitários carregados!')


# ══════════════════════════════════════════════════════════════════════════════
# 5. Carregar modelo base (Baseline)
# ══════════════════════════════════════════════════════════════════════════════
# LOCAL: apenas aponta para o arquivo — sem upload interativo do Colab
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
base_total, base_nz = count_nonzero_params(baseline_model)

print_metrics('BASELINE', base_acc, base_size, base_latency, base_total, base_nz)
save_result('Baseline', base_acc, base_size, base_latency, base_total, base_nz)

save_sparse(baseline_model, 'baseline.pth.gz')
print('Salvo: baseline.pth.gz')


# ══════════════════════════════════════════════════════════════════════════════
# 6. PRUNING
# ══════════════════════════════════════════════════════════════════════════════

# ── 6A. Unstructured Pruning ──────────────────────────────────────────────────
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


UNSTRUCTURED_AMOUNTS = [0.85, 0.9, 0.95]

for amount in UNSTRUCTURED_AMOUNTS:
    pruned = apply_unstructured_pruning(baseline_model, amount=amount)
    pruned = remove_pruning_masks(pruned)
    _, acc, _, _ = evaluate(pruned, test_loader)
    size_mb  = model_size_mb(pruned)
    latency  = measure_latency(pruned)
    tot, nz  = count_nonzero_params(pruned)
    label = f'Unstructured Pruning {int(amount*100)}%'
    print_metrics(label, acc, size_mb, latency, tot, nz)
    save_result(label, acc, size_mb, latency, tot, nz)
    fname = f'pruned_unstructured_{int(amount*100)}.pth.gz'
    save_sparse(pruned, fname)
    print(f'  Salvo: {fname}')


# ── 6B. Structured Pruning ────────────────────────────────────────────────────
def rebuild_structured_model(model):
    for name, module in model.named_modules():
        if hasattr(module, 'weight_orig'):
            prune.remove(module, 'weight')
    return model


def finetune(model, train_loader, val_loader, epochs, lr, device=DEVICE):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    model.train()
    for epoch in range(epochs):
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            criterion(model(imgs), lbls).backward()
            optimizer.step()
    return model


def apply_structured_pruning(model, amount=0.3):
    model = copy.deepcopy(model)
    safe_names = []
    for name, module in model.named_modules():
        if name == 'conv1' and isinstance(module, nn.Conv2d):
            safe_names.append(name)
        elif name.endswith('.conv1') and 'layer' in name and isinstance(module, nn.Conv2d):
            safe_names.append(name)
    return model, safe_names


def structured_pruning_with_finetune(baseline_model, amount, train_loader,
                                     val_loader, rounds=3, ft_epochs=2,
                                     lr=5e-5, device=DEVICE):
    model, safe_names = apply_structured_pruning(baseline_model, amount=0)
    model = model.to(device)
    per_round = 1 - (1 - amount) ** (1 / rounds)

    for r in range(rounds):
        print(f"\n  Round {r+1}/{rounds} — podando {per_round*100:.1f}% das camadas seguras")
        for name, module in model.named_modules():
            if name not in safe_names:
                continue
            n_filters = module.out_channels
            n_prune   = max(1, int(n_filters * per_round))
            n_prune   = min(n_prune, n_filters - 1)
            prune.ln_structured(module, name='weight', amount=n_prune, n=1, dim=0)

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        model.train()
        for epoch in range(ft_epochs):
            for imgs, lbls in train_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                optimizer.zero_grad()
                criterion(model(imgs), lbls).backward()
                optimizer.step()

        _, val_acc, _, _ = evaluate(model, val_loader, device)
        tot, nz = count_nonzero_params(model)
        print(f"  val_acc={val_acc*100:.2f}%  |  params ativos: {nz:,}")

    model = rebuild_structured_model(model)
    print(f"\n  Fine-tuning final (5 épocas) ...")
    model = finetune(model, train_loader, val_loader, epochs=5, lr=lr, device=device)
    return model


STRUCTURED_AMOUNTS = [0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

for amount in STRUCTURED_AMOUNTS:
    print(f"\n{'='*60}")
    print(f"  Structured Pruning {int(amount*100)}% (gradual + finetune intercalado)")
    print(f"{'='*60}")
    pruned = structured_pruning_with_finetune(
        baseline_model, amount, train_loader, val_loader,
        rounds=3, ft_epochs=2, lr=5e-5
    )
    _, acc, _, _ = evaluate(pruned, test_loader)
    size_mb  = model_size_mb(pruned)
    latency  = measure_latency(pruned)
    tot, nz  = count_nonzero_params(pruned)
    label = f'Structured Pruning {int(amount*100)}%'
    print_metrics(label, acc, size_mb, latency, tot, nz)
    save_result(label, acc, size_mb, latency, tot, nz)
    fname = f'pruned_structured_{int(amount*100)}.pth.gz'
    save_sparse(pruned, fname)
    print(f'  Salvo: {fname}')


# ── 6D. Iterative Pruning ─────────────────────────────────────────────────────
def iterative_pruning(model, target_amount=0.90, rounds=5, finetune_epochs=3,
                      lr=1e-4, device=DEVICE):
    model = copy.deepcopy(model).to(device)
    amounts = [1 - (1 - target_amount) ** ((r + 1) / rounds) for r in range(rounds)]
    for r, amount in enumerate(amounts):
        print(f"\n--- Round {r+1}/{rounds} | Sparsidade alvo: {amount*100:.1f}% ---")
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
                loss = criterion(model(imgs), lbls)
                loss.backward()
                optimizer.step()
        _, val_acc, _, _ = evaluate(model, val_loader, device)
        tot, nz = count_nonzero_params(model)
        print(f"  val_acc={val_acc*100:.2f}%  |  params ativos: {nz:,}")
    model = remove_pruning_masks(model)
    return model


ITERATIVE_TARGETS = [0.60]

for target in ITERATIVE_TARGETS:
    print(f"\n{'='*60}")
    print(f"  Pruning Iterativo — alvo {int(target*100)}%")
    print(f"{'='*60}")
    iter_model = iterative_pruning(baseline_model, target_amount=target,
                                   rounds=5, finetune_epochs=3)
    _, acc, _, _ = evaluate(iter_model, test_loader)
    size_mb  = model_size_mb(iter_model)
    latency  = measure_latency(iter_model)
    tot, nz  = count_nonzero_params(iter_model)
    label = f'Iterative Pruning {int(target*100)}%'
    print_metrics(label, acc, size_mb, latency, tot, nz)
    save_result(label, acc, size_mb, latency, tot, nz)
    fname = f'pruned_iterative_{int(target*100)}.pth.gz'
    save_sparse(iter_model, fname)
    print(f'  Salvo: {fname}')


# ── 6E. Unstructured Agressivo + Finetune ────────────────────────────────────
AGGRESSIVE_AMOUNTS = [0.95]

for amount in AGGRESSIVE_AMOUNTS:
    pruned = apply_unstructured_pruning(baseline_model, amount=amount)
    # IMPORTANTE: NÃO remover as máscaras antes do finetune.
    # As máscaras mantêm os pesos zerados fixos durante o treino (weight_orig é
    # atualizado, mas weight = weight_orig * mask — os zeros permanecem zeros).
    # Se removermos antes, o otimizador "ressuscita" os zeros e a sparsidade some.
    print(f"\nFine-tuning após Unstructured {int(amount*100)}% ...")
    pruned = finetune(pruned, train_loader, val_loader, epochs=5, lr=1e-4)
    pruned = remove_pruning_masks(pruned)   # consolida sparsidade DEPOIS do finetune
    _, acc, _, _ = evaluate(pruned, test_loader)
    size_mb  = model_size_mb(pruned)
    latency  = measure_latency(pruned)
    tot, nz  = count_nonzero_params(pruned)
    label = f'Unstructured Pruning {int(amount*100)}% + Finetune'
    print_metrics(label, acc, size_mb, latency, tot, nz)
    save_result(label, acc, size_mb, latency, tot, nz)
    fname = f'pruned_unstructured_{int(amount*100)}_finetuned.pth.gz'
    save_sparse(pruned, fname)
    print(f'  Salvo: {fname}')


# ── 6F. Structured 60% + Finetune ────────────────────────────────────────────
# Usa a mesma função do loop 6B (gradual + finetune intercalado).
# O apply_structured_pruning original apenas devolve (model, safe_names) sem
# aplicar nenhuma poda — por isso o modelo anterior saía com sparsidade 0%.
print("Aplicando Structured Pruning 60% + Fine-tuning ...")
pruned_struct_60 = structured_pruning_with_finetune(
    baseline_model, amount=0.60,
    train_loader=train_loader, val_loader=val_loader,
    rounds=3, ft_epochs=2, lr=5e-5
)

_, acc, _, _ = evaluate(pruned_struct_60, test_loader)
size_mb  = model_size_mb(pruned_struct_60)
latency  = measure_latency(pruned_struct_60)
tot, nz  = count_nonzero_params(pruned_struct_60)

label = 'Structured Pruning 60% + Finetune'
print_metrics(label, acc, size_mb, latency, tot, nz)
save_result(label, acc, size_mb, latency, tot, nz)
save_sparse(pruned_struct_60, 'pruned_structured_60_finetuned.pth.gz')
print('Salvo: pruned_structured_60_finetuned.pth.gz')


# ══════════════════════════════════════════════════════════════════════════════
# 7. QUANTIZATION
# ══════════════════════════════════════════════════════════════════════════════
CPU = torch.device('cpu')

# ── 7A. PTQ Dynamic ───────────────────────────────────────────────────────────
def ptq_dynamic(model):
    model_cpu = copy.deepcopy(model).to(CPU)
    model_cpu.eval()
    quantized = torch.quantization.quantize_dynamic(
        model_cpu,
        {nn.Linear, nn.Conv2d},
        dtype=torch.qint8
    )
    return quantized


print('Aplicando PTQ Dynamic ...')
ptq_model = ptq_dynamic(baseline_model)

_, acc, _, _ = evaluate(ptq_model, test_loader, device=CPU)
size_mb      = model_size_mb(ptq_model)
latency      = measure_latency(ptq_model, device=CPU)
tot, nz      = count_nonzero_params(baseline_model)

label = 'PTQ Dynamic (INT8)'
print_metrics(label, acc, size_mb, latency, tot, nz)
save_result(label, acc, size_mb, latency, tot, nz)

with gzip.open('ptq_dynamic_int8.pth.gz', 'wb', compresslevel=9) as f:
    pickle.dump(ptq_model.state_dict(), f, protocol=4)
print('Salvo: ptq_dynamic_int8.pth.gz')


# ── 7B. PTQ Static (incompatível com ResNet no PyTorch 2.x) ──────────────────
print('\nPTQ Static: incompatível com ResNet skip connections no PyTorch 2.6+')
print('Registrando como N/A...\n')
RESULTS['PTQ Static (INT8)'] = dict(
    accuracy=float('nan'),
    size_mb=float('nan'),
    latency_ms=float('nan'),
    total_params=0,
    nonzero_params=0,
    sparsity=float('nan'),
    note='Incompatível com ResNet no PyTorch 2.6+'
)


# ── 7C. QAT ──────────────────────────────────────────────────────────────────
def qat_train(model, train_loader, val_loader, epochs=5, lr=1e-4):
    model_cpu = copy.deepcopy(model).to(CPU)
    model_cpu.train()
    model_cpu.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    torch.quantization.prepare_qat(model_cpu, inplace=True)

    optimizer = optim.Adam(model_cpu.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    best_acc, best_state = 0.0, None

    for epoch in range(1, epochs + 1):
        model_cpu.train()
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(CPU), lbls.to(CPU)
            optimizer.zero_grad()
            loss = criterion(model_cpu(imgs), lbls)
            loss.backward()
            optimizer.step()
        scheduler.step()
        model_cpu.eval()
        _, val_acc, _, _ = evaluate(model_cpu, val_loader, device=CPU)
        print(f'  Epoch {epoch}/{epochs}  val_acc={val_acc*100:.2f}%')
        if val_acc > best_acc:
            best_acc   = val_acc
            best_state = copy.deepcopy(model_cpu.state_dict())

    model_cpu.load_state_dict(best_state)
    model_cpu.eval()
    torch.quantization.convert(model_cpu, inplace=True)
    return model_cpu


print('Aplicando QAT (pode demorar bastante em CPU sem GPU) ...')
qat_model = qat_train(baseline_model, train_loader, val_loader, epochs=5)

_, acc, _, _ = evaluate(qat_model, test_loader, device=CPU)
size_mb      = model_size_mb(qat_model)
latency      = measure_latency(qat_model, device=CPU)
tot, nz      = count_nonzero_params(baseline_model)

label = 'QAT (INT8)'
print_metrics(label, acc, size_mb, latency, tot, nz)
save_result(label, acc, size_mb, latency, tot, nz)

with gzip.open('qat_int8.pth.gz', 'wb', compresslevel=9) as f:
    pickle.dump(qat_model.state_dict(), f, protocol=4)
print('Salvo: qat_int8.pth.gz')


# ── 7D. Combo: Unstructured 70% + PTQ Dynamic ────────────────────────────────
print("Combo: Unstructured 70% + PTQ Dynamic (INT8) ...")
pruned_70 = apply_unstructured_pruning(baseline_model, amount=0.70)
pruned_70 = remove_pruning_masks(pruned_70)
pruned_70 = finetune(pruned_70, train_loader, val_loader, epochs=3, lr=1e-4)
combo_ptq = ptq_dynamic(pruned_70)

_, acc, _, _ = evaluate(combo_ptq, test_loader, device=CPU)
size_mb  = model_size_mb(combo_ptq)
latency  = measure_latency(combo_ptq, device=CPU)
tot, nz  = count_nonzero_params(pruned_70)

label = 'Unstructured 70% + PTQ INT8'
print_metrics(label, acc, size_mb, latency, tot, nz)
save_result(label, acc, size_mb, latency, tot, nz)

with gzip.open('combo_unstruct70_ptq.pth.gz', 'wb', compresslevel=9) as f:
    pickle.dump(combo_ptq.state_dict(), f, protocol=4)
print('Salvo: combo_unstruct70_ptq.pth.gz')


# ══════════════════════════════════════════════════════════════════════════════
# 8. Tabela comparativa de resultados
# ══════════════════════════════════════════════════════════════════════════════
import pandas as pd

df = pd.DataFrame(RESULTS).T.reset_index()
df.columns = ['Modelo', 'Acurácia (%)', 'Tamanho (MB)', 'Latência (ms)',
               'Params Totais', 'Params Ativos', 'Sparsidade (%)', 'Observações']
df = df.sort_values('Acurácia (%)', ascending=False).reset_index(drop=True)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 120)
print(df.to_string(index=False))

df.to_csv('benchmark_results.csv', index=False)
print('\nSalvo: benchmark_results.csv')


# ══════════════════════════════════════════════════════════════════════════════
# 9. Curva de Pareto — Acurácia vs. Tamanho do Modelo
# ══════════════════════════════════════════════════════════════════════════════
try:
    from adjustText import adjust_text
    HAS_ADJUST = True
except ImportError:
    HAS_ADJUST = False
    print('adjustText não instalado — labels podem sobrepor. Instale com: pip install adjustText')

df2 = pd.DataFrame(RESULTS).T.reset_index()
df2.columns = ['Modelo', 'Acurácia (%)', 'Tamanho (MB)', 'Latência (ms)',
               'Params Totais', 'Params Ativos', 'Sparsidade (%)', 'Observações']
for col in ['Acurácia (%)', 'Sparsidade (%)', 'Latência (ms)', 'Tamanho (MB)']:
    df2[col] = pd.to_numeric(df2[col], errors='coerce')
df2 = df2.dropna(subset=['Acurácia (%)'])

def categorize(nome):
    if 'Baseline'    in nome: return 'Baseline'
    if 'Iterative'   in nome: return 'Iterativo'
    if 'Unstructured' in nome and '+' not in nome: return 'Unstructured'
    if 'Structured'   in nome and '+' not in nome: return 'Structured'
    if any(x in nome for x in ['PTQ','QAT','INT8']): return 'Quantização'
    if '+' in nome: return 'Combo'
    return 'Outro'

CORES = {
    'Baseline':    '#5F5E5A',
    'Unstructured':'#378ADD',
    'Iterativo':   '#7F77DD',
    'Structured':  '#1D9E75',
    'Quantização': '#BA7517',
    'Combo':       '#D85A30',
    'Outro':       '#888780',
}
MARKERS = {
    'Baseline': 'D', 'Unstructured': 'o', 'Iterativo': 's',
    'Structured': '^', 'Quantização': 'P', 'Combo': '*', 'Outro': 'X'
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
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['left','bottom']].set_color('#DDDDDD')
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
                   s=160, zorder=5,
                   edgecolors='white', linewidths=0.8, alpha=0.92)
    for _, row in df2.iterrows():
        if pd.isna(row['Acurácia (%)']) or pd.isna(row[x_col]):
            continue
        nome = (row['Modelo']
                .replace(' + Finetune', ' +FT')
                .replace(' (INT8)', '')
                .replace('Pruning ', ''))
        t = ax.text(row[x_col], row['Acurácia (%)'], nome,
                    fontsize=8.5, color='#2C2C2A', alpha=0.9,
                    va='bottom', ha='left')
        texts.append(t)
    if HAS_ADJUST:
        adjust_text(texts, ax=ax,
                    arrowprops=dict(arrowstyle='-', color='#AAAAAA', lw=0.6),
                    expand_points=(1.8, 2.2),
                    force_points=(0.4, 0.6),
                    force_text=(0.5, 0.8))
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
ref_95 = mlines.Line2D([],[], color='#E24B4A', linestyle=':', linewidth=1.4, label='−5% acurácia baseline')
ref_90 = mlines.Line2D([],[], color='#D85A30', linestyle=':', linewidth=1.4, label='−10% acurácia baseline')

fig.legend(handles=legend_handles + [ref_95, ref_90],
           loc='lower center', ncol=6, fontsize=9.5,
           frameon=True, framealpha=0.95, edgecolor='#DDDDDD',
           bbox_to_anchor=(0.5, -0.05),
           handletextpad=0.5, columnspacing=1.2)

plt.suptitle('Pareto — ResNet-18 × PlantVillage (15 classes)',
             fontsize=14, fontweight='medium', y=1.02, color='#2C2C2A')
plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig('pareto_curves.png', dpi=180, bbox_inches='tight', facecolor='#FAFAFA')
plt.close()
print('Salvo: pareto_curves.png')

print('\n✅ Benchmark concluído! Arquivos gerados nesta pasta:')
for f in glob.glob('*.pth.gz') + ['benchmark_results.csv', 'pareto_curves.png']:
    if os.path.exists(f):
        print(f'  {f}')