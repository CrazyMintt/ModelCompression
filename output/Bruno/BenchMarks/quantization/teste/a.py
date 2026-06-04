import gzip, pickle, torch

with gzip.open("/home/bruno/Downloads/ModelCompressionResearchCopia/output/Bruno/Modelos/quantization/ptq_static/ptq_static_fx_int8.pth.gz", "rb") as f:
    p = pickle.load(f)

print("=== TIPO DO PAYLOAD ===")
print(type(p))
print()
print("=== PRIMEIRAS 30 CHAVES ===")
for k, v in list(p.items())[:30]:
    t = type(v).__name__
    q = getattr(v, 'is_quantized', False) if isinstance(v, torch.Tensor) else '-'
    print(f"  {k:<60} {t:<25} quantized={q}")

print("\n=== CHAVES DO FC ===")
for k, v in p.items():
    if 'fc' in k:
        t = type(v).__name__
        q = getattr(v, 'is_quantized', False) if isinstance(v, torch.Tensor) else '-'
        print(f"  {k:<60} {t:<25} quantized={q}")