# TODO e Observações para o Paper

Documento vivo com cuidados metodológicos, achados que merecem destaque e
itens em aberto para a escrita do paper de knowledge distillation
(resnet18 → alunos compactos, classificação de doenças em folhas).

---

## 1. Achados que merecem destaque na seção de Resultados/Discussão

### 1.1 Inversão de latência entre batch=1 e batch=32 (achado central)

Os alunos com convoluções **depthwise-separable** (mnasnet0_5,
mobilenet_v3_small, shufflenet_v2_x0_5, shufflenet_v2_x1_0) são
**2–3× MAIS LENTOS** que o baseline resnet18 em batch=1, mas se tornam
**~2× MAIS RÁPIDOS** em batch=32. O único aluno sem depthwise convs
(squeezenet1_1, baseado em "fire modules" de convs 1×1 e 3×3 densas)
é o único que já é mais rápido que o baseline em batch=1.

| Aluno | Latência b=1 (% do baseline) | Latência b=32 (% do baseline) |
|---|---|---|
| mnasnet0_5         | 194.7% | 60.7% |
| mobilenet_v3_small | 268.6% | 51.8% |
| shufflenet_v2_x0_5 | 270.8% | 33.3% |
| shufflenet_v2_x1_0 | 312.4% | 56.1% |
| squeezenet1_1      |  80.6% | 55.1% |

**Tese a defender no paper:** compressão de parâmetros e redução de MACs
**não implicam aceleração em hardware**. A métrica correta para destilação
voltada a *edge deployment* (cenário com batch=1) é **latência empírica
medida**, não params/MACs. Isso é coerente com a literatura
(ShuffleNetV2, MobileNetV3) sobre intensidade aritmética / memory-bound.

### 1.2 Limítrofe da técnica

- **Vencedor claro:** `shufflenet_v2_x0_5` — retém 99.0% do top-1 com
  31× de compressão e 2.2% dos MACs. Está na borda da zona "perda ≤ 1pp".
- **Caso de falha:** `mnasnet0_5` — único modelo *fora* da zona aceitável
  (~6.3 pp de gap). Sugerir investigação: arquitetura inadequada para
  destilação neste dataset, ou hiperparâmetros sub-ótimos para esse aluno
  especificamente.

### 1.3 Padrão de erro do mnasnet0_5

O mnasnet0_5 não erra uniformemente — ele *quebra* em classes específicas:
- `Tomato Early blight`: -30 pp de F1 vs baseline
- `Potato healthy`: -26 pp de F1
- `Tomato Target Spot`: -13 pp de F1

Outros alunos têm perdas modestas (~5 pp no pior caso) e relativamente
uniformes. Isso sugere que a **falha do mnasnet é localizada**, não
sistêmica — possível conexão com classes de baixo *support* ou
confusões inter-doenças. Vale uma análise de matriz de confusão na
discussão.

---

## 2. Cuidados metodológicos a declarar no paper

### 2.1 Benchmark de latência

A implementação em [evaluate_model.py:170-189](evaluate_model.py#L170-L189)
está correta nos pontos críticos, mas as escolhas devem ser **declaradas
explicitamente** na seção de Methodology para reprodutibilidade:

- **Warmup:** 10 iterações descartadas antes da medição (mitiga
  compilação JIT de kernel e *cache priming*, que tornam as primeiras
  inferências 5–50× mais lentas).
- **Medição:** 50 forward passes cronometrados com `time.perf_counter()`.
- **Sync point:** `float(y.flatten()[0])` força device→host copy de
  1 escalar — garante que o tempo medido inclui execução real do kernel
  em CUDA/DirectML (sem isso, mediria apenas o *enqueue* na fila).
- **Input:** `torch.randn(batch, 3, IMG_SIZE, IMG_SIZE)` — tensor
  sintético no device. Mede *compute puro*, exclui custo de I/O e
  preprocessing. **Declarar isso no paper.**
- **Hardware:** declarar device exato (DirectML / GPU / CPU), versão de
  PyTorch, versão de torch-directml, OS. Resultados de latência são
  *altamente* dependentes de plataforma.

### 2.2 Limitações de medição que devem ser anotadas

- **Overhead constante do sync.** O `float(y[0,0])` introduz overhead de
  ~50-200 µs por medição no Windows/DML (memcpy + sync). Para um modelo
  que roda em ~1.3 ms, isso infla a métrica absoluta em ~5-15%. Como
  é **constante para todos os modelos**, as comparações relativas
  permanecem válidas. Mencionar como *systematic measurement overhead*.
- **Variância de sistema.** Windows não é determinístico; std observado
  em alguns alunos chega a 33% da média (ex.: shufflenet_v2_x0_5 batch
  1: 3.65 ± 0.91 ms). Isso é alto. Considerar reportar **mediana (P50)
  ou IQR** em vez de média ± std no paper.
- **`torch.backends.cudnn.benchmark`** não se aplica em DirectML, mas
  se algum experimento for re-rodado em CUDA real, ativar e declarar.

### 2.3 Métricas: o que reportar e por quê

Para deixar a metodologia defensável na revisão:

| Métrica | Por quê |
|---|---|
| top-1, F1 macro, F1 weighted | F1 macro é crítico — o dataset é desbalanceado (`Potato___healthy` tem support=23 vs `Tomato__Tomato_YellowLeaf__Curl_Virus` com 481). Top-1 sozinho engana. |
| F1 por classe (delta vs baseline) | Mostra **onde** a destilação falha, não só *quanto*. Achado científico, não cosmético. |
| Params, MACs | Métricas teóricas — comparáveis entre estudos, independentes de hardware. |
| Latência b=1 **e** b=32, mediana e IQR | Ambas são necessárias: b=1 = cenário edge, b=32 = cenário servidor. A inversão entre elas é parte do achado. |
| Throughput em b=32 | Para complementar latência b=32 quando a inferência é em lote. |
| Fator de compressão (baseline/aluno) | Sumário do "ganho" em uma única dimensão. |
| Gap de top-1 em pp (não em %) | Pontos percentuais são mais legíveis e menos enganosos que razões em precisão >90%. |

---

## 3. Verificações / experimentos pendentes

### 3.1 Reprodutibilidade do benchmark de latência

**TODO:** Rodar `evaluate_model.py` duas vezes seguidas com `--runs 200`
em pelo menos 1 aluno para verificar se a inversão b=1 ↔ b=32 reproduz.
Se a magnitude variar >10% entre execuções, aumentar `--runs` ou
mudar para mediana.

```powershell
python evaluate_model.py --target ... --runs 200 --save-json output/students/eval_X_run1.json
python evaluate_model.py --target ... --runs 200 --save-json output/students/eval_X_run2.json
```

### 3.2 Análise da falha do mnasnet0_5

**TODO:** Gerar matriz de confusão para mnasnet0_5 vs baseline. Verificar
se as classes onde ele falha (`Tomato Early blight`, `Potato healthy`)
estão sendo confundidas com classes específicas — pode revelar se o
problema é *baixo support* ou *similaridade visual entre doenças*.

### 3.3 Ablation sobre temperatura / coeficiente de destilação

**TODO:** O mnasnet0_5 falhar enquanto modelos com menos parâmetros
(shufflenet_v2_x0_5) têm sucesso sugere que o problema não é
*capacidade*. Vale uma ablation:
- Mesma destilação com `T ∈ {1, 4, 8}` e `α ∈ {0.3, 0.5, 0.7}` para
  mnasnet, para verificar se hiperparâmetros melhores fechariam o gap.
- Se o gap fechar → conclusão é "destilação requer tuning por aluno".
- Se não fechar → conclusão é "mnasnet0_5 tem inadequação arquitetural
  para este dataset".

### 3.4 Padronizar baseline em todos os JSONs

**TODO:** Confirmar que todos os 5 eval_*.json usam exatamente o mesmo
arquivo de baseline (`output/best_model.pth`). O script `plot_evals.py`
já verifica isso e aborta se diferentes, mas vale checar manualmente
antes de gerar as figuras finais.

### 3.5 Quantização int8 — alegação vs medição

**TODO:** Os valores `int8_size_mb` nos JSONs são **estimativas teóricas**
(params × 1 byte). Para alegar redução real de tamanho/latência por
quantização no paper, é necessário *quantizar de fato* (PTQ ou QAT,
via `torch.quantization` ou ONNX Runtime) e medir top-1 e latência do
modelo quantizado. Se não for feito, **remover** as colunas int8 dos
gráficos para evitar implicação enganosa.

---

## 4. Itens de escrita / apresentação

### 4.1 Estilo das figuras

As 4 figuras geradas por [plot_evals.py](plot_evals.py) já estão em
estilo acadêmico sóbrio (serif, paleta neutra, sem decoração):

- `retention.png` — panel-of-truth do trade-off por aluno.
- `compression_vs_loss.png` — figura central da Seção de Resultados.
- `pareto.png` — fronteira qualidade × custo em 3 dimensões.
- `per_class_delta.png` — figura de discussão sobre falha localizada.
- `summary.csv` — tabela mestre para `\begin{table}` em LaTeX.

### 4.2 Estrutura sugerida da seção de Resultados

1. **Tabela mestre** (do `summary.csv`) — visão completa numérica.
2. **Figura `retention.png`** — primeira figura, mostra o trade-off
   por aluno em painéis separados.
3. **Figura `compression_vs_loss.png`** — define o "limítrofe" da técnica.
4. **Figura `pareto.png`** — análise multi-dimensional do custo.
5. **Figura `per_class_delta.png`** — análise qualitativa de falha.

### 4.3 Tom da discussão

Manter cauteloso e defensável:
- ✅ "shufflenet_v2_x0_5 retém 99% do top-1 com 31× de compressão"
- ✅ "a aceleração esperada da destilação se manifesta apenas em batch grande"
- ❌ "destilação é sempre melhor" (não é — depende do hardware e do batch)
- ❌ "mnasnet0_5 é uma arquitetura ruim" (pode ser tuning)

---

## 5. Dependências / ambiente (anotar em Appendix)

- Python 3.x (verificar versão exata)
- PyTorch com backend DirectML (verificar versão)
- `.venv-gpu` (DirectML); presença de `AdamWDML` custom em
  [train.py:167-195](train.py#L167-L195) para evitar fallback CPU do
  `lerp` no DirectML — **mencionar isso no Methodology** se latência de
  treino for discutida; não afeta inferência.
- matplotlib, numpy, sklearn

---

**Última atualização:** 2026-05-17
