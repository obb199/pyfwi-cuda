# PyFWI-CUDA: roteiro de execução e verificação

25 out. 2026 · Gabriel Carvalho · repositório privado [obb199/pyfwi-cuda](https://github.com/obb199/pyfwi-cuda)

Este roteiro transforma a [análise e plano de migração](analise-e-plano.md) em uma sequência de passos executáveis, do ambiente pronto até a versão 0.4 e a manutenção. Cada passo diz o que entra no código, quais testes o acompanham e qual critério precisa ser cumprido para seguir adiante. As decisões técnicas (CuPy como base, fonte única na paridade, modos `legacy` e `fixed`, API congelada) vêm da análise e não são rediscutidas aqui.

## Sequência resumida

| # | Passo | Versão / gate |
| --- | --- | --- |
| P | Ambientes, repositório e verificador de ambiente | **concluído** (`babbc0e`) |
| 0.1 | Higiene do fork: README, aviso GPLv3, CHANGELOG, CI de CPU | Fase 0 |
| 0.2 | Bugs conhecidos da 0.1.10 viram testes `xfail` estritos | |
| 0.3 | Gerador de golden master no ambiente de referência | |
| 0.4 | Testes de caracterização no dev e diagnóstico cruzado ref × dev | |
| 0.5 | Linha de base matemática da 0.1.10 (gradiente, analítica, convergência, adjunto) | |
| 0.6 | Benchmark de referência nos três cenários | |
| 0.7 | Bancada de profiling: Nsight Systems, Nsight Compute, compute-sanitizer | |
| 0.8 | Runner self-hosted na RTX e jobs de GPU | **G0** |
| 1.1 | Imports e empacotamento (SciPy e NumPy atuais, dependências declaradas) | Fase 1 |
| 1.2 | Seleção de dispositivo OpenCL corrigida | |
| 1.3 | Kernels em cache e argumentos fixados | |
| 1.4 | Resíduo e wavelet na GPU; cópias não bloqueantes | |
| 1.5 | Checkpoints contíguos; `W.W` como visão | |
| 1.6 | Filtro passa-baixa elemento a elemento | |
| 1.7 | Buffers com margem: fim das leituras fora da alocação | |
| 1.8 | `PyFWI.info()` e log de diagnóstico | |
| 1.9 | Release 0.1.11 do fork | **G1** |
| 2.1 | Motor e interface de backend; OpenCL atrás da interface | Fase 2 |
| 2.2 | Seletor de backend e fallback | |
| 2.3 | Backend CUDA de paridade (fonte única, modo `legacy`) | |
| 2.4 | Modos numéricos `legacy` e `fixed` | |
| 2.5 | Contrato da API no backend CUDA | |
| 2.6 | Extras `[cuda13]` e `[cuda12]` | |
| 2.7 | Relatório de paridade e release 0.2.0 | **G2** |
| 3.1–3.10 | Otimizações CUDA, uma por commit, com benchmark antes e depois | Fase 3 |
| 3.11 | Padrão `auto` e release 0.3.0 | **G3** |
| 4.1–4.6 | Função custo na GPU, boundary saving, FP16, multi-GPU, DLPack, validação geofísica | Fase 4 |
| 4.7 | Release 0.4.0 | **G4** |
| 5.x | Documentação final, upstream, publicação e otimizações condicionais | Manutenção |

Duração estimada na análise: cerca de 20 semanas com uma pessoa (Fase 0 ≈ 3, Fase 1 ≈ 3, Fase 2 ≈ 4, Fase 3 ≈ 6, Fase 4 ≈ 4) ou cerca de 14 com duas, porque a Fase 4 pode correr em paralelo com a Fase 3 depois do G2.

## Regras de trabalho

**Um passo, um ou mais commits pequenos.** Cada commit traz uma feature, os testes que a provam e uma mensagem que explica o que mudou e por quê. Antes de cada commit:

1. `ruff check` nos arquivos alterados (o código herdado só é corrigido quando for tocado).
2. Suíte rápida no ambiente dev: `../envs/dev/bin/python -m pytest -m "not slow"`.
3. Se o commit toca kernels, backends ou o laço temporal: também `-m opencl`, `-m gpu` e o `compute-sanitizer` dos kernels alterados.
4. Se o commit é de desempenho: benchmark antes e depois, com os números na mensagem do commit.
5. Commit e push para `origin/main`.

**Gates.** Ao fim de cada fase, um gate reúne as verificações que não rodam a cada commit (matriz completa de golden master, benchmarks, profiling, FWI completa). O resultado é gravado em `docs/migracao-cuda/gates/G<n>.md`, com números, versões e hashes. A fase seguinte só começa com o gate aprovado.

**Bugs conhecidos.** Cada bug herdado da 0.1.10 vira um teste `xfail(strict=True)`. Quando a correção entra, o teste passa a passar, o `strict` o derruba e obriga a atualizar o teste no mesmo commit.

**Ambientes.** Criados por `tools/setup_envs.sh` e verificados por `tools/check_env.py`, ambos já no repositório.

| Ambiente | Conteúdo | Uso |
| --- | --- | --- |
| `envs/ref-0.1.10` | Python 3.10, PyFWI 0.1.10 do PyPI, NumPy 1.21.6, SciPy 1.7.3, PyOpenCL 2022.2.4, POCL 3.0 | Gerar golden masters e a linha de base; nunca recebe código do fork |
| `envs/dev` | Python 3.12, fork editável, CuPy 14.2 (CUDA 13), cuda.core 1.2, PyOpenCL 2026.1, POCL 3.0, pytest-benchmark | Desenvolvimento e testes |
| `envs/cuda12` (Fase 2) | Igual ao dev com `cupy-cuda12x` | Testar o extra `[cuda12]` |
| `envs/torch` (Fase 4) | Dev + PyTorch com CUDA | Testar DLPack sem misturar as rodas `nvidia-*` do CuPy e do PyTorch |

## Estado atual: etapa preparatória concluída

Máquina-alvo: GeForce RTX 3060 12 GB (Ampere, sm_86), driver 580.95.05 (CUDA 13.0), Ryzen 7 5700X, 23 GB de RAM, Debian 12. CUDA Toolkit 12.4 instalado no sistema com `nvcc`, Nsight Systems 2023.4, Nsight Compute 2024.1 e compute-sanitizer 2024.1.

O que o commit `babbc0e` entregou e verificou:

- Fork local com `upstream` = AmirMardan/PyFWI (push bloqueado) e `origin` = repositório privado. Base: `v0.1.10` = `1434a16`.
- Os dois ambientes, com versões travadas em `requirements/`. Recriados do zero numa pasta temporária, ficaram idênticos aos lockfiles.
- `check_env.py` aprovado nos dois ambientes: NVRTC gera cubin, CUDA Graph captura e lança, cuFFT tem erro relativo de 3×10⁻⁷ contra o NumPy, `cupyx` tem erro máximo de 1,2×10⁻⁷ contra o SciPy, e a 0.1.10 roda forward + gradiente na RTX de forma determinística bit a bit.
- 14 testes no dev e 12 no ambiente de referência (os 2 de GPU são pulados lá).

**Achados que a análise não registrava** e que entram no roteiro:

| Achado | Consequência | Onde é tratado |
| --- | --- | --- |
| No POCL, a 0.1.10 devolve 99% de NaN no sismograma e resultados diferentes a cada execução. Com zeros depois do fim dos buffers, o sismograma fica finito e determinístico; o gradiente ainda tem NaN, coerente com as leituras antes do início (`i − 3` com `i = 2`). | O POCL não serve de referência para a 0.1.10 como está. Os golden masters vêm do OpenCL NVIDIA, e o CI em CPU só compara saídas depois do passo 1.7. | 0.2, 0.3, 1.7 |
| `segyio` é importado no topo de `model_dataset.py` e `seismic_io.py`, mas não está em `install_requires` e aparece comentado no `requirements.txt`. | `pip install PyFWI` não basta nem para gerar um modelo. | 1.1 |
| `from numpy.lib.function_base import kaiser` quebra com NumPy 2. | `model_dataset` e `seismic_io` não importam no ambiente dev. | 1.1 |
| `inpa['device']` falha quando a plataforma tem um único dispositivo: o `create_some_context` acusa uma escolha "left over". | Hoje é preciso escolher o dispositivo por `PYOPENCL_CTX`. | 0.2, 1.2 |
| O mesmo forward leva 1,42 s com PyOpenCL 2022.2.4 e 2,07 s com PyOpenCL 2026.1.4 (provável `RepeatedKernelRetrieval`). | O benchmark de referência precisa registrar as duas versões. O passo 1.3 deve eliminar essa diferença. | 0.6, 1.3 |
| O NVRTC 13.4 instalado pelo pip é mais novo que o driver (13.0): o PTX gerado não carrega, só o cubin. | O backend CUDA sempre compila para cubin da arquitetura real; um teste garante isso. O driver 580 é mantido de propósito, por ser o mínimo suportado. | 2.3, 2.6 |
| O PTX dos kernels quentes tem de 43 (`update_velx`) a 138 (`Grad`) instruções FP64 na ordem 8 (a análise mediu de 42 a 131). | Confirma a guarda de precisão; o teste já existe e inverte o sinal no modo `fixed`. | 2.4 |
| `RmProfilingAdminOnly: 1`: o Nsight Compute não lê contadores sem root. | Depende de você (ver [Pendências](#pendências-que-dependem-de-você)). | 0.7 |
| Os workflows herdados (`pr.yml`, `publish.yml`) testam Python 3.8 a 3.10 em três sistemas com as versões antigas e publicariam no PyPI como `PyFWI`. | Precisam ser trocados antes de qualquer PR ou release. | 0.1 |

## Catálogo de testes e verificações

Cada passo cita os códigos abaixo. "A cada commit" significa rodar localmente antes do commit e no CI depois do push.

| Código | O que verifica | Como | Onde roda | Quando | Critério |
| --- | --- | --- | --- | --- | --- |
| LNT | Estilo e erros estáticos | `ruff` no código novo; `flake8 --select=E9,F63,F7,F82` no repositório inteiro, como no CI original | CI CPU | A cada commit | Zero erros |
| UNI | Funções isoladas: geometria, wavelet, filtros, layout, seletor, parsing | `pytest`, sem dispositivo | CI CPU | A cada commit | Tudo passa |
| CAR | Saídas iguais às da 0.1.10 (golden master) | `.npz` gerados no ambiente de referência; comparação no dev | Local e runner GPU | Subconjunto a cada commit; matriz completa no gate | Por fase (ver [Tolerâncias](#tolerâncias)) |
| BIT | Igualdade bit a bit das saídas brutas do propagador | `np.array_equal` em sismogramas e gradientes antes do pós-processamento no host | Local e runner GPU | A cada commit da Fase 1 e nas refatorações | Idêntico |
| DET | Duas execuções iguais dão o mesmo resultado | Mesmo caso duas vezes no mesmo dispositivo | Local e runner GPU | A cada commit que toca kernels | Idêntico; sem `atomicAdd` em ponto flutuante |
| PAR | Backend CUDA de paridade × OpenCL na mesma GPU | Matriz de casos do golden, modo `legacy` | Runner GPU | A cada commit da Fase 2; matriz completa no gate | Erro relativo L2 ≤ 1e-6 |
| OTM | Kernel otimizado (`fixed`) × CUDA de paridade (`fixed`) | Mesma matriz | Runner GPU | A cada commit da Fase 3 | Sismogramas ≤ 1e-5 e gradientes ≤ 1e-4 (L2 relativo) |
| MAT | Corretude matemática | Gradiente × diferenças finitas centradas; meio homogêneo × solução analítica; ordem de convergência; produto escalar do adjunto; reflexão da PML | Local e runner GPU | Linha de base no 0.5; de novo em cada gate | Ver passo 0.5 |
| PRE | Sem FP64 involuntário | NVRTC compila os kernels e o teste conta `.f64` no PTX; não precisa de GPU | CI CPU | A cada commit que toca kernels | Modo `fixed`: zero FP64 nos kernels quentes |
| SAN | Memória e concorrência na GPU | `compute-sanitizer` com `memcheck`, `racecheck`, `initcheck` e `synccheck` num modelo pequeno | Runner GPU | A cada commit que toca kernels CUDA | Zero erros |
| API | Contrato público congelado | Assinaturas, chaves do `inpa`, tipos, `dtype`, formas, ordem de eixos, `W.W`, `W.seismogram`, `show=True`; scripts de `example/` rodando sem edição em versão reduzida | CI CPU (POCL, a partir do 1.7) e runner GPU | A cada commit | Mesmo comportamento da 0.1.10 |
| SEL | Escolha de backend e fallback | Precedência `inpa['backend']` → `PYFWI_BACKEND` → padrão; `auto` sem CuPy, sem GPU ou com driver antigo (simulados com monkeypatch); `cuda:strict`; casamento por PCI bus ID; mensagem de log | CI CPU e runner GPU | A cada commit do seletor | Todos os ramos cobertos |
| PKG | Empacotamento | Build de wheel e sdist; instalação num venv limpo; presença de `.cl`, `.cu`, `data/` e `torch/`; extras; import de todos os módulos; `PyFWI.info()` | CI CPU | A cada commit de empacotamento; sempre antes de release | Instala e importa em Python 3.10 a 3.13 |
| DES | Desempenho | `pytest-benchmark` nos três cenários; mediana de 5 execuções após aquecimento; registra forward, gradiente, iteração de FWI, tempo de host por passo, lançamentos e cópias por passo, memória de pico | Runner GPU | Antes e depois de cada commit de desempenho; noturno | Sem regressão acima de 5%; metas da fase |
| PRF | Onde está o tempo | Nsight Systems (ocupação da GPU, intervalos ociosos, número de lançamentos e cópias); Nsight Compute (banda atingida por kernel); eventos de profiling no OpenCL | Manual, na RTX | Em cada gate e quando um benchmark surpreender | Relatório anexado ao gate |
| MEM | Memória | Pico na GPU e no host; política de checkpoints sob orçamento forçado | Runner GPU | Commits de memória; gates | Dentro da estimativa; resultado igual com qualquer orçamento |
| FWI | Inversão ponta a ponta | FWI multiescala e TL-FWI em modelos pequenos; curva de misfit e erro do modelo; revisão do geofísico | Runner GPU | Gates G1, G2 e G4 | Ver [Tolerâncias](#tolerâncias) |
| INT | Interoperabilidade com PyTorch | `torchfwi.Fwi` com tensores na CPU e na GPU; DLPack sem cópia (mesmo `data_ptr`); gradiente × diferenças finitas | Runner GPU | Commits do `torchfwi`; G4 | Igual ao caminho NumPy |
| EST | Estabilidade e limites | Execuções longas, modelos perto do limite de VRAM, muitos tiros; erros claros em falta de memória | Runner GPU | Gates G3 e G4 | Sem NaN, sem travamento, mensagem útil |
| DOC | Documentação | Links relativos válidos; exemplos e notebooks executados em versão reduzida | CI CPU | A cada commit de documentação; release | Tudo resolve e executa |

### Tolerâncias

| Comparação | Critério | Origem |
| --- | --- | --- |
| Fase 1 × 0.1.10, saídas brutas do propagador no mesmo dispositivo e ambiente | Idênticas bit a bit | Análise |
| Fase 1 × 0.1.10, pós-processamento no host com NumPy e SciPy de versões diferentes | ≤ 4 ulp em float32 | Novo: depende do resultado do passo 0.4 |
| Filtro passa-baixa novo × antigo (float64) | Diferença relativa ≤ 1e-12 | Análise |
| CUDA `legacy` × OpenCL `legacy` | L2 relativo ≤ 1e-6 (meta inicial, confirmada no G2) | Análise |
| CUDA otimizado × CUDA de paridade (`fixed`) | Sismogramas ≤ 1e-5; gradientes ≤ 1e-4 | Análise |
| `fixed` × `legacy` | Medida e publicada no CHANGELOG, sem limite fixo | Análise |
| Produto escalar do adjunto | Desvio relativo ≤ 1e-5 nos kernels novos; na 0.1.10, só medido | Análise; passo 0.5 |
| FWI (2 iterações) entre backends | Modelo final com L2 relativo ≤ 1e-4; misfit por iteração ≤ 1e-4 | Novo |
| FP16 | Limites próprios, validados por FWI completa: erro do modelo final no máximo 10% acima do FP32 (meta a confirmar no 4.3) | Análise, com meta proposta |
| Desempenho entre commits | Regressão ≤ 5% | Análise |

## Fase 0: rede de segurança e linha de base

Nenhuma mudança no comportamento do pacote. Aqui se constrói tudo o que vai provar que as fases seguintes não quebraram nada.

### 0.1 Higiene do fork

- **Entra:** README do fork (o que é, como instalar, origem e citação dos autores originais), aviso de modificação exigido pela GPLv3, `CHANGELOG.md`, e `.github/workflows/ci.yml` com um job de CPU (LNT, UNI, PKG, DOC e, mais tarde, PRE). `pr.yml` e `publish.yml` são removidos, porque testam versões que o fork não suporta e publicariam com o nome do original.
- **Testes:** LNT; DOC (teste que valida os links relativos dos `.md`); o CI precisa ficar verde no primeiro push.
- **Critério:** CI verde; nenhum workflow capaz de publicar no PyPI.

### 0.2 Bugs conhecidos como testes

- **Entra:** `test/test_known_bugs.py`, com um `xfail(strict=True)` por bug: leitura fora do buffer (rodando no POCL: saída finita), `vz` gravando `vx` em `elastic_crosswell.cl`, `dt` truncado no `#define`, FP64 nos kernels quentes, `inpa['device']` com dispositivo único, imports quebrados com SciPy e NumPy atuais e `segyio` não declarado.
- **Testes:** os próprios testes; cada um precisa falhar pelo motivo certo (o teste confere a mensagem ou o valor, não só que algo falhou).
- **Critério:** todos em `xfail`; nenhum `xpass`.

### 0.3 Gerador de golden master

- **Entra:** `test/golden/generate.py`, rodado só no `envs/ref-0.1.10`, sobre OpenCL NVIDIA. Ele grava `.npz` com:
  - forward (sismogramas das 5 componentes) e gradiente, incluindo as saídas brutas, capturadas antes de `grad_lmd_to_vd` e do `gaussian_filter`;
  - 2 iterações de FWI (L-BFGS e CG), uma rodada de cada estratégia de `TimeLapse` e um forward/backward do `torchfwi.Fwi`;
  - uma matriz combinatória em pares (pairwise), não o produto completo: `acq_type` 0/1/2, `components` 0 a 4, ordem 4/8, `energy_balancing`, `chpr` 0/20/100, parametrização `dv` e λ/μ/ρ, num modelo de 60×60 com 300 passos;
  - metadados por arquivo: versões, GPU, driver, commit, `inpa` e hash SHA-256 das saídas.
- **Verificações:** o gerador roda cada caso duas vezes e aborta se não for bit a bit idêntico (DET). Os arquivos ficam no repositório se somarem menos de 20 MB comprimidos; acima disso, Git LFS.
- **Testes:** UNI do carregador (metadados completos, chaves esperadas, hashes conferem).
- **Critério:** matriz completa gerada, determinística e com metadados.

### 0.4 Caracterização no dev e diagnóstico cruzado

- **Entra:** `test/test_characterization.py`, que reexecuta cada caso do golden com o código do fork no ambiente dev e compara. Os casos de FWI só passam a rodar no dev depois do passo 1.1 (antes disso, `fwi.py` não importa) e ficam marcados `xfail` até lá.
- **Verificação-chave:** com o código ainda idêntico à 0.1.10, as saídas brutas no dev batem bit a bit com as do ambiente de referência? A diferença entre os dois ambientes é só de versões (PyOpenCL 2022 × 2026, NumPy 1.21 × 2.5). Se bater, o critério BIT da Fase 1 vale direto no dev. Se não bater, o teste localiza a etapa que diverge (geração da wavelet, montagem do modelo, kernel ou pós-processamento), e o critério passa a ser BIT nas saídas brutas mais a tolerância de ulp no pós-processamento, registrado no G0.
- **Testes:** CAR, DET.
- **Critério:** forward e gradiente de todos os casos aprovados no critério decidido; decisão registrada.

### 0.5 Linha de base matemática da 0.1.10

- **Entra:** `test/test_math.py` (marcador `slow`), com:
  - **gradiente × diferenças finitas:** ⟨g, δm⟩ contra [J(m + εδm) − J(m − εδm)] / 2ε, com ε varrido em décadas; o `g_smooth` já é 0 por padrão. Registra o melhor erro relativo;
  - **solução analítica:** limite acústico (vs = 0, ρ constante) contra a função de Green 2D convoluída com a Ricker, longe das bordas;
  - **ordem de convergência:** o erro contra a solução analítica cai com dh e dt nas taxas esperadas (4 ou 8 no espaço, 2 no tempo);
  - **PML:** energia refletida pela borda abaixo de um limiar;
  - **estabilidade:** execução longa sem crescimento de energia;
  - **produto escalar do adjunto:** ⟨A u, v⟩ × ⟨u, Aᵀ v⟩ para um passo de tempo, usando os kernels de atualização e os adjuntos. Na 0.1.10 só se mede; o critério de ≤ 1e-5 vale para os kernels novos.
- **Critério:** números registrados no G0 como linha de base. Daqui em diante, nenhuma fase pode piorar esses números (exceto o produto escalar, que precisa melhorar até o limite).

### 0.6 Benchmark de referência

- **Entra:** `bench/` com `pytest-benchmark` e três cenários: pequeno (exemplo da documentação, 100×100 + PML, 4 tiros, 1.051 passos), médio (≈400×400, 8 tiros, 3.000 passos) e grande (≈450 mil células, 6.000 passos). Mais um contador de lançamentos e cópias por passo (instrumentação via monkeypatch das chamadas do PyOpenCL).
- **Cuidados:**
  - Memória: com `chpr = 20`, o cenário médio usa ≈15 GB de RAM na 0.1.10, e a máquina tem 23 GB. O grande, com 50 tiros, não cabe; ele é medido com 1 e 2 tiros e extrapolado, e a extrapolação é registrada como tal.
  - Variabilidade: a GeForce varia o clock com a temperatura. Aquecimento, mediana de 5, temperatura e clock registrados por execução (`nvidia-smi --query-gpu`), e descarte de execuções com throttling.
  - Versões: medir a 0.1.10 com PyOpenCL 2022 (ref) e 2026 (dev), para separar o ganho do fork da regressão do PyOpenCL.
- **Testes:** DES; UNI do contador de lançamentos.
- **Critério:** coeficiente de variação abaixo de 3% em cada cenário; resultados em `bench/baseline/` com metadados da máquina.

### 0.7 Bancada de profiling

- **Entra:** `tools/profile.sh`, com receitas de `nsys`, `ncu` e `compute-sanitizer`, e um autoteste do sanitizer.
- **Verificações:**
  - Autoteste do sanitizer: um kernel CuPy com acesso fora do buffer proposital precisa ser detectado pelo `compute-sanitizer` 2024.1 com o runtime CUDA 13. Se não for, instalar o CUDA Toolkit 13.x no sistema (ver Pendências).
  - `ncu` lendo contadores (depois de liberar o acesso); `nsys` gerando a linha do tempo de um forward CUDA.
  - O OpenCL NVIDIA é medido com eventos de profiling (`CommandQueue` com `PROFILING_ENABLE`), não pelo Nsight.
- **Critério:** as três ferramentas funcionando, com um relatório de exemplo anexado.

### 0.8 Runner self-hosted e jobs de GPU

- **Entra:** runner do GitHub Actions nesta máquina, restrito ao repositório privado; jobs `opencl`, `gpu` e `sanitizer` a cada push, e um job noturno de DES.
- **Verificações:** um push de teste dispara os jobs; um teste que falha de propósito faz o job falhar; o runner não roda em PRs de forks.
- **Critério:** CI de CPU e de GPU verdes no `main`.

### Gate G0

- [ ] Golden masters completos, determinísticos, com metadados (0.3).
- [ ] Critério de caracterização decidido e aprovado no dev (0.4).
- [ ] Linha de base matemática registrada (0.5).
- [ ] Benchmark de referência com CV < 3% nos três cenários (0.6).
- [ ] Profiling e sanitizer funcionando (0.7).
- [ ] CI de CPU e de GPU verdes (0.1, 0.8).
- [ ] Todos os bugs conhecidos em `xfail` estrito (0.2).

## Fase 1: correções de host no OpenCL (versão 0.1.11 do fork)

Nenhuma conta do propagador muda; por isso o critério é BIT contra o golden master em todos os passos. A análise estima que estas duas primeiras fases já entregam a maior parte do ganho em modelos pequenos.

### 1.1 Imports e empacotamento

- **Entra:** `MemoizeJac` próprio (classe pequena, sem depender de nomes privados do SciPy); `np.kaiser` no lugar de `numpy.lib.function_base`; `segyio` com import tardio, só nas funções que leem SEG-Y; `pyopencl` em `install_requires`; `package_data` com `torch/` e `data/`; metadados em `pyproject.toml`; nome de distribuição `pyfwi-cuda` com o import `PyFWI`; `python_requires >= 3.10`.
- **Testes:** PKG (wheel num venv limpo, todos os módulos importam com NumPy 2.5 e SciPy 1.18, arquivos presentes); UNI do `MemoizeJac` (mesmo comportamento do original do SciPy 1.7.3: cache de `fun` e `jac`); CAR, agora incluindo os casos de FWI; os `xfail` de import e de `segyio` passam a passar.
- **Critério:** todos os módulos importam no dev; CAR aprovado.

### 1.2 Seleção de dispositivo

- **Entra:** escolha explícita de plataforma e dispositivo sem `create_some_context`, preservando o significado de `inpa['platform']` e `inpa['device']`.
- **Testes:** UNI com plataformas simuladas (um, vários e nenhum dispositivo); o `xfail` do dispositivo único passa; API (a mesma chamada da 0.1.10 escolhe o mesmo dispositivo).
- **Critério:** funciona com um ou vários dispositivos; o comportamento antigo se mantém onde já funcionava.

### 1.3 Kernels em cache e argumentos fixados

- **Entra:** cada kernel criado uma vez por `WavePropagator`; argumentos constantes fixados com `set_args`.
- **Testes:** BIT; um teste que trata `RepeatedKernelRetrieval` como erro; DES de host por lançamento (meta medida na análise: de 168 µs para 4 a 32 µs).
- **Critério:** BIT; nenhum aviso; diferença de desempenho entre PyOpenCL 2022 e 2026 eliminada.

### 1.4 Resíduo e wavelet na GPU; cópias não bloqueantes

- **Entra:** o resíduo `[nt, nr, 5]` e a wavelet sobem uma vez, e o kernel lê pela posição `t` (só a indexação muda, não a conta); cópias com `is_blocking=False` e eventos explícitos.
- **Testes:** BIT; DES (contador de cópias bloqueantes: −750 por tiro no exemplo da documentação); DET.
- **Critério:** BIT; contagem de cópias conforme a meta.

### 1.5 Checkpoints contíguos

- **Entra:** layout `(ns, nchp, nz, nx)`; `W.W` vira propriedade que devolve uma visão com a mesma ordem de eixos `[:, :, s, c]`.
- **Testes:** API (forma, `dtype`, valores e ordem de eixos de `W.W` iguais aos da 0.1.10; o exemplo de gradiente da documentação roda); BIT; DES (≈50× por cópia a 400×400); MEM (pico de RAM não aumenta).
- **Critério:** BIT; API intacta.

### 1.6 Filtro passa-baixa elemento a elemento

- **Entra:** `lowpass` e `adj_lowpass` com `|h|²·X` no lugar de `np.diag(h) @ X`; dado observado filtrado uma vez por banda.
- **Testes:** UNI com diferença relativa ≤ 1e-12 contra a versão antiga (mantida no teste como referência), para vários `nt`, número de traços, ordens e eixos; DES (65 a 75×); FWI de 2 iterações conforme as tolerâncias; o teste adjunto ⟨lowpass x, y⟩ = ⟨x, adj_lowpass y⟩ continua valendo.
- **Critério:** tolerâncias atendidas; ganho medido.

### 1.7 Buffers com margem

- **Entra:** alocações com linhas de margem zeradas antes e depois de cada campo, alinhadas ao `CL_DEVICE_MEM_BASE_ADDR_ALIGN` e usadas via sub-buffer, para que as leituras do estêncil caiam dentro da alocação.
- **Verificação decisiva:** na NVIDIA, o resultado com margem é bit a bit igual ao golden? Se for, a mudança fica no host, no modo `legacy`. Se não for, ela muda as contas e vai para o modo `fixed` (passo 2.4), registrada assim no G1.
- **Testes:** BIT na NVIDIA; no POCL, saída finita e DET (o `xfail` do POCL passa a passar). Com isso entra um golden POCL, gerado com a margem, e o CI de CPU passa a rodar CAR e API no POCL.
- **Critério:** POCL finito e determinístico; decisão sobre o modo registrada.

### 1.8 Diagnóstico

- **Entra:** `PyFWI.info()`, com origem do pacote (fork e commit), backend, dispositivo, versões e modo numérico; o mesmo resumo vai para o log na primeira propagação.
- **Testes:** UNI (campos presentes; funciona sem GPU); API (`import PyFWI` continua sem efeitos colaterais).

### 1.9 Release 0.1.11 do fork

- **Entra:** CHANGELOG, versão e tag.
- **Testes:** PKG completo (Python 3.10 a 3.13); scripts de `example/` rodando sem edição; DES contra a linha de base; FWI.
- **Critério:** o G1.

### Gate G1

- [ ] CAR e BIT em toda a matriz, na NVIDIA (e no POCL, a partir do 1.7).
- [ ] Ganhos medidos e registrados; nenhum cenário mais lento que a linha de base.
- [ ] Instala e importa com NumPy 2 e SciPy atuais.
- [ ] MAT sem piora.
- [ ] FWI de 2 iterações dentro da tolerância.
- [ ] Bugs corrigidos com `xfail` convertidos em testes normais.

## Fase 2: backend CUDA de paridade (versão 0.2.0)

O CUDA entra como opcional; o padrão continua `opencl`.

### 2.1 Motor e interface de backend

- **Entra:** `PyFWI/engine/` (laço temporal, política de checkpoints, lote de tiros) e `PyFWI/backends/opencl.py`, com a interface alocar / copiar / compilar com `#define` / lançar / sincronizar. `WavePropagator` vira fachada sobre `self._engine`.
- **Testes:** BIT (é refatoração pura); API; UNI da interface com um backend falso em NumPy, que permite testar o motor sem dispositivo.
- **Critério:** BIT; cobertura do motor acima de 90% só com UNI.

### 2.2 Seletor de backend e fallback

- **Entra:** precedência `inpa['backend']` → `PYFWI_BACKEND` (`opencl`, `cuda`, `auto`, `cuda:strict`) → padrão `opencl`; fallback com o motivo no log; casamento entre dispositivo OpenCL e ordinal CUDA pelo PCI bus ID; dispositivo OpenCL que não é NVIDIA mantém `auto` no OpenCL.
- **Testes:** SEL com todos os ramos: sem CuPy (import bloqueado por monkeypatch), sem GPU, driver antigo simulado, `cuda:strict` falhando com mensagem clara e `inpa` com CPU ou NVIDIA.
- **Critério:** todos os ramos cobertos e testados no CI de CPU.

### 2.3 Backend CUDA de paridade

- **Entra:** `PyFWI/kernels/` com o cabeçalho de portabilidade; `PyFWI/backends/cuda.py` com `RawModule` compilando os mesmos `.cl` em modo `legacy`, sempre para cubin da arquitetura do dispositivo; teste de limite (`if (i >= Nz || j >= Nx) return;`) nos kernels sem ele; margem dos buffers do passo 1.7.
- **Testes:**
  - PAR em toda a matriz;
  - DET;
  - SAN zero (a margem do 1.7 deixa as leituras herdadas dentro da alocação);
  - um teste de que o módulo carrega com NVRTC mais novo que o driver;
  - teste de grade não múltipla do bloco (malhas 61×67, 100×100 e 127×129).
- **Critério:** L2 relativo ≤ 1e-6 contra o OpenCL na mesma GPU; tolerância final fixada no G2.

### 2.4 Modos numéricos

- **Entra:** `PYFWI_NUMERICS=legacy|fixed`. O modo `fixed` traz literais com sufixo `f`, `dt` com 9 dígitos, limites de laço corretos (sem leitura fora do domínio), `vz` correto no poço e a margem, caso o 1.7 a tenha mandado para cá. O CUDA nasce em `fixed`; o OpenCL fica em `legacy`.
- **Testes:**
  - PRE: zero FP64 nos kernels quentes em `fixed`, no CI de CPU via NVRTC; o teste de caracterização de FP64 no `legacy` continua;
  - os `xfail` de `dt`, `vz` e leitura fora do buffer passam no `fixed` e continuam falhando no `legacy`;
  - diferenças `fixed` × `legacy` medidas para toda a matriz e publicadas no CHANGELOG;
  - MAT no `fixed` (o produto escalar do adjunto tem de melhorar ou se manter);
  - SAN.
- **Critério:** PRE zero; diferenças publicadas; MAT sem piora.

### 2.5 Contrato da API no CUDA

- **Entra:** `W.W` e `W.seismogram` copiados da GPU só quando acessados; caminho `show=True`; `torchfwi.Fwi` com tensores na CPU.
- **Testes:** API com `PYFWI_BACKEND=cuda`, os mesmos testes do OpenCL parametrizados por backend; scripts de `example/` sem edição; INT na CPU.
- **Critério:** a mesma suíte de API passa nos dois backends.

### 2.6 Extras de instalação

- **Entra:** extras `[cuda13]` (`cupy-cuda13x`) e `[cuda12]` (`cupy-cuda12x`); `envs/cuda12` em `setup_envs.sh`.
- **Testes:** PKG para cada extra num venv limpo; PAR rápido no `envs/cuda12`; `check_env.py` aprovado nos três ambientes.
- **Critério:** os dois extras instalam e passam no PAR.

### 2.7 Relatório de paridade e release 0.2.0

- **Entra:** `tools/parity_report.py`, que gera `docs/migracao-cuda/relatorios/paridade-0.2.0.md` com o erro por caso e por campo; CHANGELOG; tag.
- **Testes:** PAR completo; FWI de 2 iterações CUDA × OpenCL; DES (a paridade CUDA não pode ser mais lenta que o OpenCL com kernels em cache); PKG.

### Gate G2

- [ ] PAR em toda a matriz, com a tolerância final registrada.
- [ ] SAN zero; PRE zero no `fixed`.
- [ ] SEL completo; API parametrizada nos dois backends.
- [ ] FWI entre backends dentro da tolerância.
- [ ] Extras `cuda13` e `cuda12` funcionando.

## Fase 3: kernels CUDA otimizados (versão 0.3.0)

Cada otimização é um commit separado, com benchmark antes e depois, para que uma regressão aponte a mudança culpada. **Testes em todos os passos:** OTM contra a paridade `fixed`, DET, SAN, PRE, API e DES antes e depois, com os números na mensagem do commit.

| Passo | Otimização | Testes específicos | Meta |
| --- | --- | --- | --- |
| 3.1 | Mapeamento coalescido (`threadIdx.x` em `j`) e tamanho de bloco ajustável por GPU (32×8, 64×4…) com tabela de autotune | OTM com várias formas de bloco; malhas não múltiplas do bloco | A medir (análise) |
| 3.2 | Ordem como `template<int ORDER>`, sem leituras de `c3 = c4 = 0` e com limites corretos | PRE; SAN; OTM nas ordens 4 e 8 | Menos leituras na ordem 4 |
| 3.3 | Pré-cálculo de λ + 2μ e de 1/dx | OTM; UNI do pré-cálculo | Menos leituras e divisões |
| 3.4 | Injeção e registro 1D (`nr` threads, tabela de coordenadas) | OTM em `acq_type` 0/1/2; teste com receptores na borda | −2 kernels de malha por passo |
| 3.5 | Fusão: `update_v` e `update_tau` com injeção e registro; adjunto fundido com o imaging | OTM; MAT (produto escalar) | 4 → 2 kernels por passo no forward |
| 3.6 | Tempo no device: contador incrementado por um kernel mínimo; fonte e resíduo lidos por `t` | OTM bit a bit contra o passo anterior | Pré-requisito do 3.7 |
| 3.7 | CUDA Graphs com K passos por grafo; `show=True` fora do grafo | OTM bit a bit com e sem grafo; API com `show=True`; DES do tempo de host por passo; PRF (intervalos ociosos) | Poucos µs por kernel |
| 3.8 | Checkpoints na GPU quando couberem; orçamento GPU → host pinned assíncrono → recomputação | MEM com orçamento forçado pequeno: resultado bit a bit igual ao sem orçamento; EST perto do limite de 12 GB | Sem estouro de VRAM |
| 3.9 | Tiros em lote (grade 3D x, z, tiro); gradientes parciais somados em ordem fixa no fim | DET (crítico); OTM com B = 1, 2 e 4 | B vezes menos lançamentos |
| 3.10 | PML só nas bordas: blocos do interior pulam o amortecimento | OTM; MAT (reflexão da PML) | −2 leituras por célula no interior |

### 3.11 Padrão `auto` e release 0.3.0

- **Entra:** o padrão muda para `auto`. Fica para decidir e registrar no G3 se o OpenCL passa a usar `fixed` por padrão nesta versão.
- **Testes:** SEL do novo padrão; PKG; FWI; DES completo; PRF; EST.

### Gate G3

- [ ] Metas de desempenho contra a linha de base da 0.1.10 (estimativas da análise): 20 a 100× no cenário pequeno, 10 a 40× no médio e 3 a 10× no grande. Atingidas ou com o desvio explicado.
- [ ] Nenhuma regressão acima de 5% entre os commits da fase.
- [ ] OTM, DET, SAN e PRE em toda a matriz.
- [ ] PRF: linha do tempo do Nsight Systems sem intervalos ociosos relevantes no cenário pequeno; banda por kernel no Nsight Compute registrada, com os kernels quentes perto do teto de memória da RTX 3060.
- [ ] EST perto do limite de VRAM.

## Fase 4: recursos opt-in (versão 0.4.0)

Tudo aqui é opcional e desligado por padrão. Cada recurso tem tolerância própria, validada por uma FWI completa, não só por um sismograma.

### 4.1 Função custo e suavização na GPU

- **Entra:** `cupy.fft` e `cupyx.scipy.ndimage.gaussian_filter`; fim dos `deepcopy` repetidos.
- **Testes:** UNI (custo e resíduo adjunto contra a versão NumPy, ≤ 1e-6); FWI; DES (a CPU sai do caminho crítico, conferido no PRF).

### 4.2 Boundary saving

- **Entra:** a cada passo, só a faixa de `sdo` células junto à PML é guardada, e o interior é reconstruído.
- **Testes:**
  - gradiente contra o caminho com checkpoints (≤ 1e-4);
  - MEM: memória proporcional ao perímetro, verificada em três tamanhos de malha;
  - execução longa sem deriva da reconstrução (EST);
  - FWI completa.

### 4.3 Modo FP16

- **Entra:** campos em meia precisão com escalonamento, como no SeisCL.
- **Testes:**
  - teste de overflow e underflow com amplitudes extremas;
  - SNR do sismograma contra o FP32;
  - FWI completa com erro do modelo final dentro da meta (proposta: no máximo 10% acima do FP32);
  - DES (até ~2×) e MEM (metade).

### 4.4 Multi-GPU por tiros

- **Entra:** um processo ou stream por GPU; soma dos gradientes no host, em ordem fixa.
- **Testes:** com uma só RTX, dois processos na mesma GPU (`CUDA_VISIBLE_DEVICES`) devem dar resultado bit a bit igual a um processo (DET); falha de um processo propaga erro claro.
- **Critério:** escala quase linear numa máquina com mais GPUs, se houver uma disponível; senão, só a corretude é verificada, e isso fica registrado no gate.

### 4.5 `torchfwi` com tensores na GPU

- **Entra:** DLPack sem cópia, no `envs/torch`.
- **Testes:** INT (mesmo `data_ptr`; resultados iguais ao caminho NumPy; gradiente × diferenças finitas; tensores na CPU continuam funcionando).

### 4.6 Validação geofísica

- **Entra:** casos de referência de FWI multiescala e de todas as estratégias de `TimeLapse`, nos dois backends.
- **Testes:** FWI; revisão do geofísico com as figuras e as curvas de misfit em `docs/migracao-cuda/relatorios/`.

### 4.7 Release 0.4.0

PKG, FWI, DES, PRF, EST e CHANGELOG com as tolerâncias de cada recurso opt-in.

### Gate G4

- [ ] Cada recurso opt-in com tolerância documentada e validada por FWI completa.
- [ ] Padrões inalterados (testes de API passam sem nenhuma variável de ambiente).
- [ ] Ganhos de memória e de tempo medidos.
- [ ] Revisão geofísica registrada.

## Encerramento e manutenção

| Passo | O que é feito | Verificação |
| --- | --- | --- |
| 5.1 | Documentação do fork: instalação, backends, modos numéricos, tolerâncias; notebooks de `example/` atualizados | DOC: notebooks executados em versão reduzida no CI |
| 5.2 | Merge periódico do `upstream` | Suíte completa depois de cada merge; o `check_env.py` aprovado |
| 5.3 | Correções de bug devolvidas ao PyFWI original por PR | Só com sua autorização; cada PR com o teste que reproduz o bug |
| 5.4 | Decidir publicação: repositório público, nome no PyPI | Sua decisão; antes, PKG completo e revisão de licença (CuPy MIT e cuda.core Apache-2.0, compatíveis com GPLv3) |
| 5.5 | Otimizações condicionais: tiling, blocagem temporal, laço inteiro num grafo com nós condicionais | Só se o Nsight Compute mostrar kernels presos na banda de memória; mesmo protocolo da Fase 3 |

## Pendências que dependem de você

1. **Liberar os contadores da GPU para o Nsight Compute** (antes do passo 0.7). Requer sudo e reinício:

   ```bash
   echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-profiling.conf
   sudo update-initramfs -u && sudo reboot
   # depois: grep RmProfilingAdminOnly /proc/driver/nvidia/params   → deve mostrar 0
   ```

2. **CUDA Toolkit 13.x no sistema**, só se o autoteste do 0.7 mostrar que o `compute-sanitizer` 2024.1 não funciona com o runtime CUDA 13.
3. **Token para o runner self-hosted** (passo 0.8), gerado em *Settings → Actions → Runners* do repositório.
4. **Uma RTX de outra geração**, se possível, para não ajustar o código a uma só arquitetura (Fase 3).
5. **Apoio de um geofísico** para validar FWI e TL-FWI (Fases 0 e 4).
6. **Decisões futuras:** nome de distribuição definitivo, momento de tornar o repositório público e devolução de correções ao original.

## Comandos de referência

```bash
cd ~/Desktop/apps/cuda_pyfwi/pyfwi-cuda
tools/setup_envs.sh all                                  # recria ../envs/ref-0.1.10 e ../envs/dev
../envs/dev/bin/python tools/check_env.py                # verificação completa do ambiente
../envs/dev/bin/python -m pytest -m "not slow"           # suíte rápida (antes de cada commit)
../envs/dev/bin/python -m pytest -m "opencl or gpu"      # testes que exigem a RTX
../envs/ref-0.1.10/bin/python -m pytest                  # o que roda na referência
```
