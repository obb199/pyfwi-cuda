# PyFWI: núcleo GPU e plano de migração para CUDA

Oct 24, 2026 · @Gabriel Carvalho

## Resumo executivo

Sim: levar o núcleo do PyFWI para ferramentas baseadas em CUDA o deixa bem mais rápido em placas NVIDIA, mas o ganho vem de reestruturar a orquestração, não da troca de API em si. Na versão 0.1.10, a maior parte do tempo de uma avaliação de FWI vai para Python, sincronizações e um filtro na CPU, não para a GPU.

- **Onde está o tempo hoje.** Um forward + gradiente do exemplo da documentação (100×100, 4 tiros, 1.051 passos) dispara ≈44 mil lançamentos de kernel e ≈11,4 mil cópias bloqueantes. Quase todo lançamento recria o objeto do kernel no PyOpenCL: 168 µs de CPU por lançamento, contra 4 µs reaproveitando o kernel e seus argumentos (medido nesta máquina de teste).
- **Kernels lentos nas GeForce RTX, alvo do fork.** Constantes sem sufixo `f` fazem os kernels quentes calcularem em FP64 (42 a 131 instruções FP64 por kernel no PTX), e nas RTX o FP64 roda a uma fração mínima da taxa FP32 (1/64 na geração Ampere). O mapeamento de índices também tende a gerar acessos não coalescidos.
- **Gargalo fora da GPU.** O passa-baixa da função custo multiplica por uma matriz diagonal densa (`np.diag(h) @ X`); a multiplicação elemento a elemento dá o mesmo resultado 65–75× mais rápido.
- **Ferramenta recomendada.** CuPy 14 (kernels C++ compilados em tempo de execução pelo NVRTC, CUDA Graphs, cuFFT), com cuda.core como complemento para levar o laço inteiro para a GPU. Os três arquivos `.cl` atuais compilam como CUDA com um cabeçalho de \~10 linhas, de sm\_75 a sm\_120.
- **Ganho esperado (estimativa, a confirmar na Fase 0).** 20–100× em modelos 2D pequenos, 10–40× em médios e 3–10× em grandes; o modo FP16 opcional pode somar até \~2×.
- **Mudança controlada, num fork.** A API pública (`WavePropagator`, `FWI`, `TimeLapse`, `torchfwi.Fwi`, `W.W`) não muda, e o fork mantém o pacote de import `PyFWI`: os scripts só trocam o comando de instalação. O backend é escolhido por `PYFWI_BACKEND` ou por uma chave opcional no `inpa`, com OpenCL como padrão até a paridade ser comprovada e fallback automático.

Próximo passo: Fase 0 (testes de caracterização e benchmark de referência numa GPU NVIDIA) e Fase 1 (correções de host que já aceleram o OpenCL) antes de portar qualquer kernel.

## O PyFWI hoje

A versão mais recente é a **0.1.10**, publicada no PyPI em 5 nov. 2024; o `main` do GitHub aponta para o mesmo commit (`1434a16`) e não houve release depois disso. O pacote faz modelagem e FWI elástica 2D no domínio do tempo, com o núcleo em OpenCL via PyOpenCL ([PyPI](https://pypi.org/project/PyFWI/), [GitHub](https://github.com/AmirMardan/PyFWI), [artigo SoftwareX 2023](https://espace.inrs.ca/id/eprint/13791/), [documentação](https://pyfwi.readthedocs.io/)).

| Item | Situação na 0.1.10 |
| --- | --- |
| Física e malha | Elástico 2D (vx, vz, τxx, τzz, τxz), diferenças finitas em malha intercalada, ordem 4 ou 8 no espaço e 2 no tempo, PML por amortecimento |
| Passo interno | Limitado a 0,6 ms: `dt` do usuário é dividido em `dt_scale` subpassos |
| Gradiente | Método adjunto; o campo direto é reconstruído de trás para frente e ressincronizado em checkpoints (`chpr`, % dos passos) |
| Aquisição | Poço a poço (0), superfície (1) ou ambos (2); posições de receptores fixadas nos kernels por `#define` |
| Inversão | CG e L-BFGS do SciPy, multiescala por frequência, 7 estratégias de TL-FWI, regularização TV/Tikhonov, `torchfwi.Fwi` para PyTorch |
| Código | 15 arquivos Python (\~6.500 linhas) + 3 arquivos OpenCL (675 linhas); licença GPLv3+ |
| Testes | 1 teste (wavelet Ricker); nada cobre o núcleo GPU |

Problemas de empacotamento que a migração precisa resolver junto:

- `pyopencl` não está em `install_requires`, só em `build-system.requires`; `pip install PyFWI` não instala o motor da GPU.
- `fwi.py`, `tl_fwi.py` e `optimization.py` importam `MemoizeJac` de `scipy.optimize.optimize`, nome que o SciPy atual não exporta mais (testado: `ImportError` no SciPy 1.18.1). O `requirements.txt` fixa SciPy 1.7.3 e o pacote declara Python ≥ 3.7.
- O wheel publicado não inclui o propagador acústico experimental em PyTorch (`PyFWI/torch`) nem a pasta `data`.

## Anatomia do núcleo GPU (OpenCL)

O núcleo é um laço temporal em Python que enfileira, a cada passo e tiro por tiro, vários kernels pequenos numa única fila em ordem. No exemplo da documentação (100×100 + PML de 20, 4 tiros, `dt` de 4 ms → 1.051 passos internos, `chpr` = 20), um forward + gradiente soma ≈44 mil lançamentos e ≈11,4 mil cópias bloqueantes.

| Etapa (por tiro) | Kernels por passo | Transferências host↔GPU |
| --- | --- | --- |
| Forward | 4 atualizações (`update_velx`, `update_velz`, `update_tauz`, `update_tauxz`) + `injSrc` a cada `dt_scale` passos | 5 cópias GPU→host bloqueantes por checkpoint + 5 sismogramas no fim |
| Gradiente | 4 de reconstrução reversa + `Adj_update_tau`, `Adj_update_v`, `Grad` + `Adj_injSrc` a cada `dt_scale` passos | 5 cópias host→GPU bloqueantes de resíduo a cada `dt_scale` passos e 5 por checkpoint |

**Gargalos de orquestração (host)**

1. **Kernel recriado a cada lançamento.** `self.prg.update_velx(...)` cria um `cl.Kernel` novo por chamada; o PyOpenCL 2026.1.4 emite o aviso `RepeatedKernelRetrieval` justamente por isso ([issue #831](https://github.com/inducer/pyopencl/issues/831)). Medido: 168 µs de CPU por lançamento, 32 µs com o kernel reaproveitado e 4 µs com os argumentos fixados uma vez.
2. **Sincronização a cada poucos passos.** `cl.enqueue_copy` é bloqueante por padrão, e o resíduo sobe fatia a fatia dentro do laço adjunto. Cada cópia esvazia a fila da GPU e elimina a sobreposição entre CPU e GPU.
3. **Checkpoints no host, em layout espalhado.** `W[campo][:, :, s, c]` num array `(tnz, tnx, ns, nchp)` copia com passo de memória grande: a 400×400, 0,84 ms por campo contra 0,017 ms em layout contíguo (≈50×). A memória cresce com `5 × ns × nchp × células`: 15 GB de RAM para 400×400, 8 tiros, 3.000 passos e `chpr` = 20.
4. **Kernels pontuais lançados na malha inteira.** `injSrc` e `Adj_injSrc` usam `tnz × tnx` work-items, com resto de divisão inteira, para tocar uma fonte e algumas centenas de receptores.

**Problemas dentro dos kernels**

1. **FP64 involuntário.** As constantes entram como literais sem sufixo (`#define c1 1.125000`, `#define dt 0.000571`), que em C e OpenCL C são `double` quando o dispositivo suporta FP64. Compilando os mesmos arquivos com NVRTC 12.9, o PTX traz de 42 (`update_velx`) a 131 (`Grad`) instruções FP64 por kernel; com sufixo `f`, zero. Em GeForce, FP64 roda a 1/64 da taxa FP32 ([whitepaper GA102](https://web.cs.miami.edu/home/burt/learning/csc596.231/docs/NVIDIA-ampere-GA102-GPU-Architecture-Whitepaper-V1.pdf)).
2. **Termos de coeficiente zero ainda são calculados.** Na ordem 4, `c3 = c4 = 0`, mas as leituras e multiplicações desses termos continuam no código gerado, porque a semântica IEEE estrita impede o compilador de removê-las.
3. **Leituras fora do buffer.** As atualizações vão até `i = Nz − sdo`, mas o estêncil lê até `i + sdo`, a linha `Nz`, que não existe; na ordem 4 também lê `i − 3` com `i = 2`. O efeito fica na PML, mas é comportamento indefinido.
4. **Acesso pouco coalescido.** A dimensão 0 do NDRange, a que varia mais rápido dentro de um warp, percorre linhas `i`, enquanto a memória é contígua em `j`. Com `local_size=None`, o resultado depende da forma de grupo que o driver escolher.
5. **`dt` truncado.** O `#define dt` usa `%f` (6 casas): 0,000571 em vez de 0,000571428…, um passo 0,075% menor que o usado para amostrar a wavelet no host.
6. **Bug de registro no poço.** Em `elastic_crosswell.cl`, o sismograma `vz` grava valores de `vx`.

**Fora do laço de propagação**, o maior custo está na função custo: `lowpass` e `adj_lowpass` filtram com `np.diag(h) @ X`, uma matriz densa `nt × nt`. Só essa multiplicação custa 218 ms (nt 662, 400 traços) e 2,7 s (nt 1.512, 1.000 traços); a versão elemento a elemento dá o mesmo resultado (diferença relativa 4×10⁻¹⁶) em 3,3 ms e 36 ms, e a FWI chama o filtro 3 vezes por avaliação.

Método: medições com scripts próprios que reproduzem os padrões do PyFWI, sem executar o código do pacote, numa CPU Xeon 2,1 GHz (2 núcleos) com PyOpenCL 2026.1.4 sobre POCL e NVRTC 12.9; não havia GPU NVIDIA disponível.

## CUDA deixaria mais rápido?

Sim, desde que a migração ataque a orquestração; trocar só a API rende pouco. Com kernels equivalentes, CUDA ficou no máximo \~30% à frente do OpenCL na maioria de 16 benchmarks, diferença que some numa comparação justa ([Fang et al., ICPP 2011](https://hgpu.org/?p=5748)). O SeisCL, de dois coautores do PyFWI, relata o OpenCL um pouco mais lento em GPUs NVIDIA para modelos grandes e mais rápido para modelos pequenos ([SeisCL](https://github.com/gfabieno/SeisCL)).

Uma tradução 1:1 não basta porque o custo dominante hoje (recriar kernels, sincronizar, filtrar com matriz densa) não depende da API. Um port para CuPy com um lançamento por kernel por passo reduziria o custo por lançamento, mas o mesmo corte existe no PyOpenCL com kernels em cache.

O que só o caminho CUDA oferece, e que explica o ganho grande:

- **CUDA Graphs.** Um passo de tempo, ou um bloco de passos, vira um grafo lançado de uma vez. No teste da NVIDIA (V100), o custo efetivo por kernel caiu de 9,6 µs com lançamentos sincronizados para 3,4 µs com grafo, para um kernel de 2,9 µs ([NVIDIA, 2019](https://developer.nvidia.com/blog/cuda-graphs/)). Nós condicionais IF/WHILE (CUDA 12.4+) mantêm laços inteiros na GPU, sem voltar à CPU ([NVIDIA, 2024](https://developer.nvidia.com/blog/dynamic-control-flow-in-cuda-graphs-with-conditional-nodes)). O PyOpenCL 2026.1.4 não expõe `cl_khr_command_buffer`, o equivalente em OpenCL (verificado no código do pacote).
- **Meia precisão madura.** Modelagem elástica em FP16 com CUDA ficou quase 2× mais rápida que FP32, com metade da memória e sem prejudicar FWI e RTM ([Fabien-Ouellet, Geophysics 2020](https://publications.polymtl.ca/6303/)); o README do SeisCL lista os modos FP16 como vantagem do backend CUDA.
- **Bibliotecas e ferramentas.** cuFFT (`cupy.fft`) e `cupyx.scipy.ndimage` tiram a função custo e a suavização da CPU; streams e memória pinned tornam as cópias assíncronas; Nsight Systems/Compute e `compute-sanitizer` perfilam e flagram acessos fora do buffer; DLPack troca tensores com PyTorch sem cópia.

| Cenário | Gargalo na 0.1.10 | O que resolve | Ganho estimado |
| --- | --- | --- | --- |
| Pequeno: exemplo da doc (140×140 com PML, 4 tiros, 1.051 passos) | Overhead de Python e PyOpenCL (estimado em mais de 90% do tempo) e filtro denso | Kernels em cache, grafos, tiros em lote, filtro elemento a elemento | 20–100× |
| Médio: ≈400×400, 8 tiros, 3.000 passos | Lançamentos, cópias espalhadas, filtro denso | O anterior + checkpoints na GPU | 10–40× |
| Grande: ≈450 mil células, 50 tiros, 6.000 passos | Lançamentos e kernels FP64 pouco coalescidos | Kernels FP32 coalescidos, fusão, grafos | 3–10× |
| Grande, com modo FP16 opcional | Banda de memória | Armazenamento e cálculo em meia precisão | até \~2× adicional |

Como a estimativa foi feita: lançamentos e cópias contados no código, custo por lançamento medido nesta máquina e tempo de GPU estimado pela banda de memória de uma placa de \~1 TB/s. A Fase 0 troca essas estimativas por medições na GPU-alvo.

## Ferramentas do ecossistema CUDA

A base recomendada é o **CuPy**, com kernels C++ compilados em tempo de execução pelo NVRTC: é o caminho mais curto a partir do desenho atual, que já gera kernels C com `#define` no momento da execução. Os três `.cl` do PyFWI compilaram como CUDA com NVRTC 12.9 e um cabeçalho de \~10 linhas, de sm\_75 (Turing) a sm\_120 (Blackwell), nas ordens 4 e 8. Versões consultadas no PyPI em out. 2026:

| Ferramenta | Uso no PyFWI | Pontos fortes | Limitações | Veredito |
| --- | --- | --- | --- | --- |
| [CuPy 14.2](https://pypi.org/project/cupy-cuda13x/) (MIT) | `RawModule` via NVRTC, captura com `Stream.begin_capture`/`end_capture` e `Graph.launch`, `cupy.fft`, `cupyx.scipy.ndimage` | Reaproveita os kernels atuais; arrays no estilo NumPy para o resto do pipeline; extra `[ctk]` instala o CUDA pelo pip | Overhead de Python fora dos grafos; dependência grande | **Base recomendada** |
| [cuda.core 1.2](https://pypi.org/project/cuda-core/) (Apache-2.0) | `Program` (NVRTC), `launch`, `GraphBuilder` com nós condicionais | API oficial da NVIDIA, enxuta, acesso aos recursos novos de grafos | Sem biblioteca de arrays; ainda em desenvolvimento ativo | Complemento para o laço inteiro na GPU |
| [Warp 1.18](https://pypi.org/project/warp-lang/) (Apache-2.0) | Kernels escritos em Python, captura de grafos, tiles | Kernels legíveis, diferenciáveis, interop com PyTorch e JAX ([docs](https://warp.readthedocs.io/)) | Reescrever todos os kernels; wheels pedem driver R580+ e GPU sm\_75+ | Alternativa para física nova |
| PyTorch 2.14 com extensão C++/CUDA, ou Triton 3.8 | Operador customizado e `torch.cuda.CUDAGraph`, como no [Deepwave](https://github.com/ar4/deepwave) | Integração nativa com `torchfwi` | Dependência pesada; build por versão de CUDA e torch; Triton é pouco natural para estênceis com halo | Integrar por DLPack, não como núcleo |
| [Numba-CUDA 0.30](https://pypi.org/project/numba-cuda/) (BSD) | `@cuda.jit` | Kernels em Python | Em modo manutenção: só correções críticas; novidades migram para o Numba-CUDA-MLIR | Não recomendado |
| Extensão C++/CUDA própria (nanobind + nvcc) | Laço temporal inteiro em C++ | Teto de desempenho | Matriz de wheels por SO × CUDA × arquitetura | Só se o perfil mostrar overhead residual |
| JAX 0.12 | `lax.scan` compilado pelo XLA | Laço compilado e autodiferenciação | Reescrita total, pouco controle sobre os kernels, memória alta na autodiferenciação | Não recomendado |

Foco GeForce RTX: as quatro gerações de RTX (séries 20 a 50; sm\_75, sm\_86, sm\_89 e sm\_120) rodam com o CUDA 13, que exige driver da série 580 ou mais nova ([notas do CUDA 13.0](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html)). Placas de Maxwell a Volta ficaram fora do CUDA 13 e não são alvo do fork ([NVIDIA, 2025](https://developer.nvidia.com/blog/navigating-gpu-architecture-support-a-guide-for-nvidia-cuda-developers/)). Os kernels compilaram para as quatro arquiteturas RTX no teste com NVRTC; o extra `cuda13` vira o padrão do fork, e `cuda12` fica só para máquinas presas a drivers anteriores ao 580.

## Arquitetura-alvo

&#91;embedded content: arquitetura proposta · 6 camadas, 2 backends\]

O usuário continua chamando as mesmas classes; a fachada repassa a um motor único, e o seletor escolhe entre OpenCL e CUDA, com volta automática ao OpenCL.

**Organização do pacote (proposta)**

- `PyFWI/wave_propagation.py` mantém `WavePreparation` e `WavePropagator` como fachada e delega a `self._engine`.
- `PyFWI/engine/` concentra laço temporal, política de checkpoints, lote de tiros e captura de grafo, sem chamar APIs de GPU diretamente.
- `PyFWI/backends/opencl.py` e `PyFWI/backends/cuda.py` expõem a mesma interface: alocar, copiar, compilar com `#define`, lançar, sincronizar e, no CUDA, capturar grafo.
- `PyFWI/kernels/` guarda o cabeçalho de portabilidade, as fontes de paridade compartilhadas e os `.cu` otimizados, exclusivos do CUDA.

**Fonte única na paridade.** O cabeçalho abaixo, prefixado aos `.cl` atuais, bastou para compilar os três arquivos com NVRTC, e já leva `j`, a direção contígua na memória, para `threadIdx.x`. Kernels sem teste de limite (`MakeAllZero`, `MakeGradZero`, `injSrc`, `Adj_injSrc`) ganham `if (i >= Nz || j >= Nx) return;`, porque a grade CUDA é arredondada para múltiplos do bloco.

```c
#define __kernel extern "C" __global__
#define __global
#define __local __shared__
__device__ __forceinline__ int get_global_id(int d) {
    return d == 0 ? (int)(blockIdx.y * blockDim.y + threadIdx.y)   // i (z)
                  : (int)(blockIdx.x * blockDim.x + threadIdx.x);  // j (x), coalescido
}
```

**Laço em grafo.** O que muda a cada passo (índice de tempo, amostra da fonte, fatia do resíduo) passa a ser lido da memória da GPU, a partir de um contador que um kernel mínimo incrementa. Assim o mesmo grafo serve para todos os passos (esboço):

```python
s = cp.cuda.Stream(non_blocking=True)
with s:
    s.begin_capture()
    for _ in range(K):          # K passos por grafo, ex.: dt_scale
        step_forward(state)     # kernels fundidos; t lido do device
    graph = s.end_capture()
for _ in range(nt // K):
    graph.launch(stream=s)      # 1 chamada Python a cada K passos
```

## Plano de migração por fases

&#91;embedded content: roteiro proposto · 5 fases, 5 gates, 4 versões\]

As duas primeiras fases não tocam em CUDA: elas criam a rede de segurança e já entregam a maior parte do ganho em modelos pequenos. O CUDA entra na Fase 2 como opcional e só vira padrão depois que o gate G3 comprovar ganho sem regressão.

Detalhes que o desenho não mostra:

- **Versões.** Os números do roteiro são do fork, publicado com nome de distribuição próprio; o import continua `PyFWI`.
- **Fase 0.** Os testes de caracterização são gerados com a própria 0.1.10 num ambiente congelado (Python 3.10 com SciPy 1.7.3, a versão fixada no `requirements.txt`, que ainda exporta `MemoizeJac`), para que a referência não dependa de nenhuma correção.
- **Fase 1.** Todas as mudanças ficam no host e não alteram nenhuma conta do propagador; por isso o critério é igualdade bit a bit, não tolerância.
- **Fase 2.** Os kernels de paridade usam a mesma fonte do OpenCL e o modo `legacy`; as correções de bugs entram no modo `fixed`, com a diferença medida publicada.
- **Fase 3.** Cada otimização entra num PR separado, com benchmark antes e depois, para que uma regressão aponte a mudança culpada.
- **Fase 4.** Boundary saving, FP16 e multi-GPU são opt-in e trazem tolerâncias próprias, validadas por uma FWI completa e não só por um sismograma.

## Otimizações para máxima velocidade

&#91;embedded content: priorização qualitativa · 20 otimizações em 4 quadrantes\]

Os cinco itens do quadrante destacado atacam o gargalo que domina a 0.1.10 e não dependem de CUDA; os do quadrante à direita são os que só o caminho NVIDIA viabiliza. Detalhe de cada otimização:

| Otimização | Mecanismo | Ganho esperado | Fase |
| --- | --- | --- | --- |
| Kernels em cache | Criar cada kernel uma vez e fixar os argumentos constantes | 168 → 4–32 µs de CPU por lançamento (medido) | 1 |
| Resíduo e fonte na GPU | Subir o resíduo `[nt, nr, 5]` e a wavelet uma vez; o kernel lê pela posição `t` | Elimina 750 cópias bloqueantes por tiro no exemplo da doc | 1 |
| Checkpoints contíguos | Layout `(ns, nchp, nz, nx)` na GPU quando couber; `W.W` vira visão transposta | ≈50× por cópia a 400×400 (medido) | 1 e 3 |
| Filtro elemento a elemento | `\|h\|² · X` no domínio da frequência; dado observado filtrado uma vez por banda | 65–75×, mesmo resultado (medido) | 1 |
| Literais float e `dt` exato | `#define` gerado com sufixo `f` e 9 dígitos | Até \~2× em GeForce nos kernels de malha grande (estimado) | 2, modo `fixed` |
| Mapeamento coalescido | `threadIdx.x` em `j`; blocos 32×8 ou 64×4 ajustados por GPU | A medir: depende da forma de grupo que o driver usa hoje | 2 e 3 |
| CUDA Graphs | Capturar K passos; tempo e amostras lidos do device | Custo de lançamento cai para poucos µs por kernel | 3 |
| Fusão de kernels | `update_v` (vx, vz) e `update_tau` (τxx, τzz, τxz) com injeção e registro dentro; adjunto fundido com o imaging | 4 → 2 kernels por passo no forward e menos tráfego de memória | 3 |
| Tiros em lote | Grade 3D `(x, z, tiro)`; gradientes parciais somados ao fim | Mais paralelismo em malhas pequenas e B vezes menos lançamentos | 3 |
| Ordem como template | `template<int ORDER>`, sem leituras para `c3 = c4 = 0`, limites corretos | Menos leituras na ordem 4 e fim das leituras fora do buffer | 2 e 3 |
| Injeção e registro 1D | Kernel de `nr` threads com tabela de coordenadas dos receptores | Remove 2 kernels de malha inteira por passo | 3 |
| Pré-cálculo de parâmetros | Guardar λ + 2μ e multiplicar por 1/dx em vez de dividir por dx | Menos leituras e divisões por célula | 2 e 3 |
| PML só nas bordas | Blocos do interior pulam o amortecimento e a leitura de dxpml e dzpml | Duas leituras a menos por célula no interior | 3 |
| Boundary saving | Guardar a cada passo só a faixa de `sdo` células junto à PML e reconstruir o interior exatamente | Memória por tiro proporcional ao perímetro, não à área; dispensa o `chpr` | 4 |
| FP16 opcional | Campos em meia precisão com escalonamento, como no SeisCL | Até \~2× e metade da memória ([Fabien-Ouellet, 2020](https://publications.polymtl.ca/6303/)) | 4 |
| Multi-GPU por tiros | Um processo ou stream por GPU; soma dos gradientes no host | Quase linear no número de GPUs | 4 |
| Função custo e suavização na GPU | `cupy.fft`, `cupyx.scipy.ndimage.gaussian_filter`, sem `deepcopy` repetido | Tira a CPU do caminho crítico de cada iteração | 4 |
| Tiling, blocagem temporal, laço no grafo | Só se o Nsight Compute mostrar kernels presos na banda de memória | A medir | Depois da 0.4 |

Nas GeForce RTX, duas linhas pesam mais que o resto: os literais float, porque o FP64 dessas placas roda a uma fração mínima da taxa FP32, e o orçamento de memória dos checkpoints, porque as RTX têm de 6 a 32 GB de VRAM, conforme o modelo. Por isso boundary saving e o modo FP16 sobem de prioridade dentro da Fase 4.

## Compatibilidade: mudança controlada sem alterar código de usuários

O contrato é simples: todo script que roda na 0.1.10 roda igual nas versões novas, com os mesmos imports, classes e saídas. Só muda quem executa os kernels, e isso é decidido fora do código do usuário.

1. **API congelada.** `WavePreparation`, `WavePropagator`, `FWI`, `TimeLapse` e `torchfwi.Fwi` mantêm assinaturas, chaves do `inpa` e retornos (dicts de `np.float32`, formatos `'2d'`/`'3d'`, reshape em ordem Fortran). `W.W[campo][:, :, s, c]`, usado no exemplo de gradiente da documentação, e `W.seismogram` viram propriedades que devolvem visões NumPy na mesma ordem de eixos, copiadas da GPU só quando acessadas.
2. **Seleção fora do código**, nesta precedência: chave opcional `inpa['backend']` → variável `PYFWI_BACKEND` (`opencl`, `cuda`, `auto`, `cuda:strict`) → padrão. O padrão fica `opencl` até a 0.2.x e passa a `auto` na 0.3.0.
3. **Fallback seguro.** Em `auto`, sem CuPy, sem GPU NVIDIA ou com driver antigo, o PyFWI registra o motivo no log e usa OpenCL; `cuda:strict` falha em vez de cair, para quem precisa garantir a GPU.
4. **Dispositivo respeitado.** Se `inpa['platform']` e `inpa['device']` apontam para um dispositivo OpenCL que não é NVIDIA (CPU, AMD, Intel), `auto` fica no OpenCL. Se apontam para uma NVIDIA, o seletor usa o ordinal CUDA da mesma placa, casado pelo PCI bus ID.
5. **Modos numéricos explícitos.** `PYFWI_NUMERICS=legacy` reproduz a 0.1.10 (literais `double`, `dt` com 6 casas, bugs conhecidos) para refazer resultados publicados; `fixed` corrige tudo. O OpenCL fica em `legacy` por padrão até a 0.3.0; o CUDA nasce em `fixed`, com as diferenças medidas publicadas no CHANGELOG.
6. **Instalação do fork.** O fork sai com nome de distribuição próprio (ex.: `pyfwi-cuda`) ou direto do Git, mas mantém o pacote de import `PyFWI` e a mesma API; os scripts não mudam, só o comando de instalação. Como os dois pacotes ocupam o mesmo nome de import, o PyFWI original precisa ser desinstalado antes. Extras: `[cuda13]` para RTX com driver 580 ou mais novo e `[cuda12]` para drivers anteriores; `pyopencl` passa a ser dependência declarada.
7. **Visualização e PyTorch.** `show=True` continua funcionando: nesse modo o laço sai do grafo e copia o campo a cada 20 passos, como hoje. `torchfwi.Fwi` segue aceitando e devolvendo tensores na CPU e passa a aceitar tensores na GPU via DLPack, sem cópia.
8. **Diagnóstico e depreciação.** `PyFWI.info()` mostra a origem do pacote (fork e commit), backend, dispositivo, versões e modo numérico, e o mesmo resumo vai para o log na primeira propagação. Nada da API pública é removido; mudanças futuras passam por `DeprecationWarning` por pelo menos duas versões menores.

Na prática, um script existente passa para a RTX sem edição; só a instalação muda:

```bash
pip uninstall -y PyFWI                             # o fork usa o mesmo nome de import
pip install "pyfwi-cuda[cuda13]"                   # nome de exemplo; o import continua PyFWI
PYFWI_BACKEND=cuda python fwi_example_surface.py   # mesmo script da 0.1.10
PYFWI_BACKEND=opencl python fwi_example_surface.py # volta ao comportamento atual
```

## Testes, validação numérica e benchmarks

Nenhum padrão muda sem passar por seis camadas de teste, rodando em CI com OpenCL na CPU (POCL) e num runner com GPU NVIDIA. Hoje o repositório tem um único teste, então a Fase 0 começa por aqui.

| Camada | O que verifica | Critério de aceite |
| --- | --- | --- |
| Caracterização (golden master) | Saídas da 0.1.10 gravadas em `.npz` para modelos pequenos (ex.: 60×60, 300 passos): forward, gradiente e 2 iterações de FWI, cobrindo `acq_type` 0/1/2, `components` 0–4, ordem 4/8, `energy_balancing`, `chpr` 0/20/100 e `parameterization` | Fase 1: propagação idêntica bit a bit; filtro com diferença relativa ≤ 1e-12 |
| Paridade entre backends | Kernels CUDA de paridade (mesma fonte, modo `legacy`) × OpenCL na mesma GPU NVIDIA | Erro relativo L2 ≤ 1e-6 (meta inicial, a confirmar na Fase 2) |
| Kernels otimizados | CUDA otimizado (`fixed`) × CUDA de paridade | Sismogramas ≤ 1e-5 e gradientes ≤ 1e-4 em erro relativo L2; FP16 com limites próprios, validados por uma FWI completa |
| Corretude matemática | Teste do produto escalar do adjunto (⟨Lm, d⟩ = ⟨m, Lᵀd⟩), gradiente contra diferenças finitas, meio homogêneo contra solução analítica | Desvio relativo ≤ 1e-5 no produto escalar |
| Memória e concorrência | `compute-sanitizer` (memcheck, racecheck, initcheck) em todos os kernels novos | Zero erros |
| Guarda de precisão (sem GPU) | O NVRTC compila os kernels no CI e o teste lê o PTX dos kernels quentes, como nesta análise | Zero instruções FP64 no modo fixed |
| Desempenho | `pytest-benchmark` ou `asv` com os três cenários da estimativa, na RTX da equipe e, se possível, numa RTX de outra geração, com mediana de 5 execuções após aquecimento, porque o clock das GeForce varia com a temperatura | Sem regressão acima de 5% entre commits; metas de cada fase atingidas |

Cada benchmark registra o tempo de forward, de gradiente e de uma iteração de FWI; o tempo de host por passo; a ocupação da GPU no Nsight Systems; a banda atingida por kernel no Nsight Compute; e a memória de pico na GPU e no host. A soma dos gradientes entre tiros evita `atomicAdd` em ponto flutuante, para que duas execuções iguais deem o mesmo resultado.

## Riscos e mitigação

O maior risco não é técnico: é mudar resultados sem aviso. Por isso cada mitigação abaixo preserva um caminho de volta ao comportamento da 0.1.10.

| Risco | Impacto | Mitigação |
| --- | --- | --- |
| Diferenças numéricas entre backends | Desconfiança nos resultados | Modos `legacy` e `fixed`, tolerâncias publicadas, relatório de paridade em cada release |
| Correção de bugs herdados (leitura fora do buffer, `vz` no poço, `dt` truncado) altera saídas | Quebra de reprodutibilidade de trabalhos publicados | Correções só no modo `fixed`; `legacy` preserva o comportamento antigo no OpenCL |
| Checkpoints não cabem na VRAM da RTX (6 a 32 GB, conforme o modelo) | Falha em modelos grandes | Orçamento de memória: GPU → host pinned assíncrono → recomputação ou boundary saving |
| Grafos incompatíveis com `show=True` e com depuração passo a passo | Perda de funcionalidade | Caminho sem grafo mantido e coberto por testes |
| Dependências pesadas (CuPy + componentes CUDA) | Instalação mais difícil | Extras opcionais, `[ctk]` instalando o CUDA via pip, OpenCL como alternativa |
| Máquina com driver anterior ao 580 | Sem acesso ao CUDA 13 | Extra `cuda12` (CUDA 12.9), que funciona com drivers mais antigos |
| Manutenção de dois backends | Custo contínuo | Fonte única para os kernels de paridade; otimizações avançadas só no CUDA |
| Licenças | Conformidade com a GPLv3+ | CuPy (MIT) e cuda.core/Warp (Apache-2.0) são compatíveis com GPLv3 |
| Projeto sem release desde nov. 2024 | Fork diverge do PyFWI original | Remote upstream com merge periódico, correções de bug devolvidas ao original por PR, aviso de modificação exigido pela GPLv3 e citação dos autores originais |
|  |  |  |

## Esforço e recursos

O plano soma cerca de 20 semanas de uma pessoa experiente em CUDA e Python científico. Com duas pessoas, cai para \~14 semanas: depois do gate G2, a Fase 4 (boundary saving, multi-GPU, função custo na GPU) corre em paralelo com a Fase 3, e só o modo FP16 espera os kernels otimizados.

- **Pessoas:** um desenvolvedor CUDA/CuPy no núcleo do trabalho e apoio de um geofísico para validar FWI e TL-FWI nas Fases 0 e 4.
- **GPUs:** uma GeForce RTX, a da equipe, como alvo principal e, se possível, uma RTX de outra geração, para não ajustar o código a uma só arquitetura. GPU de data center não é necessária.
- **CI:** GitHub Actions com POCL para o OpenCL na CPU e um runner self-hosted na máquina com a RTX para paridade, `compute-sanitizer` e benchmarks noturnos.
- **Ferramentas:** CuPy 14 (`cupy-cuda13x` e `cupy-cuda12x`), Nsight Systems, Nsight Compute, `compute-sanitizer` e `pytest-benchmark` ou `asv`.

## Fontes

- [PyFWI 0.1.10 no PyPI](https://pypi.org/project/PyFWI/) e [repositório AmirMardan/PyFWI](https://github.com/AmirMardan/PyFWI) (commit `1434a16`, código-fonte analisado)
- [Mardan, Giroux e Fabien-Ouellet (2023), PyFWI, SoftwareX 22](https://espace.inrs.ca/id/eprint/13791/) e [documentação do PyFWI](https://pyfwi.readthedocs.io/)
- [PyOpenCL, issue #831: aviso de recuperação repetida de kernel](https://github.com/inducer/pyopencl/issues/831)
- [Página de manual do clBuildProgram (opção -cl-single-precision-constant)](https://manpages.debian.org/jessie/opencl-1.2-man-doc/clBuildProgram.3clc.en.html)
- [NVIDIA Ampere GA102 GPU Architecture Whitepaper](https://web.cs.miami.edu/home/burt/learning/csc596.231/docs/NVIDIA-ampere-GA102-GPU-Architecture-Whitepaper-V1.pdf)
- [Fang, Varbanescu e Sips (2011), A Comprehensive Performance Comparison of CUDA and OpenCL](https://hgpu.org/?p=5748)
- [SeisCL (Fabien-Ouellet, Gloaguen e Giroux)](https://github.com/gfabieno/SeisCL)
- [Fabien-Ouellet (2020), Seismic modeling and inversion using half-precision floating-point numbers, Geophysics 85(3)](https://publications.polymtl.ca/6303/)
- [NVIDIA (2019), Getting Started with CUDA Graphs](https://developer.nvidia.com/blog/cuda-graphs/)
- [NVIDIA (2024), Dynamic Control Flow in CUDA Graphs with Conditional Nodes](https://developer.nvidia.com/blog/dynamic-control-flow-in-cuda-graphs-with-conditional-nodes)
- [NVIDIA (2025), Navigating GPU Architecture Support](https://developer.nvidia.com/blog/navigating-gpu-architecture-support-a-guide-for-nvidia-cuda-developers/)
- [NVIDIA, CUDA Toolkit 13.0 Release Notes](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html)
- [CuPy 14.2 no PyPI](https://pypi.org/project/cupy-cuda13x/) e [código de captura de grafos do CuPy (stream.pyx)](https://github.com/cupy/cupy/blob/main/cupy/cuda/stream.pyx)
- [cuda.core no PyPI](https://pypi.org/project/cuda-core/) e [documentação do cuda.core](https://nvidia.github.io/cuda-python/cuda-core/latest/)
- [Warp no PyPI](https://pypi.org/project/warp-lang/) e [documentação do Warp](https://warp.readthedocs.io/)
- [Numba-CUDA no PyPI](https://pypi.org/project/numba-cuda/)
- [Deepwave](https://github.com/ar4/deepwave)
