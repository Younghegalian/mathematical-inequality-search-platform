# Mathematical Inequality Search Platform

![Python](https://img.shields.io/badge/python-3.11%2B-3776AB)
![Tests](https://img.shields.io/badge/tests-unittest-brightgreen)
![Domain](https://img.shields.io/badge/domain-Navier--Stokes%20inequalities-2F4858)
![Status](https://img.shields.io/badge/status-research%20prototype-orange)
![Artifacts](https://img.shields.io/badge/artifacts-JSONL%20%2B%20HTML-6C63FF)

Navier-Stokes 방정식의 비선형 항을 dissipation 항으로 흡수할 수 있는 부등식 구조를 자동 탐색하기 위한 연구용 플랫폼이다. 목표는 PDE를 직접 수치적으로 푸는 것이 아니라, 다음 형태의 후보 부등식과 증명 경로를 대량 생성하고 검증 가능한 후보만 선별하는 것이다.

```text
T(u) <= epsilon * D(u) + C * L(u)
```

대표적인 목표 구조는 3차원 vorticity formulation에서 다음과 같은 critical inequality를 찾는 것이다.

```text
integral |omega|^3 <= epsilon * ||grad omega||_2^2 + C * ||omega||_2^2
```

이 저장소는 완성된 정리 증명기가 아니라, 불평등 조합 공간을 구조적으로 탐색하고 후보를 벤치마크하기 위한 실험 플랫폼이다.

## Research Objective

Navier-Stokes regularity 문제에서 핵심 병목은 nonlinear enstrophy production을 viscosity가 제공하는 dissipation으로 제어할 수 있는지이다.

```text
nonlinear production <= absorbable dissipation + lower-order controlled term
```

플랫폼은 다음 질문을 계산 가능한 형태로 바꾼다.

1. 어떤 target functional이 scaling-critical 구조를 갖는가?
2. Holder, Sobolev, Gagliardo-Nirenberg, Young 조합으로 어떤 RHS가 생성되는가?
3. 생성된 RHS가 dissipation term을 포함하고 lower-order norm으로 닫히는가?
4. 그 후보가 단순 embedding인지, 실제 nonlinear Navier-Stokes 항과 관련된 후보인지 구분 가능한가?
5. 자동화된 symbolic, scaling, numerical falsification gate를 통과하는 후보가 존재하는가?

## Mathematical Model

현재 scaling convention은 다음을 기준으로 한다.

```text
u        -> +1
omega    -> +2
D^k f    -> scaling(f) + k
||f||_p  -> scaling(f) - 3/p
integral -> scaling(f) - 3
```

기본 dissipation과 controlled quantity는 다음과 같다.

```text
D(omega) = ||grad omega||_2^2
L(omega) = ||omega||_2^2
```

closure 판정은 다음 ODE 형태를 기준으로 한다.

```text
GOOD: X'(t) <= C X(t)
BAD:  X'(t) <= C X(t)^p, p > 1
```

## Inequality Combination Space

현재 rule library는 다음 조합 공간을 생성한다.

| Rule | Role |
| --- | --- |
| Integral-to-Norm | integral target을 Lp norm expression으로 변환 |
| Holder | product integral을 norm product로 분해 |
| Biot-Savart | velocity/strain proxy를 vorticity norm으로 연결 |
| Sobolev | H1 dissipation에서 endpoint norm 제어 |
| Gagliardo-Nirenberg | intermediate Lp norm을 L2와 H1 사이 interpolation |
| Young | interpolation product를 epsilon-dissipation + controlled term으로 흡수 |

탐색 대상은 크게 두 종류다.

| Family | Examples | Purpose |
| --- | --- | --- |
| Critical norm targets | `omega_L3`, `omega_cubic_integral`, generated `omega_Lp_crit_q` | scaling-critical 후보 공간 점검 |
| Nonlinear proxies | `vortex_stretching`, `strain_vorticity`, `velocity_strain_gradient`, `strain_strain_vorticity` | Navier-Stokes nonlinear production과의 관련성 점검 |

## Search Pipeline

```mermaid
flowchart LR
    A["Target generation"] --> B["Rule-chain sampling"]
    B --> C["Scaling filter"]
    C --> D["Early kill / complexity filter"]
    D --> E["Relevance-first scoring"]
    E --> F["Promotion queue"]
    F --> G["Verification gates"]
    G --> H["Human theorem review packet"]
```

대량 random search는 `target-policy`와 `rule-policy`를 통해 조합 공간을 샘플링한다.
현재 ranking은 nonlinear-term relevance를 closure friendliness보다 높은 신호로 둔다. 즉, 순수 Sobolev embedding처럼 잘 닫히는 후보보다 Navier-Stokes nonlinear production proxy에서 출발했거나 Holder/Biot-Savart/Integral-to-Norm 경로를 실제로 거친 후보를 먼저 promotion queue에 올린다.

```bash
python3 scripts/run_random.py \
  --budget 20000 \
  --target-policy frontier \
  --rule-policy sample
```

주요 sampling mode:

| Option | Description |
| --- | --- |
| `--target-policy coverage` | target family coverage를 우선 |
| `--target-policy family-balanced` | family별 균형 샘플링 |
| `--target-policy frontier` | verification 실패 정보를 반영하여 frontier 쪽에 가중 |
| `--rule-policy shuffle` | rule order를 무작위 섞음 |
| `--rule-policy sample` | rule subset을 무작위 선택 |

## Verification Protocol

promotion queue에 올라간 후보는 다음 gate를 순서대로 통과해야 한다.

| Gate | Implementation | Failure Meaning |
| --- | --- | --- |
| Symbolic replay | `verifier.symbolic.replay_proof` | 저장된 proof path가 현재 rule library로 재현되지 않음 |
| Scaling audit | `verifier.pipeline.scaling_audit` | LHS/RHS scaling mismatch 또는 3D critical scaling 실패 |
| Target relevance | `verifier.pipeline.target_relevance_check` | closure-friendly embedding일 뿐 nonlinear control 후보로 보기 어려움 |
| Numeric counterexample search | `verifier.numeric.stress_candidate` | spectral/adversarial sample에서 구조적으로 불안정 |
| Human math review | generated review packet | side condition, endpoint, constant dependence, domain assumption 수동 검토 필요 |

검증 실행:

```bash
python3 scripts/promote_good.py --data-dir data
python3 scripts/run_verification.py --data-dir data
```

numeric gate는 정리를 증명하지 않는다. 역할은 amplitude, frequency, profile, random Fourier family에서 ratio 폭주나 RHS degeneracy를 찾아 후보를 빠르게 반증하는 것이다.

현재 stress family:

```text
single_mode
multi_mode
localized_bump
two_scale
```

## AlphaFold-Inspired Benchmark Framing

이 프로젝트에서 "AlphaFold-inspired benchmark"는 생물학 모델을 의미하지 않는다. 의미는 다음과 같다.

```text
large candidate generation -> fixed benchmark set -> blind ranking -> staged verification
```

부등식 탐색에 맞춘 benchmark protocol은 다음과 같이 설계한다.

| Benchmark Layer | Inequality Search Equivalent |
| --- | --- |
| Fixed target set | nonlinear proxy와 scaling-critical norm target의 고정 test suite |
| Candidate ranking | closure score, scaling compatibility, proof length, rule diversity |
| Blind validation | search에 사용하지 않은 target family와 random seed로 재검증 |
| Structural falsification | spectral stress test와 family-growth ratio test |
| Expert review | 자동 gate 통과 후보에 대한 수동 lemma reconstruction |

권장 benchmark suite:

| Suite | Targets | Primary Metric |
| --- | --- | --- |
| `critical-norm` | generated `omega_Lp_crit_q` | critical scaling pass rate |
| `nonlinear-core` | vortex/strain/velocity nonlinear proxies | relevance pass rate |
| `absorption` | Young absorption candidates | GOOD closure yield |
| `stress-falsification` | promoted candidates | max ratio growth, failure family |
| `holdout-targets` | search에 쓰지 않은 generated targets | generalization survival |

핵심 성능 지표:

```text
deduplicated survivor yield
GOOD closure rate
symbolic replay pass rate
critical scaling pass rate
nonlinear relevance pass rate
numeric stress pass rate
top-k human-reviewable candidate count
```

## Data and Artifacts

실험 결과는 GitHub에 포함하지 않는다. 대량 탐색 로그는 크기가 빠르게 증가하므로 `data/`는 `.gitignore` 처리되어 있다.

| Artifact | Path |
| --- | --- |
| run events | `data/runs/*.jsonl` |
| run summaries | `data/results/summary_*.json` |
| promotion queue | `data/promotions/verification_queue.json` |
| verification results | `data/verifications/*.json` |
| review packets | `data/verifications/review_packets/*.md` |
| HTML report | `data/reports/index.html` |
| experiment bundle | `data/experiments/<id>/` |

## Reproducible Commands

Install:

```bash
python3 -m pip install -r requirements.txt
```

List available targets:

```bash
python3 scripts/run_search.py --list-targets --no-save
```

Run a small search:

```bash
python3 scripts/run_random.py --budget 1000 --target-policy frontier --rule-policy sample
```

Build report:

```bash
python3 scripts/build_index.py
python3 scripts/report.py
```

The report integrates mathematical-symbol notation for frontier inequalities and spatial field diagnostics for the most advanced promoted candidates.

Run verification:

```bash
python3 scripts/promote_good.py --data-dir data
python3 scripts/run_verification.py --data-dir data
```

Render spatial field diagnostics for a promoted candidate:

```bash
python3 scripts/visualize_contour.py \
  --queue-path data/promotions/verification_queue.json \
  --candidate-index 0 \
  --profile two_scale \
  --grid-size 64
```

Render only the most advanced frontier candidates:

```bash
python3 scripts/visualize_contour.py \
  --queue-path data/promotions/verification_queue.json \
  --frontier \
  --top 6 \
  --dedupe
```

Run tests:

```bash
python3 -m unittest
```

## Repository Layout

```text
core/        expression AST, canonicalization, scaling, rules, search, filters
engine/      runner, recording, indexing, promotion, reporting
ns/          Navier-Stokes variables, target definitions, generated target families
verifier/    symbolic replay, scaling audit, relevance gate, numerical stress tests
scripts/     command-line entry points
tests/       regression tests for search, reporting, promotion, verification
config/      default search configuration
data/        local-only generated artifacts
```

## Limitations

이 저장소의 자동 gate는 수학적 증명을 대체하지 않는다.

- symbolic replay는 구현된 rule library 내부의 재현성만 확인한다.
- scaling audit는 현재 AST scaling model에 대한 consistency check이다.
- numeric stress test는 falsification 도구이며 proof가 아니다.
- endpoint estimates, boundary condition, function space assumptions, constant dependence는 human review 단계에서 별도로 검토해야 한다.
- Navier-Stokes regularity에 대한 claim은 자동 gate 통과만으로 성립하지 않는다.

## Development Status

현재 목표는 theorem prover가 아니라 benchmarkable inequality-search engine을 만드는 것이다. 단기 개발 우선순위는 다음과 같다.

1. nonlinear target family 확장
2. duplicated proof path 제거
3. holdout benchmark split 도입
4. numeric stress family 확장
5. promoted candidate의 formal lemma export
6. 탐색 로그를 ML ranking dataset으로 변환
