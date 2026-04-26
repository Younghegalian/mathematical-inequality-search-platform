# Mathematical Inequality Search Platform

Navier-Stokes 방정식의 비선형 항과 dissipation 사이에서 성립할 수 있는 핵심 부등식 구조를 자동으로 탐색하는 연구용 플랫폼입니다.

이 프로젝트는 PDE를 직접 수치적으로 푸는 코드가 아닙니다. 대신 다음 형태의 부등식 후보를 대량으로 생성하고, 필터링하고, 검증하고, 좋은 후보만 다음 단계로 올리는 proof-search 시스템입니다.

```text
T(u) <= epsilon * D(u) + C * L(u)
```

예를 들어 vorticity omega에 대해 다음과 같은 구조를 찾는 것이 목표입니다.

```text
integral |omega|^3 <= epsilon * ||grad omega||_2^2 + C * ||omega||_2^2
```

이런 형태가 실제로 닫히면 에너지 부등식이 선형 Gronwall 형태로 떨어질 수 있고, global boundedness와 regularity 논의로 이어질 수 있습니다.

## 핵심 아이디어

Navier-Stokes의 어려운 부분은 대략 다음 균형입니다.

```text
nonlinear growth  vs  viscous dissipation
```

플랫폼은 이 균형을 직접 증명하지 않고, 그 증명에 쓰일 수 있는 부등식 경로를 탐색합니다.

탐색 과정은 다음 순서로 진행됩니다.

1. target term 생성
2. Holder, Sobolev, Gagliardo-Nirenberg, Young 등의 규칙 적용
3. scaling consistency 검사
4. 물리적으로 관련 없는 후보 제거
5. closure 형태 판정
6. symbolic / numeric verifier 통과 여부 확인
7. 좋은 후보만 survivor queue와 report에 기록

## 현재 구현 상태

현재 버전은 연구 프로토타입입니다. 이미 다음 기능이 들어 있습니다.

- 수식 AST와 canonical key
- Navier-Stokes 변수와 target 생성기
- Holder / Sobolev / GN / Young 계열 규칙
- beam search와 random search
- scaling filter
- closure classifier
- symbolic, scaling, relevance, numeric verifier
- 대량 탐색 로그 저장 구조
- HTML report 생성
- 대량 랜덤 탐색에서 좋은 후보만 집중적으로 보여주는 report pipeline

## 프로젝트 구조

```text
core/        수식 표현, 규칙, 탐색, 필터, closure, scoring
engine/      실행 상태, beam search, reporting
ns/          Navier-Stokes 변수, target, physics proxy
verifier/    symbolic / scaling / relevance / numeric 검증
scripts/     실행 스크립트
tests/       단위 테스트
config/      기본 설정
data/        로컬 탐색 결과 저장 위치
```

`data/` 아래의 대량 결과물은 GitHub에 올리지 않습니다. 몇만 개만 돌려도 파일이 커지기 때문에, 저장소에는 코드와 문서만 유지하고 실험 산출물은 로컬 또는 별도 artifact storage로 관리하는 방향입니다.

## 설치

Python 3.11 이상을 권장합니다.

```bash
python3 -m pip install -r requirements.txt
```

## 기본 실행

사용 가능한 target을 확인합니다.

```bash
python3 scripts/run_search.py --list-targets --no-save
```

작은 탐색을 실행합니다.

```bash
python3 scripts/run_random.py --budget 1000 --target-policy frontier --rule-policy sample
```

대량 탐색 예시는 다음과 같습니다.

```bash
python3 scripts/run_random.py --budget 20000 --target-policy frontier --rule-policy sample
```

## Report 생성

탐색 결과를 인덱싱하고 HTML report를 생성합니다.

```bash
python3 scripts/build_index.py
python3 scripts/report.py
```

생성된 report는 기본적으로 다음 위치에 만들어집니다.

```text
data/reports/index.html
```

실험별 report는 `data/experiments/.../reports/index.html` 아래에 생성될 수 있습니다.

## Verification

좋은 후보는 별도의 verifier pipeline으로 넘겨 symbolic, scaling, relevance, numeric 검사를 수행합니다.

```bash
python3 scripts/run_verification.py --data-dir data
```

검증 단계의 목적은 단순히 score가 높은 후보를 보여주는 것이 아니라, 실제 증명 후보로 이어질 수 있는 식만 다음 단계로 올리는 것입니다.

## 테스트

```bash
python3 -m unittest
```

## 연구 방향

앞으로 중요한 확장 방향은 다음과 같습니다.

- 중복 후보 제거 강화
- target generation 다양화
- scaling-critical family 집중 탐색
- verifier의 수학적 엄밀성 강화
- survivor candidate 기반 Monte Carlo local refinement
- 탐색 로그를 ML 학습 데이터셋으로 변환
- 좋은 후보가 나오면 proof assistant 또는 수동 증명 문서로 넘기는 후처리

## 저장소 이름

권장 GitHub 저장소 이름:

```text
mathematical-inequality-search-platform
```

약어 중심 이름 대신, 연구 플랫폼이라는 성격이 바로 드러나도록 전문적인 영어 이름을 사용합니다.
