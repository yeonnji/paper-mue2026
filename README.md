# Removal of Specular Highlights in Endoscopic Images using 3D Gaussian Splatting (MUE 2026)

이 저장소는 논문 **"Removal of Specular Highlights in Endoscopic Images using 3D Gaussian Splatting"** (Yeon Ji, Jae-Hyeong Kim, Byeong-Seok Shin, Inha University)의 연구 코드입니다. [INRIA GRAPHDECO의 원본 3D Gaussian Splatting(3DGS)](https://github.com/graphdeco-inria/gaussian-splatting)을 기반 코드로 사용했으며, 내시경 영상에서 반복적으로 나타나는 specular highlight 문제를 다루기 위해 [PR-ENDO](https://github.com/SanoScience/PR-ENDO)의 구현을 참고하여 확장했습니다.

이 논문은 [MUE 2026](http://www.mue-conference.org/2026/) (The 20th International Conference on Multimedia and Ubiquitous Engineering)에서 **Best Paper Award**를 수상했습니다.

> 원본 3DGS의 설치 방법, CLI 옵션 전체 목록, SIBR 뷰어 사용법 등 이 프로젝트에서 수정하지 않은 부분은 [원본 저장소 README](https://github.com/graphdeco-inria/gaussian-splatting#readme)를 참고하세요.

## 프로젝트 개요

- **데이터셋**: Colonoscopy 3D Video Dataset (C3VD), EndoGSLAM 전처리 파이프라인 사용
- **베이스라인**: 3DGS에 내시경 데이터셋 학습을 위한 최소한의 수정만 가한 버전은 `gs_endo_origin`(별도 브랜치)에 있습니다.
- **핵심 아이디어**: 입력 영상을 view-independent한 base 구조(`I_base`)와 view-dependent한 specular 성분(`I_spec`)으로 분해하여, specular 성분을 제거한 tissue appearance를 재구성합니다.

## 브랜치 구성

이 저장소는 하나의 repo 안에서 연구 코드와 베이스라인을 브랜치로 구분해 관리합니다.

| 브랜치     | 내용                                                                                                                                     |
| ---------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `main`     | `gs_endo` — specular highlight 모델링을 포함한 연구 코드 (이 README가 설명하는 코드)                                                     |
| `baseline` | `gs_endo_origin` — 원본 3DGS에 C3VD/내시경 데이터셋 학습을 위한 최소한의 수정만 가한 버전. `main`이 어디서 갈라져 나왔는지 비교하는 용도 |

`baseline` 브랜치 코드를 받으려면:

```bash
git clone --recursive -b baseline <이 저장소 URL>
```

## 재현성(Reproducibility) 관련 유의사항

**논문 제출 시점의 코드는 별도로 백업해두지 않았습니다.** 이후에도 연구가 계속 진행되면서 현재 이 저장소의 코드는 논문 제출 이후 추가로 발전된 버전이며, 논문 본문(Method, Eq. 1–7)에 기술된 방법과 세부 구현이 다음과 같이 차이가 있습니다.

|                     | 논문에 기술된 방법                                                             | 현재 코드 (`utils/specular_module.py`, `utils/specular_losses.py`)                     |
| ------------------- | ------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------- |
| `I_spec` 추정 방식  | 신경망 없이 HSV prior + residual 기반으로 직접 최적화 (decoupled optimization) | `PoseEmbedding`(카메라 pose → MLP latent) + `SpecularBranch`(FiLM 컨디셔닝 CNN)로 예측 |
| Loss 구성           | `L_spec` (Eq.4), `L_base` (Eq.6) 2개 항                                        | `l_res, l_prior, l_sparse, l_tv, l_z, l_final` 6개 항                                  |
| Base loss weighting | `w = clamp(1 - α·P_HSV, w_min, 1)`                                             | `weight_map = 1 + α·prior` (부호/형태 다름)                                            |

즉 **이 저장소의 코드를 그대로 실행해도 논문 Table 1에 보고된 수치(PSNR/SSIM/LPIPS)가 그대로 재현된다는 보장은 없습니다.** 논문에 보고된 정확한 실험을 재현하려면 논문 제출 시점 근처의 커밋/백업을 별도로 찾아야 하며, 현재는 그런 스냅샷이 존재하지 않습니다. 이 코드는 "논문 아이디어의 최신/발전된 구현"으로 이해하고 사용해야 합니다.

## 논문 및 발표자료

`document/` 폴더에 아래 파일이 포함되어 있습니다.

| 파일 | 설명 |
| ---- | ---- |
| [`document/Removal of Specular Highlights in Endoscopic Images using 3D Gaussian Splatting.pdf`](document/Removal%20of%20Specular%20Highlights%20in%20Endoscopic%20Images%20using%203D%20Gaussian%20Splatting.pdf) | MUE 2026 제출 논문 |
| [`document/MUE2026_presentation.pdf`](document/MUE2026_presentation.pdf) | MUE 2026 발표 슬라이드 |

## 설치

원본 3DGS와 동일하게 submodule을 포함해 clone하고, conda 환경을 구성합니다.

```bash
git clone --recursive <이 저장소 URL>
conda env create --file environment.yml
conda activate gaussian_splatting
```

CUDA/컴파일러 요구사항, submodule 빌드 트러블슈팅 등은 [원본 3DGS README](https://github.com/graphdeco-inria/gaussian-splatting#readme)를 참고하세요.

## 실행 방법

```bash
# 학습 (specular branch 사용)
python train.py --model_path <output_dir> --eval -s <sequence_path> \
    --iteration 40000 --use_specular_branch --spec_schedule staged \
    --spec_start_iter 3000 --lambda_prior 0.05 --spec_prior_gamma 1.0

# 렌더링
python render.py --model_path <output_dir> --eval -s <sequence_path> \
    --iteration 40000 --use_specular_branch

# 전체 C3VD 시퀀스 일괄 실행
bash run_endo.sh

# 지표 계산 (PSNR/SSIM/LPIPS)
python metrics.py <output_dir 상위 폴더>
```

주요 하이퍼파라미터는 `arguments/__init__.py`의 `SpecularParams`에 정의되어 있습니다 (`use_specular_branch`, `spec_latent_dim`, `lambda_prior`, `lambda_sparse`, `lambda_tv`, `lambda_z`, `spec_schedule` 등). `train.py`, `render.py`, `metrics.py`의 기본 인자(`--source_path`, `--resolution`, `--iterations` 등)는 원본 3DGS와 동일합니다.

## 참고

- 원본 3DGS 논문/코드: [https://github.com/graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)
  - Kerbl, B., Kopanas, G., Leimkühler, T., Drettakis, G. "3D Gaussian Splatting for Real-Time Radiance Field Rendering." ACM TOG 42(4), 2023.
- PR-ENDO 코드: [https://github.com/SanoScience/PR-ENDO](https://github.com/SanoScience/PR-ENDO)
  - Kaleta, J., et al. "PR-ENDO: Physically Based Relightable Gaussian Splatting for Endoscopy." MICCAI 2025.
