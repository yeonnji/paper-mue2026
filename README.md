# 3DGS Endoscopy Baseline

이 브랜치(`baseline`)는 [INRIA GRAPHDECO의 원본 3D Gaussian Splatting(3DGS)](https://github.com/graphdeco-inria/gaussian-splatting)을 C3VD 내시경 데이터셋에 학습할 수 있도록 최소한으로 수정한 **베이스라인 코드**입니다. specular highlight 모델링을 포함한 연구 코드는 `main` 브랜치를 참고하세요.

> 원본 3DGS의 설치 방법, CLI 옵션 전체 목록, SIBR 뷰어 사용법 등 이 프로젝트에서 수정하지 않은 부분은 [원본 저장소 README](https://github.com/graphdeco-inria/gaussian-splatting#readme)를 참고하세요.

## 프로젝트 개요

- **데이터셋**: Colonoscopy 3D Video Dataset (C3VD), EndoGSLAM 전처리 파이프라인 사용
- **역할**: `main` 브랜치(specular 모델링 포함)와 비교하기 위한 베이스라인. 이 코드에서 어떤 부분이 추가·수정되었는지 확인하는 용도
- **핵심 수정 사항**: 원본 3DGS 대비, 내시경 데이터셋 학습을 위한 최소 수정 — evaluation 지표(PSNR/SSIM/LPIPS/MS-SSIM) 로깅, `results.txt` 저장, C3VD 시퀀스 일괄 실행 스크립트 추가

## 브랜치 구성

| 브랜치     | 내용                                                                                                  |
| ---------- | ----------------------------------------------------------------------------------------------------- |
| `main`     | `gs_endo` — specular highlight 모델링을 포함한 연구 코드 (MUE 2026 Best Paper)                        |
| `baseline` | `baseline` — 원본 3DGS에 C3VD/내시경 데이터셋 학습을 위한 최소한의 수정만 가한 버전 **(현재 브랜치)** |

`main` 브랜치 코드를 받으려면:

```bash
git clone --recursive <이 저장소 URL>
# 기본이 main 브랜치
```

## 설치

원본 3DGS와 동일하게 submodule을 포함해 clone하고, conda 환경을 구성합니다.

```bash
git clone --recursive -b baseline <이 저장소 URL>
conda env create --file environment.yml
conda activate gaussian_splatting
```

CUDA/컴파일러 요구사항, submodule 빌드 트러블슈팅 등은 [원본 3DGS README](https://github.com/graphdeco-inria/gaussian-splatting#readme)를 참고하세요.

## 실행 방법

```bash
# 학습
python train.py --model_path <output_dir> --eval -s <sequence_path> --iteration 30000

# 렌더링
python render.py --model_path <output_dir> --eval -s <sequence_path> --iteration 30000

# 전체 C3VD 시퀀스 일괄 실행
bash run_endo.sh

# 지표 계산 (PSNR/SSIM/LPIPS/MS-SSIM)
python metrics.py <output_dir 상위 폴더>
```

학습 중 각 evaluation iteration마다 `results.txt`에 지표(L1, PSNR, LPIPS, SSIM, MS-SSIM)의 평균 및 표준편차가 기록되고, 학습 종료 후 최고 성능 iteration이 함께 저장됩니다.

## 참고

- 원본 3DGS 논문/코드: [https://github.com/graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)
  - Kerbl, B., Kopanas, G., Leimkühler, T., Drettakis, G. "3D Gaussian Splatting for Real-Time Radiance Field Rendering." ACM TOG 42(4), 2023.
- PR-ENDO 코드: [https://github.com/SanoScience/PR-ENDO](https://github.com/SanoScience/PR-ENDO)
  - Kaleta, J., et al. "PR-ENDO: Physically Based Relightable Gaussian Splatting for Endoscopy." MICCAI 2025.
