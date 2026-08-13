AEGIR 현재 서버 환경 구성 (authoritative, Git-tracked)
갱신일: 2026-08-13

옛 /home/tako, /home/megapose, /home/hohyun 및 AEGIR_old 기반 명령은 실행하지
않는다. 이전 문서는 다음 경로에 보관했다.

  /workspace/hohyun_ws/AEGIR/docs/archive/ENVIRONMENT_SETUP_SERVER_legacy_2026-08-11.txt

현재 환경

  /workspace/.conda/envs/dnsplatter        Python 3.10, DN-Splatter, AEGIR,
                                            2DGS rasterizer, IRGS tracer
  /workspace/.conda/envs/diffusion_renderer DiffusionRenderer inference
  /workspace/.conda/envs/aegir_colmap       COLMAP executable 전용

활성화 없이 실행하는 권장 방식

  export BASE=/workspace
  export WORKSPACE="$BASE/hohyun_ws"
  export ROOT="$WORKSPACE/AEGIR"
  export DNS_ENV="$BASE/.conda/envs/dnsplatter"
  export DR_ENV="$BASE/.conda/envs/diffusion_renderer"
  export COLMAP_ENV="$BASE/.conda/envs/aegir_colmap"
  export PYTHONPATH="$ROOT:$WORKSPACE/external/dn-splatter${PYTHONPATH:+:$PYTHONPATH}"

  "$DNS_ENV/bin/python" -c \
    'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
  "$COLMAP_ENV/bin/colmap" -h

CUDA/extension 경로

  export CUDA_ROOT="$DNS_ENV/lib/python3.10/site-packages/nvidia/cu13"
  export CUDA_HOME="$CUDA_ROOT"
  export PATH="$DNS_ENV/bin:$CUDA_ROOT/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_ROOT/lib:$DNS_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

  /workspace/hohyun_ws/AEGIR/.cuda -> $CUDA_ROOT
  /workspace/hohyun_ws/AEGIR/.torch_extensions \
    -> /workspace/hohyun_ws/AEGIR/outputs/torch_extensions_cache_workspace

Packed environment 확인 결과 (2026-08-13)

  세 환경의 Python executable, sys.prefix, scripts path가 모두 현재
  /workspace/.conda/envs/<name>을 가리킨다. 옛 prefix는 conda-unpack 자체와
  conda-meta/history에만 남아 있다. 따라서 실행 중인 환경에 conda-unpack을
  다시 적용하지 않는다.

주의

  - root shell이므로 apt 설치에 sudo를 사용하지 않는다.
  - 컨테이너가 교체되면 apt로 설치한 git/tmux는 다시 설치해야 할 수 있다.
  - GPU의 물리 VRAM은 약 98 GB로 보이지만 작업 한도는 24 GiB다.
  - source revision과 외부 repo patch는
    /workspace/hohyun_ws/AEGIR/docs/SOURCE_PROVENANCE.md를 확인한다.
