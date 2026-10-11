# Pixal3D (TencentARC) - image GPU pour RTX 3080 Ti Laptop (sm_86)
# Environnement aligné sur l'install officielle TRELLIS.2 / Pixal3D :
#   Python 3.10 (python3 d'Ubuntu 22.04), torch 2.6.0 + CUDA 12.4
# Les extensions CUDA sont compilées depuis les sources pour sm_86
# (les wheels de la démo HF ciblent les GPU H-series).
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST="8.6" \
    NATTEN_CUDA_ARCH="8.6" \
    NATTEN_N_WORKERS=8 \
    MAX_JOBS=8 \
    PIP_NO_CACHE_DIR=1 \
    PIP_CONSTRAINT=/etc/pip-constraints.txt \
    PYTHONUNBUFFERED=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    LOW_VRAM=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget build-essential cmake ninja-build \
        python3 python3-dev python3-pip python-is-python3 \
        libgl1 libglib2.0-0 libjpeg-dev \
    && rm -rf /var/lib/apt/lists/*

# Verrouille torch pour toutes les installations pip suivantes
# (sinon une dépendance non épinglée le met à jour et casse les extensions compilées).
RUN printf 'torch==2.6.0\ntorchvision==0.21.0\ntriton==3.2.0\n' > /etc/pip-constraints.txt && \
    python -m pip install --upgrade pip setuptools wheel && \
    pip install torch==2.6.0 torchvision==0.21.0 triton==3.2.0 \
        --index-url https://download.pytorch.org/whl/cu124

WORKDIR /workspace
RUN git clone https://github.com/TencentARC/Pixal3D.git

# Dépendances Python de Pixal3D + celles de l'app web
# (trimesh==4.10.1 est incompatible avec MoGe, qui exige une version plus récente)
RUN sed -i 's/^trimesh==.*/trimesh>=4.11/' /workspace/Pixal3D/requirements.txt && \
    pip install -r /workspace/Pixal3D/requirements.txt && \
    pip install spaces nest_asyncio ninja lpips pandas && \
    pip install https://github.com/LDYang694/Storages/releases/download/20260430/utils3d-0.0.2-py3-none-any.whl

# flash-attn : wheel précompilée pour torch 2.6 / cu12 / cp310
RUN pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Extensions CUDA de TRELLIS.2 (compilées pour sm_86)
WORKDIR /tmp/extensions
RUN git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git && \
    pip install ./nvdiffrast --no-build-isolation
RUN git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git && \
    pip install ./nvdiffrec --no-build-isolation
RUN git clone --recursive https://github.com/JeffreyXiang/CuMesh.git && \
    pip install ./CuMesh --no-build-isolation
RUN git clone --recursive https://github.com/JeffreyXiang/FlexGEMM.git && \
    pip install ./FlexGEMM --no-build-isolation
# o-voxel : --no-deps car cumesh/flex_gemm sont déjà installés ci-dessus
RUN git clone --recursive https://github.com/microsoft/TRELLIS.2.git && \
    pip install ./TRELLIS.2/o-voxel --no-build-isolation --no-deps

# natten (utilisé par l'upsampler NAF chargé via torch.hub)
RUN pip install natten==0.21.0 --no-build-isolation

WORKDIR /workspace/Pixal3D
RUN rm -rf /tmp/extensions

# L'interface renvoie l'URL de l'image détourée au serveur, qui doit la retélécharger :
# la protection SSRF de gradio refuse localhost/IP privées (ça ne marche que sur un Space public).
# On télécharge l'image côté navigateur et on la ré-uploade à la place.
RUN sed -i 's|handle_file(preprocessedFile)|handle_file(await (await fetch(preprocessedFile)).blob())|' index.html && \
    grep -q 'fetch(preprocessedFile)' index.html

# 1536 ne tient pas sur 16 Go de VRAM (le mode cascade alloue ~6,75 Go d'un coup) : on retire l'option.
RUN sed -i '/<option value="1536">/d' index.html && \
    ! grep -q 'option value="1536"' index.html

# En low-VRAM, chaque étape envoie un modèle sur le GPU puis le renvoie sur CPU, sans try/finally :
# après une erreur (ex. CUDA OOM) les modèles restent sur le GPU et toutes les requêtes suivantes
# échouent jusqu'au redémarrage. On enveloppe les routes API pour tout renvoyer sur CPU en cas d'erreur.
RUN python3 - <<'EOF'
import pathlib
p = pathlib.Path("app.py")
s = p.read_text()
helper = '''
import functools as _functools, gc as _gc, traceback as _traceback

def _free_vram_on_error(fn):
    @_functools.wraps(fn)
    def wrapper(*args, **kwargs):
        err = None
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            _traceback.print_exc()
            err = f"{type(e).__name__}: {e}"
        # hors du bloc except : l'exception et ses frames (qui retiennent des tenseurs GPU) sont libérées
        if LOW_VRAM and pipeline is not None:
            mods = list(pipeline.models.values()) + [pipeline.rembg_model, moge_model] + [
                getattr(pipeline, a, None) for a in (
                    "image_cond_model_ss", "image_cond_model_shape_512",
                    "image_cond_model_shape_1024", "image_cond_model_tex_1024")]
            for m in mods:
                if m is not None:
                    try:
                        m.cpu()
                    except Exception:
                        pass
        _gc.collect()
        torch.cuda.empty_cache()
        print(f"[free_vram_on_error] VRAM libérée, {torch.cuda.memory_allocated() / 1e9:.1f} Go encore alloués", flush=True)
        raise RuntimeError(err)
    return wrapper

'''
assert s.count("@app.api()\n") == 3, s.count("@app.api()\n")
first = s.index("@app.api()\n")
s = s[:first] + helper + s[first:]
s = s.replace("@app.api()\n", "@app.api()\n@_free_vram_on_error\n")
p.write_text(s)
EOF

EXPOSE 7860
CMD ["python", "app.py"]
