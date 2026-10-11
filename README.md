# pixal3D

Image Docker pour faire tourner [Pixal3D](https://github.com/TencentARC/Pixal3D) (TencentARC, image → mesh 3D GLB)
sur un GPU NVIDIA 16 Go (testé sur RTX 3080 Ti Laptop, sm_86, 31 Go de RAM).

Environnement : CUDA 12.4, Python 3.10, torch 2.6.0 (verrouillé), extensions TRELLIS.2
(cumesh, o-voxel, FlexGEMM, nvdiffrast, nvdiffrec) et natten compilées pour sm_86.

## Prérequis

- Docker + NVIDIA Container Toolkit
- Un compte Hugging Face ayant accepté la licence de [briaai/RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0)
  (modèle de détourage, accès restreint)
- Un token Hugging Face de type **Read** (ou fine-grained avec l'accès aux dépôts « gated »)

## Lancement

```bash
echo "HF_TOKEN=hf_xxx" > .env      # ne jamais committer ce fichier
docker compose up -d --build       # premier build : ~30 min (compilation des extensions CUDA)
docker logs -f pixal3d_app         # attendre "Running on local URL"
```

Interface web : http://localhost:7860 (depuis un autre poste : `ssh -L 7860:localhost:7860 ...`).

## Notes

- **Résolution** : rester en 1024. Le mode 1536 (cascade) demande ~16 Go de VRAM et plante sur une carte 16 Go ;
  l'option est retirée de l'interface.
- **Mémoire** : en mode low-VRAM les modèles restent en RAM (~21 Go). Les variables `MALLOC_*` du compose
  évitent que la RAM gonfle jusqu'à l'OOM.
- **VRAM partagée** : fermer Blender ou tout autre logiciel GPU pendant les générations.
- L'app d'origine lance aussi un lien public `*.gradio.live` (`share=True`).
