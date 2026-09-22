@echo off
REM ==============================================================================
REM Download VLA Model Checkpoints using official Hugging Face CLI (hf download)
REM Models: SmolVLA, Octo-Small (PyTorch + Berkeley Flax), Google RT-1
REM ==============================================================================

echo [1/4] Downloading SmolVLA (lerobot/smolvla_base)...
hf download lerobot/smolvla_base --local-dir checkpoints/smolvla_base --max-workers 1

echo [2/4] Downloading Octo PyTorch Safetensors (lilkm/octo-pt-small-1.5)...
hf download lilkm/octo-pt-small-1.5 --local-dir checkpoints/octo_pt_small --max-workers 1

echo [3/4] Downloading Octo Berkeley Flax (rail-berkeley/octo-small-1.5)...
hf download rail-berkeley/octo-small-1.5 --local-dir checkpoints/octo_small --max-workers 1

echo [4/4] Downloading Google RT-1 (PommesPeter/rt_1_tf_trained_for_000400120)...
hf download PommesPeter/rt_1_tf_trained_for_000400120 --local-dir checkpoints/rt_1 --max-workers 1

echo ==============================================================================
echo All VLA models downloaded successfully into the checkpoints/ directory!
echo ==============================================================================
