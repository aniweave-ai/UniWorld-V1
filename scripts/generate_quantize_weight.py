import shutil
from pathlib import Path

import torch
from transformers import AutoProcessor, BitsAndBytesConfig

from univa.models.qwen2p5vl.modeling_univa_qwen2p5vl import (
    UnivaQwen2p5VLForConditionalGeneration,
)


def save_lvlm_nf4_with_projectors(
    pretrained_lvlm_path: str,
    output_path: str,
    denoise_projector_path: str = None,
    siglip_projector_path: str = None,
    device: str = "cuda",
):
    """
    Load FP16 LVLM + projectors, quantize to NF4, and save all weights + processor.

    Args:
        pretrained_lvlm_path: folder of original FP16 LVLM
        output_path: folder to save NF4 model
        denoise_projector_path: path to denoise projector weights (optional)
        siglip_projector_path: path to siglip projector weights (optional)
        device: "cuda" or "cpu"
    """

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # ---------------- Step 1: Load FP16 model ----------------
    lvlm_model = UnivaQwen2p5VLForConditionalGeneration.from_pretrained(
        pretrained_lvlm_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        quantization_config=None,
    ).to(device)

    # ---------------- Step 2: Load projector weights ----------------
    if denoise_projector_path:
        denoise_proj = torch.load(denoise_projector_path, map_location=device)
        msg = lvlm_model.load_state_dict(denoise_proj, strict=False)
        missing = msg[1]
        assert len(missing) == 0, f"Missing keys in denoise projector: {missing}"

    if siglip_projector_path:
        siglip_proj = torch.load(siglip_projector_path, map_location=device)
        msg = lvlm_model.load_state_dict(siglip_proj, strict=False)
        missing = msg[1]
        assert len(missing) == 0, f"Missing keys in siglip projector: {missing}"

    # ---------------- Step 3: Save temporary FP16 model with projectors ----------------
    temp_fp16_dir = Path(pretrained_lvlm_path + "_with_projectors_fp16")
    if temp_fp16_dir.exists():
        shutil.rmtree(temp_fp16_dir)
    temp_fp16_dir.mkdir(parents=True, exist_ok=True)

    lvlm_model.save_pretrained(temp_fp16_dir)

    # Save processor
    processor = AutoProcessor.from_pretrained(pretrained_lvlm_path)
    processor.save_pretrained(temp_fp16_dir)

    # free memory
    del lvlm_model
    torch.cuda.empty_cache()

    # ---------------- Step 4: Reload with NF4 quantization ----------------
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )

    lvlm_model_nf4 = UnivaQwen2p5VLForConditionalGeneration.from_pretrained(
        temp_fp16_dir,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )

    # ---------------- Step 5: Save final NF4 model ----------------
    lvlm_model_nf4.save_pretrained(output_path)

    # Optionally cleanup temp folder
    shutil.rmtree(temp_fp16_dir)

    # Cleanup
    del lvlm_model_nf4
    torch.cuda.empty_cache()

    print(f"NF4 model with projectors saved to {output_path}")
