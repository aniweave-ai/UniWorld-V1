import shutil
from pathlib import Path
import argparse
import torch
from transformers import AutoProcessor, BitsAndBytesConfig
import sys

sys.path.append("../")
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
    Load FP16 LVLM + projectors, quantize to NF4, and save all weights + processor + JSON configs.

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

    # Copy all JSON files from original FP16 folder
    for json_file in Path(pretrained_lvlm_path).glob("*.json"):
        shutil.copy(json_file, temp_fp16_dir)

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

    # Copy all JSON files from original FP16 folder
    for json_file in Path(pretrained_lvlm_path).glob("*.json"):
        dest_file = Path(output_path) / json_file.name
        if not dest_file.exists():
            shutil.copy(json_file, dest_file)
    # Optionally cleanup temp folder
    shutil.rmtree(temp_fp16_dir)

    # Cleanup
    del lvlm_model_nf4
    torch.cuda.empty_cache()

    print(f"NF4 model with projectors saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize LVLM to NF4 with projectors")

    parser.add_argument(
        "--pretrained_lvlm_path", type=str, required=True,
        help="Path to original FP16 LVLM"
    )
    parser.add_argument(
        "--output_path", type=str, required=True,
        help="Path to save NF4 quantized model"
    )
    parser.add_argument(
        "--denoise_projector_path", type=str, default=None,
        help="Optional path to denoise projector weights"
    )
    parser.add_argument(
        "--siglip_projector_path", type=str, default=None,
        help="Optional path to siglip projector weights"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cuda", "cpu"],
        help="Device to use for loading (default: cuda)"
    )

    args = parser.parse_args()

    save_lvlm_nf4_with_projectors(
        pretrained_lvlm_path=args.pretrained_lvlm_path,
        output_path=args.output_path,
        denoise_projector_path=args.denoise_projector_path,
        siglip_projector_path=args.siglip_projector_path,
        device=args.device,
    )

    """
    Example Usage:
    
    python save_nf4_model.py \
        --pretrained_lvlm_path ./checkpoints/qwen2p5vl_fp16 \
        --denoise_projector_path ./projectors/denoise.pt \
        --siglip_projector_path ./projectors/siglip.pt \
        --output_path ./checkpoints/qwen2p5vl_nf4 \
        --device cuda
    """
