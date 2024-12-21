import logging
from argparse import ArgumentParser
from pathlib import Path

import torch
import torchaudio

from mmaudio.eval_utils import (
    ModelConfig,
    all_model_cfg,
    generate,
    load_video,
    make_video,
    setup_eval_logging
)
from mmaudio.model.flow_matching import FlowMatching
from mmaudio.model.networks import MMAudio, get_my_mmaudio
from mmaudio.model.utils.features_utils import FeaturesUtils

log = logging.getLogger()

@torch.inference_mode()
def main():
    setup_eval_logging()

    parser = ArgumentParser()
    parser.add_argument('--variant',
                        type=str,
                        default='large_44k_v2',
                        help='Model variant: small_16k, small_44k, medium_44k, large_44k, large_44k_v2')
    parser.add_argument('--video', type=Path, help='Path to the video file')
    parser.add_argument('--prompt', type=str, help='Input prompt', default='')
    parser.add_argument('--negative_prompt', type=str, help='Negative prompt', default='')
    parser.add_argument('--duration', type=float, default=8.0)
    parser.add_argument('--cfg_strength', type=float, default=4.5)
    parser.add_argument('--num_steps', type=int, default=25)
    parser.add_argument('--mask_away_clip', action='store_true')
    parser.add_argument('--output', type=Path, help='Output directory', default='./output')
    parser.add_argument('--seed', type=int, help='Random seed', default=42)
    parser.add_argument('--skip_video_composite', action='store_true')
    parser.add_argument('--full_precision', action='store_true')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run inference on (cuda or cpu)')

    args = parser.parse_args()

    # Validate the model variant
    if args.variant not in all_model_cfg:
        raise ValueError(f'Unknown model variant: {args.variant}')
    model: ModelConfig = all_model_cfg[args.variant]

    # Device setup
    device = args.device.lower()
    if device not in ('cuda', 'cpu'):
        log.warning(f"Unrecognized device '{device}' - defaulting to CPU")
        device = 'cpu'

    # Allow TF32 only if on CUDA
    if device == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    dtype = torch.float32 if args.full_precision else torch.bfloat16

    # We no longer call model.download_if_needed() here.
    # Assume weights are pre-downloaded or cached via the Docker build steps.

    # Setup output directory
    output_dir: Path = args.output.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare video path if provided
    if args.video:
        video_path: Path = args.video.expanduser()
    else:
        video_path = None

    prompt: str = args.prompt
    negative_prompt: str = args.negative_prompt
    seed: int = args.seed
    num_steps: int = args.num_steps
    duration: float = args.duration
    cfg_strength: float = args.cfg_strength
    skip_video_composite: bool = args.skip_video_composite
    mask_away_clip: bool = args.mask_away_clip

    seq_cfg = model.seq_cfg

    # Load the model
    net: MMAudio = get_my_mmaudio(model.model_name).eval()

    # Map checkpoint to CPU or CUDA
    map_location = torch.device(device)
    net.load_weights(
        torch.load(model.model_path, map_location=map_location),
        weights_only=True
    )

    net = net.to(device, dtype)
    log.info(f'Loaded weights from {model.model_path} onto {device}')

    # FlowMatching setup
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    fm = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=num_steps)

    feature_utils = FeaturesUtils(
        tod_vae_ckpt=model.vae_path,
        synchformer_ckpt=model.synchformer_ckpt,
        enable_conditions=True,
        mode=model.mode,
        bigvgan_vocoder_ckpt=model.bigvgan_16k_path,
        need_vae_encoder=False
    ).to(device, dtype).eval()

    # Load video if present
    if video_path is not None:
        log.info(f'Using video {video_path}')
        video_info = load_video(video_path, duration)
        clip_frames = video_info.clip_frames
        sync_frames = video_info.sync_frames
        duration = video_info.duration_sec

        if mask_away_clip:
            clip_frames = None
        else:
            clip_frames = clip_frames.unsqueeze(0)

        sync_frames = sync_frames.unsqueeze(0)
    else:
        log.info('No video provided -- text-to-audio mode')
        clip_frames = sync_frames = None

    seq_cfg.duration = duration
    net.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)

    log.info(f'Prompt: {prompt}')
    log.info(f'Negative prompt: {negative_prompt}')

    # Generate audio
    audios = generate(
        clip_frames,
        sync_frames,
        [prompt],
        negative_text=[negative_prompt],
        feature_utils=feature_utils,
        net=net,
        fm=fm,
        rng=rng,
        cfg_strength=cfg_strength
    )
    audio = audios.float().cpu()[0]

    # Save audio
    if video_path is not None:
        save_path = output_dir / f'{video_path.stem}.flac'
    else:
        safe_filename = prompt.replace(' ', '_').replace('/', '_').replace('.', '')
        save_path = output_dir / f'{safe_filename}.flac'

    torchaudio.save(save_path, audio, seq_cfg.sampling_rate)
    log.info(f'Audio saved to {save_path}')

    # Optionally make a new video with the generated audio
    if video_path is not None and not skip_video_composite:
        video_save_path = output_dir / f'{video_path.stem}.mp4'
        make_video(video_info, video_save_path, audio, sampling_rate=seq_cfg.sampling_rate)
        log.info(f'Video saved to {video_save_path}')

    # Log memory usage if on CUDA
    if device == 'cuda':
        log.info('Memory usage: %.2f GB', torch.cuda.max_memory_allocated() / (2**30))
    else:
        log.info('Ran on CPU, no CUDA memory usage to report.')


if __name__ == '__main__':
    main()
