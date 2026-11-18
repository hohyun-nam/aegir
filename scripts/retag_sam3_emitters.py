"""Retag an initialized scene using visible SAM mask contribution across views."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aegir.data import MultiViewSceneDataset
from aegir.gaussian_scene import GaussianMaterialBackend, surfel_view_matrix
from aegir.math_utils import quat_to_matrix
from aegir.provenance import sha256_file, validate_sam3_ldr_view, write_json


def mask_contributions(scene, camera, mask, valid):
    # For fixed geometry RGB = sum_i (T_i alpha_i) color_i. The color
    # derivative therefore sums the actual visible contribution, not just
    # projected centers or unoccluded Gaussian footprints.
    from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    device = scene.means.device
    c2w = camera['c2w'].to(device)
    w, h = int(camera['width']), int(camera['height'])
    fx, fy = float(camera['fx']), float(camera['fy'])
    view = surfel_view_matrix(c2w)
    proj = torch.zeros((4, 4), device=device)
    proj[0, 0], proj[1, 1] = 2 * fx / w, 2 * fy / h
    proj[0, 2] = (2 * float(camera['cx']) - (w - 1)) / w
    proj[1, 2] = (2 * float(camera['cy']) - (h - 1)) / h
    proj[2, 2], proj[2, 3], proj[3, 2] = 100 / 99.99, -1 / 99.99, 1
    settings = GaussianRasterizationSettings(
        image_height=h, image_width=w, tanfovx=w/(2*fx), tanfovy=h/(2*fy),
        bg=torch.zeros(3, device=device), scale_modifier=1.,
        viewmatrix=view.T.contiguous(), projmatrix=(proj @ view).T.contiguous(),
        sh_degree=0, campos=c2w[:3, 3], prefiltered=False, debug=False,
    )
    colors = torch.zeros_like(scene.means, requires_grad=True)
    rgb, _, _ = GaussianRasterizer(settings)(
        means3D=scene.means.detach(), means2D=torch.zeros_like(scene.means),
        colors_precomp=colors, opacities=scene.opacities.detach(),
        scales=scene.scales.detach(), rotations=scene.quats.detach(),
    )
    valid = valid.to(device).squeeze(-1).float()
    mask = mask.to(device).squeeze(-1).float() * valid
    gradient, = torch.autograd.grad((rgb[0]*mask + rgb[1]*valid).sum(), colors)
    return gradient[:, 0].clamp_min(0), gradient[:, 1].clamp_min(0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--image-size', type=int, default=512)
    p.add_argument('--image-width', type=int)
    p.add_argument('--image-height', type=int)
    p.add_argument('--min-views', type=int, default=3)
    p.add_argument('--view-support', type=float, default=.6)
    p.add_argument('--consistency', type=float, default=.7)
    p.add_argument('--min-contribution', type=float, default=.25)
    args = p.parse_args()
    if (args.image_width is None) != (args.image_height is None):
        p.error('image-width and image-height must be specified together')
    if args.image_width is not None:
        from aegir.data import image_dimensions
        image_dimensions((args.image_height, args.image_width))
        args.image_size = (args.image_height, args.image_width)
    if args.min_views < 1 or args.min_contribution <= 0 or not all(
        0 < x <= 1 for x in (args.view_support, args.consistency)
    ):
        p.error('invalid voting thresholds')
    if args.output.exists():
        raise FileExistsError(f'Choose a fresh output directory: {args.output}')
    validate_sam3_ldr_view(args.data)
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    scene = GaussianMaterialBackend.from_state(payload['source_dnsplatter_checkpoint'], payload['scene_state']).cuda()
    scene.requires_grad_(False)
    dataset = MultiViewSceneDataset(args.data, payload['source_dnsplatter_checkpoint'],
                                   image_size=args.image_size, split='train')
    if dataset.metadata.get('emitter_mask_source') != 'sam3':
        raise ValueError('Requires explicit SAM3 masks')
    observed = torch.zeros(scene.gaussian_count, device='cuda', dtype=torch.long)
    positive = torch.zeros_like(observed)
    for i, frame in enumerate(dataset.frames):
        if frame.emitter_mask is None:
            raise ValueError('Missing SAM mask')
        numerator, denominator = mask_contributions(scene, frame.camera, frame.emitter_mask, frame.mask)
        visible = denominator >= args.min_contribution
        observed += visible
        positive += visible & (numerator / denominator.clamp_min(1e-8) >= args.view_support)
        if i % 25 == 0:
            print(f'Voting {i+1}/{len(dataset.frames)}', flush=True)
    accepted = (observed >= args.min_views) & (positive >= args.min_views) & (
        positive.float() / observed.clamp_min(1) >= args.consistency)
    # Only identity matching uses the recovered emitter geometry. It never
    # rejects a SAM-confirmed row for lying outside the analytic ellipsoid.
    state = payload['emitter_state']
    centers = state['centers'].cuda()
    rotations = quat_to_matrix(state['quaternions'].cuda())
    scales = state['scales'].cuda()
    if len(centers) == 0:
        raise ValueError('No initialized emitter identities')
    if scales.shape[1] == 2:
        scales = torch.cat((scales, scales.amin(-1, keepdim=True)*.1), -1)
    delta = scene.means[accepted, None] - centers[None]
    local = torch.einsum('gei,eij->gej', delta, rotations)
    distance = (local / scales[None].clamp_min(.01)).square().sum(-1)
    ids = torch.full_like(observed, -1)
    ids[accepted] = distance.argmin(-1)
    payload['scene_state']['emitter_ids'] = ids.cpu()
    payload['scene_state']['albedo_logits'][accepted.cpu()] = 0. # gray 0.5
    # This is a derived initialization artifact, not an exact optimizer resume.
    payload.pop('training_state', None)
    summary = dict(method='sam3_visible_contribution_multiview', views=len(dataset.frames),
                   min_views=args.min_views, view_support=args.view_support,
                   consistency=args.consistency, min_contribution=args.min_contribution,
                   tagged=int(accepted.sum()), previous_tagged=int((scene.emitter_ids >= 0).sum()),
                   counts=[int((ids == i).sum()) for i in range(len(centers))],
                   identity_method='nearest_normalized_analytic_support_after_sam_vote',
                   source_checkpoint=str(args.checkpoint.resolve()), off_albedo=.5,
                   source_checkpoint_sha256=sha256_file(args.checkpoint),
                   data=str(args.data.resolve()),
                   data_metadata_sha256=sha256_file(args.data / 'transforms.json'),
                   exact_resume_supported=False)
    payload['sam3_retagging'] = summary
    # Replace the preliminary overlap diagnostic with the accepted SAM votes.
    payload.setdefault('dataset', {})['emitter_gaussian_association'] = summary
    args.output.mkdir(parents=True)
    checkpoint = args.output / 'aegir_retagged.pt'
    torch.save(payload, checkpoint)
    torch.save(dict(observed=observed.cpu(), positive=positive.cpu(), emitter_ids=ids.cpu()), args.output/'votes.pt')
    write_json(args.output/'tagging.json', summary)
    exporter = Path(__file__).with_name('export_scene_supersplat.py')
    for name, flags in [('map_with_initial_emitters.ply', []),
                        ('map_with_emitter_tagged_gaussians_green.ply', ['--visualize-emitter-tags', '--omit-analytic-emitters'])]:
        subprocess.run([sys.executable, str(exporter), '--checkpoint', str(checkpoint),
                        '--output', str(args.output/name), *flags], check=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
