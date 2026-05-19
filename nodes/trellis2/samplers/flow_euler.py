from typing import *
import logging
import torch
import numpy as np
from ._easydict import EasyDict as edict
from .base import Sampler
from .classifier_free_guidance_mixin import ClassifierFreeGuidanceSamplerMixin
from .guidance_interval_mixin import GuidanceIntervalSamplerMixin

log = logging.getLogger("trellis2")


class FlowEulerSampler(Sampler):
    """
    Generate samples from a flow-matching model using Euler sampling.

    Args:
        sigma_min: The minimum scale of noise in flow.
    """
    def __init__(
        self,
        sigma_min: float,
    ):
        self.sigma_min = sigma_min

    def _eps_to_xstart(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (x_t - (self.sigma_min + (1 - self.sigma_min) * t) * eps) / (1 - t)

    def _xstart_to_eps(self, x_t, t, x_0):
        assert x_t.shape == x_0.shape
        return (x_t - (1 - t) * x_0) / (self.sigma_min + (1 - self.sigma_min) * t)

    def _v_to_xstart_eps(self, x_t, t, v):
        assert x_t.shape == v.shape
        eps = (1 - t) * v + x_t
        x_0 = (1 - self.sigma_min) * x_t - (self.sigma_min + (1 - self.sigma_min) * t) * v
        return x_0, eps
    
    def _pred_to_xstart(self, x_t, t, pred):
        return (1 - self.sigma_min) * x_t - (self.sigma_min + (1 - self.sigma_min) * t) * pred

    def _xstart_to_pred(self, x_t, t, x_0):
        return ((1 - self.sigma_min) * x_t - x_0) / (self.sigma_min + (1 - self.sigma_min) * t)

    def _inference_model(self, model, x_t, t, cond=None, **kwargs):
        t = torch.tensor([1000 * t] * x_t.shape[0], device=x_t.device, dtype=torch.float32)

        # Cast inputs to model dtype (ComfyUI-native pattern: model_base.py:183)
        model_dtype = next(model.parameters()).dtype
        if hasattr(x_t, 'replace'):
            x_t = x_t.replace(feats=x_t.feats.to(dtype=model_dtype))
        else:
            x_t = x_t.to(dtype=model_dtype)
        if cond is not None and hasattr(cond, 'to'):
            cond = cond.to(dtype=model_dtype)

        # Cast kwargs tensors to model dtype (e.g. concat_cond SparseTensor)
        for key in kwargs:
            v = kwargs[key]
            if hasattr(v, 'replace') and hasattr(v, 'feats'):
                kwargs[key] = v.replace(feats=v.feats.to(dtype=model_dtype))
            elif hasattr(v, 'dtype') and hasattr(v, 'to'):
                if v.dtype != torch.int and v.dtype != torch.long:
                    kwargs[key] = v.to(dtype=model_dtype)

        out = model(x_t, t, cond, **kwargs)

        # Cast output to fp32 for sampling loop (ComfyUI-native: model_base.py:210)
        if hasattr(out, 'replace'):
            return out.replace(feats=out.feats.float())
        return out.float()

    def _get_model_prediction(self, model, x_t, t, cond=None, **kwargs):
        pred_v = self._inference_model(model, x_t, t, cond, **kwargs)
        pred_x_0, pred_eps = self._v_to_xstart_eps(x_t=x_t, t=t, v=pred_v)
        return pred_x_0, pred_eps, pred_v

    @torch.no_grad()
    def sample_once(
        self,
        model,
        x_t,
        t: float,
        t_prev: float,
        cond: Optional[Any] = None,
        **kwargs
    ):
        """
        Sample x_{t-1} from the model using Euler method.
        
        Args:
            model: The model to sample from.
            x_t: The [N x C x ...] tensor of noisy inputs at time t.
            t: The current timestep.
            t_prev: The previous timestep.
            cond: conditional information.
            **kwargs: Additional arguments for model inference.

        Returns:
            a dict containing the following
            - 'pred_x_prev': x_{t-1}.
            - 'pred_x_0': a prediction of x_0.
        """
        pred_x_0, pred_eps, pred_v = self._get_model_prediction(model, x_t, t, cond, **kwargs)
        pred_x_prev = x_t - (t - t_prev) * pred_v
        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})

    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond: Optional[Any] = None,
        steps: int = 50,
        rescale_t: float = 1.0,
        verbose: bool = True,
        tqdm_desc: str = "Sampling",
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.

        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            verbose: If True, show a progress bar.
            tqdm_desc: A customized tqdm desc.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        import comfy.utils
        pbar = comfy.utils.ProgressBar(steps)

        sample = noise
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_seq = t_seq.tolist()
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})
        for t, t_prev in t_pairs:
            out = self.sample_once(model, sample, t, t_prev, cond, **kwargs)
            sample = out.pred_x_prev
            ret.pred_x_t.append(out.pred_x_prev)
            ret.pred_x_0.append(out.pred_x_0)
            pbar.update(1)
        ret.samples = sample
        return ret


class FlowEulerCfgSampler(ClassifierFreeGuidanceSamplerMixin, FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance.
    """
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        guidance_strength: float = 3.0,
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            guidance_strength: The strength of classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, guidance_strength=guidance_strength, **kwargs)


class FlowEulerGuidanceIntervalSampler(GuidanceIntervalSamplerMixin, ClassifierFreeGuidanceSamplerMixin, FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance and interval.
    """
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        guidance_strength: float = 3.0,
        guidance_interval: Tuple[float, float] = (0.0, 1.0),
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            guidance_strength: The strength of classifier-free guidance.
            guidance_interval: The interval for classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, guidance_strength=guidance_strength, guidance_interval=guidance_interval, **kwargs)


class FlowEulerMultiViewSampler(FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling
    with spatial blending of multiple view conditionings.

    Supports up to 6 views: front, back, left, right, top, bottom.
    At each step, runs the model once per view and blends predictions
    based on which voxels face which direction.
    """

    _VIEW_DIRS = {
        'z': {
            'front':  (0, 0, +1),
            'back':   (0, 0, -1),
            'right':  (+1, 0, 0),
            'left':   (-1, 0, 0),
            'top':    (0, +1, 0),
            'bottom': (0, -1, 0),
        },
        'x': {
            'front':  (+1, 0, 0),
            'back':   (-1, 0, 0),
            'right':  (0, 0, +1),
            'left':   (0, 0, -1),
            'top':    (0, +1, 0),
            'bottom': (0, -1, 0),
        },
    }

    def __init__(self, sigma_min: float, resolution: int):
        super().__init__(sigma_min)
        self.resolution = resolution

    def _compute_view_weights_sparse(self, coords, views, front_axis='z', blend_temperature=2.0):
        """Compute per-voxel blending weights for sparse tensors.

        Args:
            coords: [N, 4] sparse coords (batch, x, y, z)
            views: list of active view names
            front_axis: 'z' or 'x'
            blend_temperature: softmax temperature

        Returns:
            weights: [N, num_views] tensor
        """
        # Normalize coordinates to [-1, 1]
        x = (coords[:, 1].float() / self.resolution) * 2 - 1.0
        y = (coords[:, 2].float() / self.resolution) * 2 - 1.0
        z = (coords[:, 3].float() / self.resolution) * 2 - 1.0

        dirs = self._VIEW_DIRS[front_axis]
        scores = []
        for view in views:
            dx, dy, dz = dirs[view]
            score = dx * x + dy * y + dz * z
            scores.append(score)

        scores = torch.stack(scores, dim=1)  # (N, num_views)
        return torch.softmax(scores * blend_temperature, dim=1)

    def _compute_view_weights_dense(self, shape, device, views, front_axis='z', blend_temperature=2.0):
        """Compute blending weights for dense tensors (B, C, D, H, W).

        Returns:
            weights: [num_views, D, H, W] tensor
        """
        D, H, W = shape[2], shape[3], shape[4]
        # D=X, H=Y, W=Z (standard 3D tensor layout)
        dx = torch.linspace(-1, 1, D, device=device)
        dy = torch.linspace(-1, 1, H, device=device)
        dz = torch.linspace(-1, 1, W, device=device)
        grid_x, grid_y, grid_z = torch.meshgrid(dx, dy, dz, indexing='ij')

        dirs = self._VIEW_DIRS[front_axis]
        scores = []
        for view in views:
            vx, vy, vz = dirs[view]
            score = vx * grid_x + vy * grid_y + vz * grid_z
            scores.append(score)

        scores = torch.stack(scores, dim=0)  # (num_views, D, H, W)
        return torch.softmax(scores * blend_temperature, dim=0)

    @torch.no_grad()
    def sample_once(
        self,
        model,
        x_t,
        t: float,
        t_prev: float,
        conds: Dict[str, Any],
        views: List[str],
        front_axis: str = 'z',
        blend_temperature: float = 2.0,
        **kwargs
    ):
        is_sparse = hasattr(x_t, 'coords')

        if is_sparse:
            weights = self._compute_view_weights_sparse(
                x_t.coords, views, front_axis, blend_temperature)
        else:
            weights = self._compute_view_weights_dense(
                x_t.shape, x_t.device, views, front_axis, blend_temperature)

        pred_v_accum = 0
        for i, view in enumerate(views):
            cond = conds[view]
            # _inference_model goes through mixin chain (CFG, interval)
            if isinstance(cond, dict) and 'cond' in cond:
                pred_v_view = self._inference_model(
                    model, x_t, t, **cond, **kwargs)
            else:
                pred_v_view = self._inference_model(
                    model, x_t, t, cond=cond, **kwargs)

            if is_sparse:
                w = weights[:, i].unsqueeze(1)
                v_feats = pred_v_view.feats if hasattr(pred_v_view, 'feats') else pred_v_view
                pred_v_accum = pred_v_accum + v_feats * w
            else:
                w = weights[i].unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
                pred_v_accum = pred_v_accum + pred_v_view * w

        if is_sparse:
            pred_v = x_t.replace(feats=pred_v_accum)
        else:
            pred_v = pred_v_accum

        pred_x_0, pred_eps = self._v_to_xstart_eps(x_t=x_t, t=t, v=pred_v)
        pred_x_prev = x_t - (t - t_prev) * pred_v
        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})

    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        conds: Dict[str, Any],
        views: List[str],
        steps: int = 50,
        rescale_t: float = 1.0,
        verbose: bool = True,
        tqdm_desc: str = "Sampling (multi-view)",
        front_axis: str = 'z',
        blend_temperature: float = 2.0,
        **kwargs
    ):
        import comfy.utils
        pbar = comfy.utils.ProgressBar(steps)

        sample = noise
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_seq = t_seq.tolist()
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})

        for t, t_prev in t_pairs:
            out = self.sample_once(
                model, sample, t, t_prev,
                conds=conds, views=views,
                front_axis=front_axis,
                blend_temperature=blend_temperature,
                **kwargs,
            )
            sample = out.pred_x_prev
            ret.pred_x_t.append(out.pred_x_prev)
            ret.pred_x_0.append(out.pred_x_0)
            pbar.update(1)
        ret.samples = sample
        return ret


class FlowEulerMultiViewGuidanceIntervalSampler(
    GuidanceIntervalSamplerMixin,
    ClassifierFreeGuidanceSamplerMixin,
    FlowEulerMultiViewSampler,
):
    """Multi-view Euler sampler with CFG and guidance interval."""

    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        conds: Dict[str, Any],
        views: List[str],
        steps: int = 50,
        rescale_t: float = 1.0,
        guidance_strength: float = 3.0,
        guidance_interval: Tuple[float, float] = (0.0, 1.0),
        verbose: bool = True,
        front_axis: str = 'z',
        blend_temperature: float = 2.0,
        **kwargs
    ):
        return super().sample(
            model, noise, conds=conds, views=views,
            steps=steps, rescale_t=rescale_t, verbose=verbose,
            guidance_strength=guidance_strength,
            guidance_interval=guidance_interval,
            front_axis=front_axis,
            blend_temperature=blend_temperature,
            **kwargs,
        )
