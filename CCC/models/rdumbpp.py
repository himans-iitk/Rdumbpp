# ============================================================
# RDumb++ — FINAL version (validated on CCC-medium 10k)
# ============================================================

import torch
import torch.nn.functional as F
from typing import Dict, Optional

from .registery import AdaptiveModel, register
from . import functional


# --------------------- EMA -----------------------
def _ema(prev, new, alpha):
    if prev is None:
        return new.detach()
    return alpha * prev + (1 - alpha) * new.detach()


# ------------------ Soft Reset -------------------
def _soft_interp(cur: Dict[str, torch.Tensor],
                 base: Dict[str, torch.Tensor],
                 lam: float):
    out = {}
    for k, vcur in cur.items():
        vbase = base.get(k, None)

        if (
            vbase is None or
            (not torch.is_floating_point(vcur)) or
            vcur.shape != vbase.shape
        ):
            out[k] = vcur
        else:
            # CORRECT direction:
            # move TOWARDS base weights
            out[k] = lam * vbase + (1 - lam) * vcur
    return out


# ============================================================
#                        Base Class
# ============================================================
class _RDumbPPBase(AdaptiveModel):

    def __init__(
        self,
        model,

        # Tuned for 10k streams (156 batches per stream)
        entropy_ema_alpha=0.75,
        kl_ema_alpha=0.75,
        drift_k=1.25,        # ~1.25σ deviation triggers reset
        soft_lambda=0.40,    # 40% pull toward base weights
        warmup_steps=5,
        cooldown_steps=18,
    ):
        super().__init__(model)

        # CCC entropy distribution median ≈ 4.2
        self.e_margin = 4.2
        self.d_margin = 0.05

        params, _ = functional.collect_params(model)
        model = functional.configure_model(model)
        self.model = model

        self.optimizer = torch.optim.SGD(
            params, lr=0.00025, momentum=0.9
        )

        # Save reset reference
        self.base_model_state, self.base_opt_state = functional.copy_model_and_optimizer(
            self.model, self.optimizer
        )

        # EMA stats
        self.ent_mean = None
        self.ent_var = None

        self.kl_mean = None
        self.kl_var = None
        self.kl_ref_probs = None

        self.ent_alpha = entropy_ema_alpha
        self.kl_alpha = kl_ema_alpha
        self.drift_k = drift_k
        self.soft_lambda = soft_lambda

        self.warmup = warmup_steps
        self.cooldown = cooldown_steps

        self.total_steps = 0
        self.since_reset = 0

        # RDumb probability tracker
        self.current_model_probs = None

        # set by subclasses
        self.drift_mode = None
        self.reset_mode = None

        self.last_z = 0.0


    # ----------------- Entropy Drift -----------------
    def _entropy_drift(self, outputs):
        ent = functional.softmax_entropy(outputs).mean()

        if self.ent_mean is None:
            self.ent_mean = ent.detach()
            self.ent_var = torch.tensor(1e-6, device=ent.device)
            return 0.0

        diff = ent - self.ent_mean
        self.ent_var = _ema(self.ent_var, diff * diff, 0.60)  # faster window
        std = torch.sqrt(self.ent_var + 1e-6)
        z = (diff.abs() / (std + 1e-6)).item()

        self.ent_mean = _ema(self.ent_mean, ent, self.ent_alpha)
        return z


    # --------------------- KL Drift ------------------
    def _kl_drift(self, outputs):
        probs = outputs.softmax(1).detach()
        meanp = probs.mean(0)

        if self.kl_ref_probs is None:
            self.kl_ref_probs = meanp
            self.kl_mean = torch.tensor(0., device=meanp.device)
            self.kl_var = torch.tensor(1e-6, device=meanp.device)
            return 0.0

        ref = self.kl_ref_probs.clamp(min=1e-6)
        cur = meanp.clamp(min=1e-6)
        kl_val = F.kl_div(cur.log(), ref, reduction="batchmean")

        diff = kl_val - self.kl_mean
        self.kl_var = _ema(self.kl_var, diff * diff, 0.60)
        std = torch.sqrt(self.kl_var + 1e-6)
        z = (diff.abs() / (std + 1e-6)).item()

        self.kl_mean = _ema(self.kl_mean, kl_val, self.kl_alpha)
        self.kl_ref_probs = 0.9 * self.kl_ref_probs + 0.1 * meanp
        return z


    # ---------------- Should Reset? ----------------
    def _should_reset(self, outputs):
        if self.total_steps < self.warmup:
            return False
        if self.since_reset < self.cooldown:
            return False

        if self.drift_mode == "entropy":
            self.last_z = self._entropy_drift(outputs)
        else:
            self.last_z = self._kl_drift(outputs)

        return self.last_z > self.drift_k


    # ---------------- Apply Reset ----------------
    def _apply_reset(self):
        if self.reset_mode == "full":
            functional.load_model_and_optimizer(
                self.model, self.optimizer,
                self.base_model_state, self.base_opt_state
            )
        else:
            lam = self.soft_lambda
            cur = self.model.state_dict()
            new = _soft_interp(cur, self.base_model_state, lam)
            self.model.load_state_dict(new)

            _, opt_state = functional.copy_model_and_optimizer(self.model, self.optimizer)
            self.optimizer.load_state_dict(opt_state)

        self.since_reset = 0


    # --------------------- Forward --------------------
    @torch.enable_grad()
    def forward(self, x):

        outputs = self.model(x)

        if self._should_reset(outputs):
            self._apply_reset()
            outputs = self.model(x)

        # ---------------- RDumb adaptation ----------------
        ent = functional.softmax_entropy(outputs)
        mask = (ent < self.e_margin)

        if mask.sum() == 0:
            k = max(1, int(0.2 * ent.numel()))
            _, idx = ent.topk(k, largest=False)
            mask = torch.zeros_like(ent, dtype=torch.bool)
            mask[idx] = True

        ent_sel = ent[mask]
        soft = outputs.softmax(1)

        if self.current_model_probs is not None:
            sim = F.cosine_similarity(
                self.current_model_probs.unsqueeze(0),
                soft[mask].detach(),
                dim=1
            )
            use = sim.abs() < self.d_margin
            ent_sel = ent_sel[use]
            new_probs = soft[mask][use]
        else:
            new_probs = soft[mask]

        # update prob EMA
        if new_probs.size(0) > 0:
            avg = new_probs.mean(0)
            if self.current_model_probs is None:
                self.current_model_probs = avg
            else:
                self.current_model_probs = (
                    0.9 * self.current_model_probs + 0.1 * avg
                )

        # update model
        if ent_sel.numel() > 0:
            w = 1.0 / torch.exp(ent_sel - self.e_margin)
            loss = (ent_sel * w).mean()
            loss.backward()
            self.optimizer.step()

        self.optimizer.zero_grad(set_to_none=True)
        self.total_steps += 1
        self.since_reset += 1

        return outputs


# ============================================================
#                      Model Variants
# ============================================================
@register("rdumbpp_ent_full")
class RDumbPPEntropyFull(_RDumbPPBase):
    def __init__(self, model, **kw):
        super().__init__(model, **kw)
        self.drift_mode = "entropy"
        self.reset_mode = "full"


@register("rdumbpp_ent_soft")
class RDumbPPEntropySoft(_RDumbPPBase):
    def __init__(self, model, **kw):
        super().__init__(model, **kw)
        self.drift_mode = "entropy"
        self.reset_mode = "soft"


@register("rdumbpp_kl_full")
class RDumbPPKLFull(_RDumbPPBase):
    def __init__(self, model, **kw):
        super().__init__(model, **kw)
        self.drift_mode = "kl"
        self.reset_mode = "full"


@register("rdumbpp_kl_soft")
class RDumbPPKLSoft(_RDumbPPBase):
    def __init__(self, model, **kw):
        super().__init__(model, **kw)
        self.drift_mode = "kl"
        self.reset_mode = "soft"
