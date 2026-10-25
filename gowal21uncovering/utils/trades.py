import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from core.utils import ctx_noparamgrad_and_eval

from core.metrics import accuracy
from core.utils import SmoothCrossEntropyLoss
from core.utils import track_bn_stats
import random
from typing import List


def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened**2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()


def _kl_div(logit1, logit2):
    return F.kl_div(F.log_softmax(logit1, dim=1), F.softmax(logit2, dim=1), reduction="batchmean")


def _jensen_shannon_div(logit1, logit2, T=1.0):
    prob1 = F.softmax(logit1 / T, dim=1)
    prob2 = F.softmax(logit2 / T, dim=1)
    mean_prob = 0.5 * (prob1 + prob2)

    logsoftmax = torch.log(mean_prob.clamp(min=1e-8))
    jsd = F.kl_div(logsoftmax, prob1, reduction="batchmean")
    jsd += F.kl_div(logsoftmax, prob2, reduction="batchmean")
    return jsd * 0.5


def trades_loss(
    model,
    x_natural,
    y,
    optimizer,
    step_size=0.003,
    epsilon=0.031,
    perturb_steps=10,
    beta=1.0,
    attack="linf-pgd",
    label_smoothing=0.1,
    use_cutmix=False,
    use_consistency=False,
    cons_lambda=0.0,
    cons_tem=0.0,
):
    """
    TRADES training (Zhang et al, 2019).
    """

    criterion_ce = SmoothCrossEntropyLoss(reduction="mean", smoothing=label_smoothing)
    criterion_kl = nn.KLDivLoss(reduction="batchmean")
    model.train()
    track_bn_stats(model, False)
    batch_size = len(x_natural)

    x_adv = x_natural.detach() + torch.FloatTensor(x_natural.shape).uniform_(-epsilon, epsilon).cuda().detach()
    x_adv = torch.clamp(x_adv, 0.0, 1.0)

    if use_cutmix:  # CutMix
        p_natural = y
    else:
        p_natural = F.softmax(model(x_natural), dim=1)
        p_natural = p_natural.detach()

    if attack == "linf-pgd":
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1), p_natural)
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

    elif attack == "l2-pgd":
        delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1), p_natural)
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        raise ValueError(f"Attack={attack} not supported for TRADES training!")
    model.train()
    track_bn_stats(model, True)

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)

    optimizer.zero_grad()
    # calculate robust loss
    logits_natural = model(x_natural)
    logits_adv = model(x_adv)

    if use_cutmix:  # CutMix
        loss_natural = criterion_kl(F.log_softmax(logits_natural, dim=1), y)
    else:
        loss_natural = criterion_ce(logits_natural, y)

    loss_robust = criterion_kl(F.log_softmax(logits_adv, dim=1), F.softmax(logits_natural, dim=1))
    loss = loss_natural + beta * loss_robust

    if use_consistency:
        logits_adv1, logits_adv2 = logits_adv.chunk(2)
        loss = loss + cons_lambda * _jensen_shannon_div(logits_adv1, logits_adv2, cons_tem)

    if use_cutmix:  # CutMix
        batch_metrics = {"loss": loss.item(), "clean_acc": -1, "adversarial_acc": -1}
    else:
        batch_metrics = {
            "loss": loss.item(),
            "clean_acc": accuracy(y, logits_natural.detach()),
            "adversarial_acc": accuracy(y, logits_adv.detach()),
        }

    return loss, batch_metrics


def trades_loss_LSE(
    model,
    x_natural,
    y,
    optimizer,
    step_size=0.003,
    epsilon=0.031,
    perturb_steps=10,
    beta=1.0,
    attack="linf-pgd",
    label_smoothing=0.1,
    clip_value=0,
    use_cutmix=False,
    num_classes=10,
):
    """
    SCORE training (Ours).
    """
    # criterion_ce = SmoothCrossEntropyLoss(reduction='mean', smoothing=label_smoothing)
    # criterion_kl = nn.KLDivLoss(reduction='sum')
    model.train()
    track_bn_stats(model, False)
    batch_size = len(x_natural)

    x_adv = x_natural.detach() + torch.FloatTensor(x_natural.shape).uniform_(-epsilon, epsilon).cuda().detach()
    x_adv = torch.clamp(x_adv, 0.0, 1.0)

    if use_cutmix:  # CutMix
        p_natural = y
    else:
        p_natural = F.softmax(model(x_natural), dim=1)
        p_natural = p_natural.detach()

    if attack == "linf-pgd":
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            output_adv = F.softmax(model(x_adv), dim=1)
            with torch.enable_grad():
                loss_lse = torch.sum((output_adv - p_natural) ** 2)
            grad = torch.autograd.grad(loss_lse, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif attack == "l2-pgd":
        delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                output_adv = F.softmax(model(adv), dim=1)
                loss = (-1) * torch.sum((output_adv - p_natural) ** 2)
            loss.backward()

            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        raise ValueError(f"Attack={attack} not supported for TRADES training!")
    model.train()
    track_bn_stats(model, True)

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)

    optimizer.zero_grad()

    if use_cutmix:  # CutMix
        y_onehot = y
    else:
        y_onehot = (1 - num_classes * label_smoothing / (num_classes - 1)) * F.one_hot(
            y, num_classes=num_classes
        ) + label_smoothing / (num_classes - 1)

    logits_natural = F.softmax(model(x_natural), dim=1)
    logits_adv = F.softmax(model(x_adv), dim=1)
    loss_natural = torch.sum((logits_natural - y_onehot) ** 2, dim=-1)
    loss_robust = torch.sum((logits_adv - logits_natural) ** 2, dim=-1)

    loss_robust = F.relu(loss_robust - clip_value)  # clip loss value

    loss = loss_natural.mean() + beta * loss_robust.mean()

    if use_cutmix:  # CutMix
        batch_metrics = {"loss": loss.item(), "clean_acc": -1, "adversarial_acc": -1}
    else:
        batch_metrics = {
            "loss": loss.item(),
            "clean_acc": accuracy(y, logits_natural.detach()),
            "adversarial_acc": accuracy(y, logits_adv.detach()),
        }

    return loss, batch_metrics


def trades_loss_sur(
    model,
    sur_model,
    x_natural,
    y,
    optimizer,
    step_size=0.003,
    epsilon=0.031,
    perturb_steps=10,
    beta=1.0,
    attack="linf-pgd",
    label_smoothing=0.1,
    use_cutmix=False,
    use_consistency=False,
    cons_lambda=0.0,
    cons_tem=0.0,
):
    """
    TRADES training (Zhang et al, 2019).
    """

    criterion_ce = SmoothCrossEntropyLoss(reduction="mean", smoothing=label_smoothing)
    criterion_kl = nn.KLDivLoss(reduction="batchmean")
    model.train()
    sur_model.train()
    track_bn_stats(sur_model, False)
    track_bn_stats(model, False)
    batch_size = len(x_natural)

    # 0.5 chance to select atk model from model/sur_model
    atk_model = random.choice([model, sur_model])
    # atk_model = sur_model

    x_adv = x_natural.detach() + torch.FloatTensor(x_natural.shape).uniform_(-epsilon, epsilon).cuda().detach()
    x_adv = torch.clamp(x_adv, 0.0, 1.0)

    if use_cutmix:  # CutMix
        p_natural = y
    else:
        # NOTE: Not sure if p_natural during attack should be original model or surrogate model
        p_natural = F.softmax(atk_model(x_natural), dim=1)
        p_natural = p_natural.detach()

    if attack == "linf-pgd":
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(atk_model(x_adv), dim=1), p_natural)
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

    elif attack == "mi-fgsm":
        momentum = torch.zeros_like(x_adv, device="cuda")
        decay = 1.0
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(atk_model(x_adv), dim=1), p_natural)
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            grad_norm = torch.norm(grad, p=1)
            grad /= grad_norm
            grad += momentum * decay
            momentum = grad
            x_adv = x_adv.detach() + step_size * grad.sign()
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

    elif attack == "l2-pgd":
        delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(F.log_softmax(atk_model(adv), dim=1), p_natural)
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        raise ValueError(f"Attack={attack} not supported for TRADES training!")
    model.train()
    track_bn_stats(model, True)
    sur_model.train()
    track_bn_stats(sur_model, True)

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)

    optimizer.zero_grad()
    # calculate robust loss
    logits_natural = model(x_natural)
    logits_adv = model(x_adv)

    sur_logits_natural = sur_model(x_natural)
    sur_logits_adv = sur_model(x_adv)

    if use_cutmix:  # CutMix
        loss_natural = criterion_kl(F.log_softmax(logits_natural, dim=1), y)
        sur_loss_natural = criterion_kl(F.log_softmax(sur_logits_natural, dim=1), y)
    else:
        loss_natural = criterion_ce(logits_natural, y)
        sur_loss_natural = criterion_ce(sur_logits_natural, y)

    loss_robust = criterion_kl(F.log_softmax(logits_adv, dim=1), F.softmax(logits_natural, dim=1))
    sur_loss_robust = criterion_kl(F.log_softmax(sur_logits_adv, dim=1), F.softmax(sur_logits_natural, dim=1))
    # loss = loss_natural + beta * loss_robust
    loss = loss_natural + beta * loss_robust + sur_loss_natural + beta * sur_loss_robust

    if use_consistency:
        logits_adv1, logits_adv2 = logits_adv.chunk(2)
        loss = loss + cons_lambda * _jensen_shannon_div(logits_adv1, logits_adv2, cons_tem)

        sur_logits_adv1, sur_logits_adv2 = sur_logits_adv.chunk(2)
        loss = loss + cons_lambda * _jensen_shannon_div(sur_logits_adv1, sur_logits_adv2, cons_tem)

    if use_cutmix:  # CutMix
        batch_metrics = {"loss": loss.item(), "clean_acc": -1, "adversarial_acc": -1}
    else:
        batch_metrics = {
            "loss": loss.item(),
            "clean_acc": accuracy(y, logits_natural.detach()),
            "adversarial_acc": accuracy(y, logits_adv.detach()),
            "sur_clean_acc": accuracy(y, sur_logits_natural.detach()),
            "sur_robus_acc": accuracy(y, sur_logits_adv.detach()),
        }

    return loss, batch_metrics


def trades_loss_rand(
    model,
    x_natural,
    y,
    optimizer,
    step_size=0.003,
    epsilon=0.031,
    perturb_steps=10,
    beta=1.0,
    attack="linf-pgd",
    label_smoothing=0.1,
    use_cutmix=False,
    use_consistency=False,
    cons_lambda=0.0,
    cons_tem=0.0,
):
    """
    TRADES training (Zhang et al, 2019).
    """

    criterion_ce = SmoothCrossEntropyLoss(reduction="mean", smoothing=label_smoothing)
    criterion_kl = nn.KLDivLoss(reduction="batchmean")
    model.train()
    track_bn_stats(model, False)
    batch_size = len(x_natural)

    x_adv = x_natural.detach() + torch.FloatTensor(x_natural.shape).uniform_(-epsilon, epsilon).cuda().detach()
    x_adv = torch.clamp(x_adv, 0.0, 1.0)

    if use_cutmix:  # CutMix
        p_natural = y
    else:
        p_natural = F.softmax(model(x_natural), dim=1)
        p_natural = p_natural.detach()

    attack = random.choice(["linf-pgd", "random"])

    if attack == "linf-pgd":
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1), p_natural)
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

    elif attack == "l2-pgd":
        delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1), p_natural)
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    elif attack == "random":
        pass
    else:
        raise ValueError(f"Attack={attack} not supported for TRADES training!")

    model.train()
    track_bn_stats(model, True)

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)

    optimizer.zero_grad()
    # calculate robust loss
    logits_natural = model(x_natural)
    logits_adv = model(x_adv)

    if use_cutmix:  # CutMix
        loss_natural = criterion_kl(F.log_softmax(logits_natural, dim=1), y)
    else:
        loss_natural = criterion_ce(logits_natural, y)

    loss_robust = criterion_kl(F.log_softmax(logits_adv, dim=1), F.softmax(logits_natural, dim=1))
    loss = loss_natural + beta * loss_robust

    if use_consistency:
        logits_adv1, logits_adv2 = logits_adv.chunk(2)
        loss = loss + cons_lambda * _jensen_shannon_div(logits_adv1, logits_adv2, cons_tem)

    if use_cutmix:  # CutMix
        batch_metrics = {"loss": loss.item(), "clean_acc": -1, "adversarial_acc": -1}
    else:
        batch_metrics = {
            "loss": loss.item(),
            "clean_acc": accuracy(y, logits_natural.detach()),
            "adversarial_acc": accuracy(y, logits_adv.detach()),
        }

    return loss, batch_metrics


def trades_loss_repa(
    model,
    x_natural,
    y,
    optimizer,
    feature_model,
    step_size=0.003,
    epsilon=0.031,
    perturb_steps=10,
    beta=1.0,
    attack="linf-pgd",
    label_smoothing=0.1,
    use_cutmix=False,
    use_consistency=False,
    cons_lambda=0.0,
    cons_tem=0.0,
    feat_reg_coeff=3.0,
    CFG=False,
    cond_align=False,
    repa_class=[],
):
    """
    TRADES training (Zhang et al, 2019).
    """
    assert not CFG, "CFG removed"
    assert not repa_class, "repa_class removed"

    criterion_ce = SmoothCrossEntropyLoss(reduction="mean", smoothing=label_smoothing)
    criterion_kl = nn.KLDivLoss(reduction="batchmean")
    model.train()
    track_bn_stats(model, False)
    batch_size = len(x_natural)

    x_adv = x_natural.detach() + torch.FloatTensor(x_natural.shape).uniform_(-epsilon, epsilon).cuda().detach()
    x_adv = torch.clamp(x_adv, 0.0, 1.0)

    if use_cutmix:  # CutMix
        p_natural = y
    else:
        p_natural = F.softmax(model(x_natural), dim=1)
        p_natural = p_natural.detach()

    if epsilon > 0.062 and attack != "l2-pgd":
        raise ValueError(f"Epsilon {epsilon} is too large for {attack} attack, please check if it is in the correct range")

    if attack == "linf-pgd":
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1), p_natural)
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

    elif attack == "l2-pgd":
        delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        with ctx_noparamgrad_and_eval(model):
            for _ in range(perturb_steps):
                adv = x_natural + delta

                # optimize
                optimizer_delta.zero_grad()
                with torch.enable_grad():
                    loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1), p_natural)
                loss.backward()
                # renorming gradient
                grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
                delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
                # avoid nan or inf if gradient is 0
                if (grad_norms == 0).any():
                    delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
                optimizer_delta.step()

                # projection
                delta.data.add_(x_natural)
                delta.data.clamp_(0, 1).sub_(x_natural)
                delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        raise ValueError(f"Attack={attack} not supported for TRADES training!")
    model.train()
    track_bn_stats(model, True)

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)

    # calculate robust loss
    logits_natural = model(x_natural)
    
    # Only compute REPA-related features if feat_reg_coeff is non-zero
    if feat_reg_coeff and feat_reg_coeff > 0:
        result = feature_model(x_natural, y, return_embed=cond_align)
        reg_feature = result["features"]
        cond_embed = result["emb"]
        # weight = result["weight"]
        if cond_align:
            logits_adv, repa_out = model(x_adv, repa=True, cond_embed=cond_embed)
        else:
            logits_adv, repa_out = model(x_adv, repa=True)
        assert isinstance(repa_out, List)
        assert isinstance(reg_feature, List)
        # NOTE: Currently only support one layer alignment
        assert len(repa_out) == len(reg_feature) == 1
        repa_out = repa_out[0]
        reg_feature = reg_feature[0]
        
        # Check if we should do patch-wise similarity (when num_token > 1)
        if repa_out.dim() == 3 and repa_out.size(1) > 1:
            # Patch-wise alignment
            repa_out = F.normalize(repa_out, dim=-1)
            
            # reg_feature: (bs*num_target, C, H, W) -> (bs*num_target, H*W, C)
            reg_feature = reg_feature.flatten(2).transpose(1, 2)
            # spatial normalization
            # NOTE: Not sure why the original paper uses 0.6, will check 1.0 later
            reg_feature = reg_feature - 0.6 * reg_feature.mean(dim=1, keepdim=True)
            reg_feature = reg_feature / (reg_feature.std(dim=1, keepdim=True) + 1e-6)
            # normalize for cosine similarity
            reg_feature = F.normalize(reg_feature, dim=-1)
            assert reg_feature.size(1) == repa_out.size(1), f"reg_feature.size(1) {reg_feature.size(1)} must be equal to repa_out.size(1) {repa_out.size(1)}"
            
            bs = repa_out.size(0)
            if reg_feature.size(0) % bs != 0:
                raise ValueError(f"reg_feature batch size {reg_feature.size(0)} must be divisible by repa_out batch size {bs}")
            
            num_target_per_image = reg_feature.size(0) // bs
            
            # repa_out: (bs, num_token, dim) -> (bs, 1, num_token, dim) -> (bs, num_target, num_token, dim)
            repa_out_expanded = repa_out.unsqueeze(1).expand(-1, num_target_per_image, -1, -1)
            
            # reg_feature: (bs*num_target, num_token, dim) -> (num_target, bs, num_token, dim) -> (bs, num_target, num_token, dim)
            reg_feature_reshaped = reg_feature.view(num_target_per_image, bs, reg_feature.size(1), -1).permute(1, 0, 2, 3)
            
            if repa_out_expanded.size(2) != reg_feature_reshaped.size(2):
                 raise ValueError(f"Token count mismatch: repa {repa_out_expanded.size(2)} vs reg {reg_feature_reshaped.size(2)}")
            
            # Cosine similarity per token then average: (bs, num_target)
            cos_sim = (repa_out_expanded * reg_feature_reshaped).sum(dim=-1).mean(dim=-1)
        else:
            repa_out = repa_out.view(-1, repa_out.size(-1))
            repa_out = F.normalize(repa_out, dim=-1)
            reg_feature = F.avg_pool2d(reg_feature, reg_feature.size(-1)).view(-1, repa_out.size(-1))
            reg_feature = F.normalize(reg_feature, dim=-1)
            # reg_feature may be a multiple of repa_out, which means there are multiple time step feature target for the same image
            # we need to average the cosine similarity over the time steps
            # assert reg_feature.size(0) % repa_out.size(0) == 0 and repa_out.size(-1) == reg_feature.size(-1)
            if reg_feature.size(0) % repa_out.size(0) != 0:
                raise ValueError(f"reg_feature.size(0) {reg_feature.size(0)} must be divisible by repa_out.size(0) {repa_out.size(0)}")
            if repa_out.size(-1) != reg_feature.size(-1):
                raise ValueError(f"repa_out.size(-1) {repa_out.size(-1)} must be equal to reg_feature.size(-1) {reg_feature.size(-1)}")
            num_target_per_image = reg_feature.size(0) // repa_out.size(0)
            bs = repa_out.size(0)
            
            # repa_out: (bs, dim) -> (bs, 1, dim) -> (bs, num_target, dim)
            repa_out_expanded = repa_out.unsqueeze(1).expand(-1, num_target_per_image, -1)
            
            # reg_feature: (bs*num_target, dim) -> (bs, num_target, dim)
            # reg_feature is [Batch_s1, Batch_s2], so we need to separate batches first
            reg_feature_reshaped = reg_feature.view(num_target_per_image, bs, -1).permute(1, 0, 2)
            
            # weight: (bs*num_target) -> (bs, num_target)
            # weight_reshaped = weight.view(bs, num_target_per_image)
            
            # Normalize weights per image
            # weight_probs = weight_reshaped / (weight_reshaped.sum(dim=1, keepdim=True) + 1e-10)
            
            # Cosine similarity: (bs, num_target)
            # Both features are already normalized
            cos_sim = (repa_out_expanded * reg_feature_reshaped).sum(dim=-1)
        
        # Weighted average of cosine similarity
        # maximize similarity <=> minimize (1 - similarity)
        # weighted_sim = (cos_sim * weight_probs).sum(dim=1)
        
        # cosine_loss = (1.0 - weighted_sim).mean()
        cosine_loss = 1.0 - cos_sim.mean()
    else:
        # When feat_reg_coeff is 0, use standard forward pass (like baseline)
        logits_adv = model(x_adv)
        cosine_loss = 0.0

    if use_cutmix:  # CutMix
        loss_natural = criterion_kl(F.log_softmax(logits_natural, dim=1), y)
    else:
        loss_natural = criterion_ce(logits_natural, y)

    loss_robust = criterion_kl(F.log_softmax(logits_adv, dim=1), F.softmax(logits_natural, dim=1))
    loss = loss_natural + beta * loss_robust + feat_reg_coeff * cosine_loss

    if use_consistency:
        logits_adv1, logits_adv2 = logits_adv.chunk(2)
        loss = loss + cons_lambda * _jensen_shannon_div(logits_adv1, logits_adv2, cons_tem)

    if use_cutmix:  # CutMix
        batch_metrics = {"loss": loss.item(), "clean_acc": -1, "adversarial_acc": -1}
    else:
        batch_metrics = {
            "loss": loss.item(),
            "clean_acc": accuracy(y, logits_natural.detach()),
            "adversarial_acc": accuracy(y, logits_adv.detach()),
        }
        # Only include cosine_loss and repa_samples metrics when REPA is active
        if feat_reg_coeff > 0:
            batch_metrics["cosine_loss"] = cosine_loss.item()
            # if len(repa_class) > 0:
            #     batch_metrics["num_repa_samples"] = num_repa_samples.item() if isinstance(num_repa_samples, torch.Tensor) else num_repa_samples

    return loss, batch_metrics

def repa_distillation_loss(
    model,
    x_natural,
    y,
    feature_model,
    step_size=0.003,
    epsilon=0.031,
    perturb_steps=10,
    attack="linf-pgd",
):
    """
    Repa distillation loss.
    """

    model.train()
    track_bn_stats(model, False)
    batch_size = len(x_natural)

    reg_feature = feature_model(x_natural, y)
    x_adv = x_natural.detach() + torch.FloatTensor(x_natural.shape).uniform_(-epsilon, epsilon).cuda().detach()
    x_adv = torch.clamp(x_adv, 0.0, 1.0)

    if attack == "linf-pgd":
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                # loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1), p_natural)
                _, repa_out_adv = model(x_adv, repa=True)
                loss_alignment = _alignment_loss(reg_feature, repa_out_adv)
            grad = torch.autograd.grad(loss_alignment, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

    elif attack == "l2-pgd":
        delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                # loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1), p_natural)
                _, repa_out_adv = model(adv, repa=True)
                loss_alignment = (-1) * _alignment_loss(reg_feature, repa_out_adv)
            loss_alignment.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        raise ValueError(f"Attack={attack} not supported for TRADES training!")
    model.train()
    track_bn_stats(model, True)

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)

    _, repa_out = model(x_adv, repa=True)
    assert isinstance(repa_out, List)
    assert isinstance(reg_feature, List)
    if len(repa_out) != len(reg_feature) + 1:
        raise ValueError(f"repa_out and reg_feature must have the same length, got {len(repa_out)} and {len(reg_feature)}")
    cosine_loss = _alignment_loss(reg_feature, repa_out)

    batch_metrics = {
        "cosine_loss": cosine_loss.item(),
    }

    return cosine_loss, batch_metrics

def _alignment_loss(reg_feature, repa_out):
    assert isinstance(repa_out, List)
    assert isinstance(reg_feature, List)
    if len(repa_out) != len(reg_feature):
        raise ValueError(f"repa_out and reg_feature must have the same length, got {len(repa_out)} and {len(reg_feature)}")
    cosine_loss = 0.0
    for i, repa_out_i in enumerate(repa_out):
        reg_feature_i = reg_feature[-i]
        reg_feature_i = F.avg_pool2d(reg_feature_i, reg_feature_i.size(-1)).view(repa_out_i.size(0), 1, repa_out_i.size(2))
        repa_out_i = F.normalize(repa_out_i, dim=2)
        reg_feature_i = F.normalize(reg_feature_i, dim=2)
        if repa_out_i.size() != reg_feature_i.size():
            print(f"repa_out_i.size() = {repa_out_i.size()}, reg_feature_i.size() = {reg_feature_i.size()}")
            raise ValueError("repa_out_i and reg_feature_i must have the same size")
        tmp_cos_sim = torch.sum(repa_out_i * reg_feature_i, dim=2)
        patch_mean_cos_sim = torch.mean(tmp_cos_sim, dim=1)
        batch_mean_cos_sim = torch.mean(patch_mean_cos_sim)
        cosine_loss += (1.0 - batch_mean_cos_sim)

    return cosine_loss


def standard_loss_repa(
    model,
    x_natural,
    y,
    feature_model,
    feat_reg_coeff=3.0,
    cond_align=False,
):
    """
    TRADES training (Zhang et al, 2019).
    """

    criterion_ce = SmoothCrossEntropyLoss(reduction="mean")
    model.train()
    track_bn_stats(model, True)
    if feat_reg_coeff:
        reg_feature, cond_embed = feature_model(x_natural, y, return_embed=True)
        if cond_align:
            logits_natural, repa_out = model(x_natural, repa=True, cond_embed=cond_embed)
        else:
            logits_natural, repa_out = model(x_natural, repa=True)
        assert isinstance(repa_out, List)
        assert isinstance(reg_feature, List)
        # assert len(repa_out) == len(reg_feature)
        if len(repa_out) != len(reg_feature):
            raise ValueError(f"repa_out and reg_feature must have the same length, got {len(repa_out)} and {len(reg_feature)}")
        # for repa_out_i, reg_feature_i in zip(repa_out, reg_feature):
        cosine_loss = _alignment_loss(reg_feature, repa_out)
        loss_natural = criterion_ce(logits_natural, y)

        loss = loss_natural + feat_reg_coeff * cosine_loss
        batch_metrics = {
            "loss": loss.item(),
            "clean_acc": accuracy(y, logits_natural.detach()),
            "cosine_loss": cosine_loss.item(),
        }
    else:
        logits_natural = model(x_natural)
        loss_natural = criterion_ce(logits_natural, y)
        loss = loss_natural
        batch_metrics = {
            "loss": loss.item(),
            "clean_acc": accuracy(y, logits_natural.detach()),
        }

    return loss, batch_metrics


def standard_loss_supervision(
    model,
    x_natural,
    y_binary,
    y_orig10,
    sup_weight=1.0,
):
    """
    Standard training with auxiliary supervision head.
    Main task: binary classification; Auxiliary: 10-class supervision on features.
    """

    criterion_ce_main = SmoothCrossEntropyLoss(reduction="mean")
    criterion_ce_sup = nn.CrossEntropyLoss(reduction="mean")

    model.train()
    track_bn_stats(model, True)

    logits_natural, sup_logits = model(x_natural, supervision=True)
    loss_main = criterion_ce_main(logits_natural, y_binary)
    loss_sup = criterion_ce_sup(sup_logits, y_orig10)

    loss = loss_main + sup_weight * loss_sup

    batch_metrics = {
        "loss": loss.item(),
        "clean_acc": accuracy(y_binary, logits_natural.detach()),
        "sup_acc": accuracy(y_orig10, sup_logits.detach()),
    }

    return loss, batch_metrics


def trades_supervision(
    model,
    x_natural,
    y_binary,
    y_orig10,
    optimizer,
    step_size=0.003,
    epsilon=0.031,
    perturb_steps=10,
    beta=1.0,
    attack="linf-pgd",
    label_smoothing=0.1,
    sup_weight=1.0,
):
    """
    TRADES adversarial training with auxiliary supervision head.
    Main task uses TRADES on binary logits; auxiliary supervision is standard CE on natural features.
    """

    criterion_ce_main = SmoothCrossEntropyLoss(reduction="mean", smoothing=label_smoothing)
    criterion_kl = nn.KLDivLoss(reduction="batchmean")
    # criterion_ce_sup = nn.CrossEntropyLoss(reduction="mean", smoothing=label_smoothing)
    # criterion_kl_sup = nn.KLDivLoss(reduction="batchmean")

    model.train()
    track_bn_stats(model, False)
    batch_size = len(x_natural)

    x_adv = x_natural.detach() + torch.FloatTensor(x_natural.shape).uniform_(-epsilon, epsilon).cuda().detach()
    x_adv = torch.clamp(x_adv, 0.0, 1.0)

    with torch.no_grad():
        p_natural = F.softmax(model(x_natural), dim=1)

    if attack == "linf-pgd":
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1), p_natural)
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif attack == "l2-pgd":
        delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=True)

        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1), p_natural)
            loss.backward()
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        raise ValueError(f"Attack={attack} not supported for TRADES training!")

    model.train()
    track_bn_stats(model, True)

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)

    optimizer.zero_grad()

    # Compute natural logits and supervision logits in one forward
    # logits_natural, sup_logits = model(x_natural, supervision=True)
    logits_natural, sup_logits_natural = model(x_natural, supervision=True)
    logits_adv, sup_logits_adv = model(x_adv, supervision=True)

    loss_natural = criterion_ce_main(logits_natural, y_binary)
    loss_robust = criterion_kl(F.log_softmax(logits_adv, dim=1), F.softmax(logits_natural, dim=1))

    sup_loss_natural = criterion_ce_main(sup_logits_natural, y_orig10)
    sup_loss_robust = criterion_kl(F.log_softmax(sup_logits_adv, dim=1), F.softmax(sup_logits_natural, dim=1))

    loss = loss_natural + beta * loss_robust + sup_weight * (sup_loss_natural + beta * sup_loss_robust)

    batch_metrics = {
        "loss": loss.item(),
        "clean_acc": accuracy(y_binary, logits_natural.detach()),
        "adversarial_acc": accuracy(y_binary, logits_adv.detach()),
        "sup_acc": accuracy(y_orig10, sup_logits_natural.detach()),
    }

    return loss, batch_metrics


def compute_alignment_loss_latent(repa_out, reg_feature, patch_wise=False):
    """
    Compute cosine similarity loss between classifier REPA output and diffusion features.
    
    Args:
        repa_out: (B, num_token, repa_dim) or (B, repa_dim) from classifier
        reg_feature: (B, num_patches, hidden_size) from diffusion (LightningDiT)
        patch_wise: Whether to use patch-wise alignment
    
    Returns:
        cosine_loss: scalar loss value (1 - mean cosine similarity)
    """
    # If using pooled features (single token)
    if not patch_wise or repa_out.dim() == 2 or (repa_out.dim() == 3 and repa_out.size(1) == 1):
        repa_out = repa_out.view(repa_out.size(0), -1)
        repa_out = F.normalize(repa_out, dim=-1)
        
        # Pool diffusion features: (B, num_patches, D) -> (B, D)
        reg_feature = reg_feature.mean(dim=1)
        reg_feature = F.normalize(reg_feature, dim=-1)
        
        cos_sim = (repa_out * reg_feature).sum(dim=-1)
        return 1.0 - cos_sim.mean()
    
    # Patch-wise alignment
    else:
        repa_out = F.normalize(repa_out, dim=-1)
        
        # Spatial normalization on diffusion features (following REPA paper)
        reg_feature = reg_feature - 0.6 * reg_feature.mean(dim=1, keepdim=True)
        reg_feature = reg_feature / (reg_feature.std(dim=1, keepdim=True) + 1e-6)
        reg_feature = F.normalize(reg_feature, dim=-1)
        
        # Check dimension compatibility
        if repa_out.size(1) != reg_feature.size(1):
            raise ValueError(
                f"Token count mismatch: repa {repa_out.size(1)} vs reg {reg_feature.size(1)}"
            )
        
        # Per-token cosine similarity, then average
        cos_sim = (repa_out * reg_feature).sum(dim=-1).mean(dim=-1)
        return 1.0 - cos_sim.mean()


def trades_loss_repa_teacher(
    model,
    x_natural,      # Raw images (B, 3, 224, 224) for classifier
    teacher_inputs, # Raw images or pre-computed latents for the frozen teacher
    y,
    optimizer,
    feature_model,   # Frozen teacher wrapper
    teacher_input="latent",
    step_size=0.003,
    epsilon=0.031,
    perturb_steps=10,
    beta=1.0,
    attack="linf-pgd",
    label_smoothing=0.1,
    feat_reg_coeff=3.0,
    num_timesteps=1,
    patch_wise=False,
    mix_fixed_sampled_timesteps=False,
    log_grad_cosine=False,
):
    """
    TRADES training with REPA using a frozen teacher.
    
    This is designed for ImageNet training with LightningDiT where:
    - Teacher features are extracted from either raw images or pre-computed latents
    - Classifier still uses raw images for adversarial training
    
    Args:
        model: Classifier model with REPA head
        x_natural: Raw images (B, 3, H, W) for classifier
        teacher_inputs: Teacher inputs, either raw images or pre-computed latents
        y: Labels (B,)
        optimizer: Optimizer (for interface compatibility, not used here)
        feature_model: Frozen teacher feature extractor
        teacher_input: Whether the teacher consumes `image` or `latent` inputs
        step_size: PGD step size
        epsilon: Perturbation budget
        perturb_steps: Number of PGD steps
        beta: TRADES beta (weight for robust loss)
        attack: Attack type ('linf-pgd' or 'l2-pgd')
        label_smoothing: Label smoothing factor
        feat_reg_coeff: REPA regularization coefficient
        num_timesteps: Number of timesteps to sample for averaging
        patch_wise: Whether to use patch-wise alignment
        mix_fixed_sampled_timesteps: When True and feature_model has a fixed timestep,
            use the fixed timestep for one batch and sample the rest (num_timesteps - 1).
    
    Returns:
        loss: Total loss
        batch_metrics: Dictionary of metrics
    """
    criterion_ce = SmoothCrossEntropyLoss(reduction="mean", smoothing=label_smoothing)
    criterion_kl = nn.KLDivLoss(reduction="batchmean")
    model.train()
    track_bn_stats(model, False)
    batch_size = len(x_natural)

    if teacher_input not in {"image", "latent"}:
        raise ValueError(f"teacher_input must be 'image' or 'latent', got {teacher_input}.")

    # Initialize adversarial examples
    x_adv = x_natural.detach() + torch.FloatTensor(x_natural.shape).uniform_(-epsilon, epsilon).cuda().detach()
    x_adv = torch.clamp(x_adv, 0.0, 1.0)

    # Get natural predictions for KL divergence target
    p_natural = F.softmax(model(x_natural), dim=1).detach()

    # Sanity check for epsilon
    if epsilon > 0.062 and attack != "l2-pgd":
        raise ValueError(
            f"Epsilon {epsilon} is too large for {attack} attack, "
            "please check if it is in the correct range (ImageNet typically uses 4/255 or 8/255)"
        )

    # Generate adversarial examples using PGD
    if attack == "linf-pgd":
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1), p_natural)
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

    elif attack == "l2-pgd":
        delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=True)

        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        with ctx_noparamgrad_and_eval(model):
            for _ in range(perturb_steps):
                adv = x_natural + delta

                optimizer_delta.zero_grad()
                with torch.enable_grad():
                    loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1), p_natural)
                loss.backward()
                
                grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
                delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
                if (grad_norms == 0).any():
                    delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
                optimizer_delta.step()

                delta.data.add_(x_natural)
                delta.data.clamp_(0, 1).sub_(x_natural)
                delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        raise ValueError(f"Attack={attack} not supported for TRADES training!")

    model.train()
    track_bn_stats(model, True)
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)

    # Calculate natural loss
    logits_natural = model(x_natural)

    # Compute REPA loss using latents
    if feat_reg_coeff and feat_reg_coeff > 0:
        # Get REPA output from classifier on adversarial image
        logits_adv, repa_out = model(x_adv, repa=True)
        
        # repa_out is a list, get first element
        assert isinstance(repa_out, List), f"Expected list, got {type(repa_out)}"
        repa_out = repa_out[0]  # (B, num_token, repa_dim)
        
        # Extract teacher features using images or latents.
        # Batched multi-timestep: expand batch, run single forward pass, compute loss per timestep
        if num_timesteps > 1:
            # Expand teacher inputs and labels for batched teacher inference.
            teacher_inputs_expanded = teacher_inputs.repeat(num_timesteps, 1, 1, 1)
            y_expanded = y.repeat(num_timesteps)
            
            # Handle mixed fixed + sampled timesteps
            if mix_fixed_sampled_timesteps and hasattr(feature_model, "timestep") and feature_model.timestep is not None:
                # Create mixed timestep tensor: [fixed, sampled, sampled, ...]
                device = teacher_inputs.device
                
                # First timestep batch uses the fixed value
                fixed_t = torch.full((batch_size,), feature_model.timestep, device=device)
                
                # Remaining timestep batches are sampled
                from core.models.lightning_dit.feature_extractor import sample_timestep_lognorm_like
                sampled_t = sample_timestep_lognorm_like(
                    batch_size * (num_timesteps - 1), 
                    device, 
                    feature_model.timestep_low, 
                    feature_model.timestep_high
                )
                
                # Concatenate: (B * num_timesteps,)
                mixed_timesteps = torch.cat([fixed_t, sampled_t], dim=0)
                
                # Single batched forward pass with explicit per-sample timesteps
                result = feature_model(
                    teacher_inputs_expanded,
                    y_expanded,
                    timestep_override=mixed_timesteps,
                    return_embed=False,
                )
            else:
                # Original behavior: feature_model handles timesteps internally
                result = feature_model(teacher_inputs_expanded, y_expanded, return_embed=False)
            
            # Extract features: (B * num_timesteps, N, D)
            if isinstance(result, dict):
                reg_feature = result['features'][0]
            elif isinstance(result, list):
                reg_feature = result[0]
            else:
                reg_feature = result
            
            # Expand repa_out to match: (B, N, D) -> (B * num_timesteps, N, D)
            repa_out = repa_out.repeat(num_timesteps, 1, 1)
            
            # Compute loss - averages over all B * num_timesteps samples
            # This is equivalent to: mean over timesteps of (mean over batch of per-sample loss)
            cosine_loss = compute_alignment_loss_latent(repa_out, reg_feature, patch_wise=patch_wise)
        else:
            # Single timestep - no expansion needed
            result = feature_model(teacher_inputs, y, return_embed=False)
            if isinstance(result, dict):
                reg_feature = result['features'][0]
            elif isinstance(result, list):
                reg_feature = result[0]
            else:
                reg_feature = result
            
            # Compute alignment loss
            cosine_loss = compute_alignment_loss_latent(repa_out, reg_feature, patch_wise=patch_wise)
    else:
        # Standard forward without REPA
        logits_adv = model(x_adv)
        cosine_loss = 0.0

    # Compute losses
    loss_natural = criterion_ce(logits_natural, y)
    loss_robust = criterion_kl(F.log_softmax(logits_adv, dim=1), F.softmax(logits_natural, dim=1))
    loss = loss_natural + beta * loss_robust + feat_reg_coeff * cosine_loss

    # Metrics
    batch_metrics = {
        "loss": loss.item(),
        "clean_acc": accuracy(y, logits_natural.detach()),
        "adversarial_acc": accuracy(y, logits_adv.detach()),
    }
    if feat_reg_coeff > 0:
        batch_metrics["cosine_loss"] = cosine_loss.item() if isinstance(cosine_loss, torch.Tensor) else cosine_loss

    # Diagnostic: cosine between the AT-loss gradient and the alignment-loss gradient.
    # Two extra backward passes (retain_graph keeps the graph alive for the caller's backward()).
    if log_grad_cosine and isinstance(cosine_loss, torch.Tensor) and cosine_loss.requires_grad:
        params_list = [p for p in model.parameters() if p.requires_grad]
        at_loss = loss_natural + beta * loss_robust
        grads_at = torch.autograd.grad(at_loss, params_list, retain_graph=True, allow_unused=True)
        grads_align = torch.autograd.grad(cosine_loss, params_list, retain_graph=True, allow_unused=True)
        dot = at_sq = align_sq = 0.0
        for g_at, g_al in zip(grads_at, grads_align):
            if g_at is not None:
                g_at = g_at.float()
                at_sq = at_sq + g_at.pow(2).sum()
            if g_al is not None:
                g_al = g_al.float()
                align_sq = align_sq + g_al.pow(2).sum()
            if g_at is not None and g_al is not None:
                dot = dot + (g_at * g_al).sum()
        if isinstance(at_sq, torch.Tensor) and isinstance(align_sq, torch.Tensor):
            denom = at_sq.sqrt() * align_sq.sqrt() + 1e-12
            batch_metrics["grad_cosine"] = (dot / denom).item()

    return loss, batch_metrics


def trades_loss_repa_latent(
    model,
    x_natural,
    latents,
    y,
    optimizer,
    feature_model,
    step_size=0.003,
    epsilon=0.031,
    perturb_steps=10,
    beta=1.0,
    attack="linf-pgd",
    label_smoothing=0.1,
    feat_reg_coeff=3.0,
    num_timesteps=1,
    patch_wise=False,
    mix_fixed_sampled_timesteps=False,
):
    """
    Backward-compatible wrapper for the old latent-only ImageNet REPA path.
    """
    return trades_loss_repa_teacher(
        model,
        x_natural,
        latents,
        y,
        optimizer,
        feature_model,
        teacher_input="latent",
        step_size=step_size,
        epsilon=epsilon,
        perturb_steps=perturb_steps,
        beta=beta,
        attack=attack,
        label_smoothing=label_smoothing,
        feat_reg_coeff=feat_reg_coeff,
        num_timesteps=num_timesteps,
        patch_wise=patch_wise,
        mix_fixed_sampled_timesteps=mix_fixed_sampled_timesteps,
    )