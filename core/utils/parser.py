import argparse

from core.attacks import ATTACKS
from core.data import DATASETS
from core.models import MODELS
from .train import SCHEDULERS

from .utils import str2bool, str2float


def parser_train():
    """
    Parse input arguments (train.py).
    """
    parser = argparse.ArgumentParser(description="Standard + Adversarial Training.")

    # parser.add_argument('--augment', type=str2bool, default=True, help='Augment training set.')
    parser.add_argument(
        "--augment",
        type=str,
        default="base",
        choices=["none", "base", "cutout", "autoaugment", "randaugment", "idbh", "3-augment", "base_randaugment", "base-256", "rrc"],
        help="Augment training set.",
    )

    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size for training.")
    parser.add_argument("--batch-size-validation", type=int, default=512, help="Batch size for testing.")

    parser.add_argument("--data-dir", type=str, default="./data/")
    parser.add_argument("--log-dir", type=str, default="./logs/")

    parser.add_argument("-d", "--data", type=str, default="cifar10s", choices=DATASETS, help="Data to use.")
    parser.add_argument(
        "--desc", type=str, required=True, help="Description of experiment. It will be used to name directories."
    )

    parser.add_argument(
        "-m", "--model", choices=MODELS, default="wrn-28-10-swish", help="Model architecture to be used."
    )
    parser.add_argument("--normalize", type=str2bool, default=False, help="Normalize input.")
    parser.add_argument("--pretrained-file", type=str, default=None, help="Pretrained weights file name.")

    parser.add_argument("-na", "--num-adv-epochs", type=int, default=400, help="Number of adversarial training epochs.")
    parser.add_argument(
        "--stop-epoch",
        type=int,
        default=None,
        help="Optional last epoch to run before exiting early. Useful when you want to stop and resume later.",
    )
    parser.add_argument("--adv-eval-freq", type=int, default=25, help="Adversarial evaluation frequency (in epochs).")

    parser.add_argument("--beta", default=None, type=float, help="Stability regularization, i.e., 1/lambda in TRADES.")

    parser.add_argument("--lr", type=float, default=0.4, help="Learning rate for optimizer (SGD).")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="Optimizer (SGD) weight decay.")
    parser.add_argument("--scheduler", choices=SCHEDULERS, default="cosinew", help="Type of scheduler.")
    parser.add_argument("--nesterov", type=str2bool, default=True, help="Use Nesterov momentum.")
    parser.add_argument("--clip-grad", type=float, default=None, help="Gradient norm clipping.")

    parser.add_argument(
        "-a", "--attack", type=str, choices=ATTACKS + ["mi-fgsm"], default="linf-pgd", help="Type of attack."
    )
    parser.add_argument("--attack-eps", type=str2float, default=8 / 255, help="Epsilon for the attack.")
    parser.add_argument("--attack-step", type=str2float, default=2 / 255, help="Step size for PGD attack.")
    parser.add_argument(
        "--attack-iter", type=int, default=10, help="Max. number of iterations (if any) for the attack."
    )
    parser.add_argument(
        "--keep-clean", type=str2bool, default=False, help="Use clean samples during adversarial training."
    )

    parser.add_argument(
        "--debug", action="store_true", default=False, help="Debug code. Run 1 epoch of training and evaluation."
    )
    parser.add_argument("--mart", action="store_true", default=False, help="MART training.")

    parser.add_argument("--unsup-fraction", type=float, default=0.7, help="Ratio of unlabelled data to labelled data.")
    parser.add_argument(
        "--aux-data-filename",
        type=str,
        help="Path to additional Tiny Images data.",
        default="/cluster/scratch/rarade/cifar10s/ti_500K_pseudo_labeled.pickle",
    )

    parser.add_argument("--seed", type=int, default=1, help="Random seed.")

    ### Consistency
    parser.add_argument("--consistency", action="store_true", default=False, help="use Consistency.")
    parser.add_argument("--cons_lambda", type=float, default=1.0, help="lambda for Consistency.")
    parser.add_argument("--cons_tem", type=float, default=0.5, help="temperature for Consistency.")

    ### Resume
    parser.add_argument("--resume_path", default="", type=str)

    ### Our methods
    parser.add_argument("--LSE", action="store_true", default=False, help="LSE training.")
    parser.add_argument("--ls", type=float, default=0.1, help="label smoothing.")
    parser.add_argument("--clip_value", default=0, type=float)
    parser.add_argument("--CutMix", action="store_true", default=False, help="use CutMix.")

    # parser.add_argument("--adam", action="store_true", default=False, help="Use Adam optimizer instead of SGD.")
    parser.add_argument("--opt-type", type=str, default="sgd", choices=["sgd", "adam", "lion", "muon"], help="Type of optimizer.")

    parser.add_argument("--tau", type=float, default=0.995, help="Weight averaging decay.")
    parser.add_argument("--feat-reg-coeff", type=float, default=3.0, help="Feature regularization coefficient.")
    parser.add_argument(
        "--feat-reg-adv",
        action="store_true",
        help="Input for diffusion uses adv examples instead of natural examples.",
    )
    parser.add_argument(
        "--alignment_v3",
        action="store_true",
        help="Also attack representation alignment for adversarial example augmentation.",
    )
    parser.add_argument(
        "--feat-reg-warmup-epochs",
        type=int,
        default=0,
        help="Linearly warm up feat-reg-coeff from 0 over this many epochs (0 = no warmup).",
    )
    parser.add_argument(
        "--log-grad-cosine",
        action="store_true",
        help="Log cosine similarity between the AT-loss gradient and the alignment-loss gradient (teacher REPA path only).",
    )
    parser.add_argument(
        "--grad-cosine-interval",
        type=int,
        default=20,
        help="Compute the grad-cosine diagnostic every N batches (costs two extra backward passes).",
    )
    parser.add_argument(
        "--teacher-feat-standardize",
        action="store_true",
        help="Per-channel standardization of SSL teacher target features (batch statistics).",
    )
    parser.add_argument("--unconditional", action="store_true", help="Use unconditional feature extractor.")
    parser.add_argument("--sigma", type=float, default=None, help="Diffusion noise scale. pass None to use sampled sigma.")
    parser.add_argument("--dual-time", action="store_true", help="Align with two time step (one fixed, one sampled) for diffusion alignment.")
    parser.add_argument("--duplicate-time", action="store_true", help="Align with two time step (both fixed or sampled) for diffusion alignment.")
    parser.add_argument("--perturb", action="store_true", help="Use perturbed image as the input of the feature extractor.")
    parser.add_argument(
        "--feature_layer_name",
        nargs='+',
        type=str,
        default=["16x16_block0"],
        help="Feature layer name.",
        # choices=["16x16_block0", "8x8_block4"],
    )
    parser.add_argument(
        "--accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps.",
    )
    parser.add_argument(
        "--CFG",
        action="store_true",
        help="Uses CFG for representation alignment.",
    )
    parser.add_argument(
        "--repa_dim",
        type=int,
        default=256,
        help="Dimensionality of the repa head output.",
    )
    parser.add_argument(
        "--cond_align",
        action="store_true",
        help="Use conditional alignment head for representation alignment.",
    )
    parser.add_argument(
        "--dummy-orth-feat",
        action="store_true",
        help="Use dummy orthogonal feature extractor.",
    )
    parser.add_argument(
        "--use-disc-feat",
        action="store_true",
        help="Use a copy of discriminative feature extractor for representation alignment.",
    )
    parser.add_argument(
        "--disc-feat-path",
        type=str,
        default=None,
        help="Path to the discriminative feature extractor.",
    )
    parser.add_argument(
        "--disc-feat-model",
        type=str,
        default=None,
        help="Model name of the discriminative feature extractor.",
        choices=["wrn-28-10-swish-repa-pool-8x8-pool", "wrn-28-10-swish-noisy", "diffusion-backbone-8x8_block4-uncond-sigma0.1-drop0.0-ft-scratch-perturb"],
    )
    parser.add_argument(
        "--repa-class",
        type=str,
        default=None,
        help="Comma-separated list of class indices to apply REPA loss to (e.g., '0,1,2,4'). If None, apply to all classes.",
    )

    parser.add_argument(
        "--teacher-type",
        type=str,
        default="lightning-dit",
        choices=["lightning-dit", "dinov3", "mae", "clip"],
        help="Teacher used for ImageNet REPA alignment.",
    )
    parser.add_argument(
        "--teacher-model-name",
        type=str,
        default=None,
        help="Optional pretrained teacher model name. If omitted, a default is chosen for the teacher type.",
    )
    parser.add_argument(
        "--teacher-pool",
        type=str,
        default="avg",
        choices=["avg", "cls", "patch", "pre-logits"],
        help="How to pool teacher features before alignment.",
    )
    
    parser.add_argument(
        "--cached-feature-teacher",
        action="store_true",
        help="Alignment targets are pre-computed pooled feature vectors read from --latent-dir "
             "(e.g. SD-1.5 mid-block features); no teacher forward pass at train time.",
    )
    # LightningDiT feature extractor arguments for ImageNet REPA
    parser.add_argument(
        "--use-latent-repa",
        action="store_true",
        help="Use latent-based REPA with LightningDiT (for ImageNet with pre-computed VAE latents).",
    )
    parser.add_argument(
        "--lightning-dit-config",
        type=str,
        default=None,
        help="Path to LightningDiT config YAML file.",
    )
    parser.add_argument(
        "--lightning-dit-ckpt",
        type=str,
        default=None,
        help="Path to LightningDiT checkpoint file.",
    )
    parser.add_argument(
        "--latent-dir",
        type=str,
        default=None,
        help="Path to directory containing pre-computed VAE latents for ImageNet (safetensors).",
    )
    parser.add_argument(
        "--aux-latent-dir",
        type=str,
        default=None,
        help="Path to directory containing pre-computed VAE latents for aux/synthetic data (safetensors).",
    )
    parser.add_argument(
        "--dit-feature-layers",
        type=str,
        default="14",
        help="Comma-separated list of LightningDiT layer indices to extract features from (e.g., '14' or '7,14,21').",
    )
    parser.add_argument(
        "--dit-timestep",
        type=float,
        default=-1.0,
        help="Fixed timestep for LightningDiT feature extraction (0-1). Use -1 for sampling.",
    )
    parser.add_argument(
        "--dit-timestep-low",
        type=float,
        default=0.7,
        help="Lowest timestep for LightningDiT feature extraction (0-1).",
    )
    parser.add_argument(
        "--dit-timestep-high",
        type=float,
        default=0.9,
        help="Highest timestep for LightningDiT feature extraction (0-1).",
    )
    parser.add_argument(
        "--num-timesteps",
        type=int,
        default=1,
        help="Number of timesteps to sample and average for REPA loss.",
    )
    parser.add_argument(
        "--mix-fixed-sampled-timesteps",
        action="store_true",
        help="When num-timesteps > 1 and dit-timestep is set, use the fixed timestep for one sample and sample the rest.",
    )
    parser.add_argument(
        "--patch-wise-alignment",
        action="store_true",
        help="Use patch-wise alignment instead of global pooling for REPA.",
    )
    
    return parser


def parser_eval():
    """
    Parse input arguments (eval-adv.py, eval-corr.py, eval-aa.py).
    """
    parser = argparse.ArgumentParser(description="Robustness evaluation.")

    parser.add_argument("--data", type=str, default="cifar10", choices=DATASETS, help="Data to use.")
    parser.add_argument("--data-dir", type=str, default="/tmp2/dataset/")
    parser.add_argument("--log-dir", type=str, default="./checkpoint/eval/")

    parser.add_argument("--desc", type=str, required=True, help="Description of model to be evaluated.")
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of test samples.")

    # eval-aa.py
    parser.add_argument("--train", action="store_true", default=False, help="Evaluate on training set.")
    parser.add_argument(
        "-v", "--version", type=str, default="standard", choices=["custom", "plus", "standard", "rand"], help="Version of AA."
    )

    # eval-adv.py
    parser.add_argument("--source", type=str, default=None, help="Path to source model for black-box evaluation.")
    parser.add_argument("--wb", action="store_true", default=False, help="Perform white-box PGD evaluation.")

    # eval-rb.py
    parser.add_argument(
        "--threat",
        type=str,
        default="corruptions",
        choices=["corruptions", "Linf", "L2"],
        help="Threat model for RobustBench evaluation.",
    )

    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    return parser
