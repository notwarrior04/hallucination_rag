import logging
import numpy as np
import scipy.optimize as opt

logger = logging.getLogger(__name__)

class TemperatureScaler:
    """
    Fits and applies temperature scaling to calibrate confidence scores.
    Uses validation set logits and binary labels.
    """
    def __init__(self):
        self.temperature = 1.0

    def fit(self, logits: np.ndarray, labels: np.ndarray):
        """
        Optimizes temperature T using validation set logits and labels.
        Labels should be 0.0 (correct) or 1.0 (hallucinated).
        """
        # Define the negative log-likelihood (binary cross-entropy loss) function
        def loss_fn(t):
            t = t[0]
            scaled_logits = logits / t
            # Sigmoid activation to get probabilities
            probs = 1.0 / (1.0 + np.exp(-scaled_logits))
            # Clip probabilities to avoid log(0)
            probs = np.clip(probs, 1e-15, 1.0 - 1e-15)
            # BCE loss
            bce = -np.mean(labels * np.log(probs) + (1.0 - labels) * np.log(1.0 - probs))
            return bce

        # Minimize the loss with respect to T (constrained to be positive)
        res = opt.minimize(loss_fn, x0=[1.0], bounds=[(0.01, 10.0)], method="L-BFGS-B")
        if not res.success:
            logger.warning(f"Temperature scaling optimization failed: {res.message}. Falling back to x0.")
        self.temperature = float(res.x[0])
        logger.info(f"Optimal temperature found: {self.temperature:.4f}")

    def scale(self, logits: np.ndarray) -> np.ndarray:
        """Scales logits by the learned temperature."""
        return logits / self.temperature


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error (ECE)."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for i in range(n_bins):
        bin_lower = bins[i]
        bin_upper = bins[i + 1]
        
        # Find indices of samples falling into the bin
        in_bin = (probs >= bin_lower) & (probs < bin_upper)
        if i == n_bins - 1:  # Include upper boundary in the last bin
            in_bin = in_bin | (probs == bin_upper)
            
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(labels[in_bin])
            avg_confidence_in_bin = np.mean(probs[in_bin])
            ece += prop_in_bin * np.abs(avg_confidence_in_bin - accuracy_in_bin)
            
    return float(ece)


def compute_brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Computes Brier Score."""
    return float(np.mean((probs - labels) ** 2))
