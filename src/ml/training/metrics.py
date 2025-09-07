"""
Evaluation metrics for flood prediction model.
Implementation of metrics specified in APPROACH.md: IoU, F1, AUCPR, Brier score.
"""

import torch
import numpy as np
from sklearn.metrics import (
    precision_recall_curve, 
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
    calibration_curve
)
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class FloodMetrics:
    """Compute comprehensive metrics for flood prediction evaluation."""
    
    def __init__(self, threshold: float = 0.5, device: str = 'cpu'):
        """
        Initialize metrics calculator.
        
        Args:
            threshold: Default threshold for binary predictions
            device: Device for tensor operations
        """
        self.threshold = threshold
        self.device = device
        
    def iou_score(self, predictions: torch.Tensor, targets: torch.Tensor, threshold: Optional[float] = None) -> float:
        """
        Compute Intersection over Union (Jaccard index).
        Primary metric mentioned in APPROACH.md.
        
        Args:
            predictions: Model predictions [B, 1, H, W] or probabilities
            targets: Ground truth labels [B, 1, H, W]  
            threshold: Probability threshold (uses self.threshold if None)
            
        Returns:
            IoU score
        """
        if threshold is None:
            threshold = self.threshold
            
        # Convert to binary predictions if probabilities
        if predictions.max() <= 1.0 and predictions.min() >= 0.0:
            pred_binary = (predictions > threshold).float()
        else:
            pred_binary = (torch.sigmoid(predictions) > threshold).float()
            
        pred_binary = pred_binary.view(-1)
        targets = targets.view(-1)
        
        intersection = (pred_binary * targets).sum()
        union = pred_binary.sum() + targets.sum() - intersection
        
        # Handle edge case where both are empty
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
            
        iou = intersection / (union + 1e-8)
        return iou.item()
    
    def f1_score(self, predictions: torch.Tensor, targets: torch.Tensor, threshold: Optional[float] = None) -> float:
        """
        Compute F1 score.
        Mentioned in APPROACH.md alongside IoU.
        
        Args:
            predictions: Model predictions
            targets: Ground truth labels
            threshold: Probability threshold
            
        Returns:
            F1 score
        """
        if threshold is None:
            threshold = self.threshold
            
        if predictions.max() <= 1.0 and predictions.min() >= 0.0:
            pred_binary = (predictions > threshold).float()
        else:
            pred_binary = (torch.sigmoid(predictions) > threshold).float()
            
        pred_binary = pred_binary.view(-1)
        targets = targets.view(-1)
        
        tp = (pred_binary * targets).sum()
        fp = (pred_binary * (1 - targets)).sum()
        fn = ((1 - pred_binary) * targets).sum()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        return f1.item()
    
    def precision_recall_f1(self, predictions: torch.Tensor, targets: torch.Tensor, threshold: Optional[float] = None) -> Tuple[float, float, float]:
        """
        Compute precision, recall, and F1 score.
        
        Args:
            predictions: Model predictions
            targets: Ground truth labels  
            threshold: Probability threshold
            
        Returns:
            Tuple of (precision, recall, f1)
        """
        if threshold is None:
            threshold = self.threshold
            
        if predictions.max() <= 1.0 and predictions.min() >= 0.0:
            pred_binary = (predictions > threshold).float()
        else:
            pred_binary = (torch.sigmoid(predictions) > threshold).float()
            
        pred_binary = pred_binary.view(-1)
        targets = targets.view(-1)
        
        tp = (pred_binary * targets).sum().item()
        fp = (pred_binary * (1 - targets)).sum().item()
        fn = ((1 - pred_binary) * targets).sum().item()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8) 
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        return precision, recall, f1
    
    def compute_aucpr(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Compute Area Under Precision-Recall Curve.
        Mentioned in APPROACH.md as robust with imbalance.
        
        Args:
            predictions: Model predictions (logits or probabilities)
            targets: Ground truth labels
            
        Returns:
            AUCPR score
        """
        # Convert to probabilities if logits
        if predictions.max() > 1.0 or predictions.min() < 0.0:
            predictions = torch.sigmoid(predictions)
            
        pred_np = predictions.detach().cpu().numpy().flatten()
        targets_np = targets.detach().cpu().numpy().flatten()
        
        try:
            aucpr = average_precision_score(targets_np, pred_np)
        except ValueError:
            # Handle case where all targets are same class
            logger.warning("Could not compute AUCPR - all targets same class")
            aucpr = 0.0
            
        return aucpr
    
    def compute_auroc(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Compute Area Under ROC Curve.
        
        Args:
            predictions: Model predictions
            targets: Ground truth labels
            
        Returns:
            AUROC score  
        """
        if predictions.max() > 1.0 or predictions.min() < 0.0:
            predictions = torch.sigmoid(predictions)
            
        pred_np = predictions.detach().cpu().numpy().flatten()
        targets_np = targets.detach().cpu().numpy().flatten()
        
        try:
            auroc = roc_auc_score(targets_np, pred_np)
        except ValueError:
            logger.warning("Could not compute AUROC - all targets same class")
            auroc = 0.5
            
        return auroc
    
    def brier_score(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Compute Brier score for probability calibration.
        Mentioned in APPROACH.md for calibration assessment.
        
        Args:
            predictions: Model predictions
            targets: Ground truth labels
            
        Returns:
            Brier score (lower is better)
        """
        if predictions.max() > 1.0 or predictions.min() < 0.0:
            predictions = torch.sigmoid(predictions)
            
        pred_np = predictions.detach().cpu().numpy().flatten()
        targets_np = targets.detach().cpu().numpy().flatten()
        
        brier = brier_score_loss(targets_np, pred_np)
        return brier
    
    def reliability_curve(self, predictions: torch.Tensor, targets: torch.Tensor, n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute reliability curve for calibration assessment.
        Mentioned in APPROACH.md.
        
        Args:
            predictions: Model predictions
            targets: Ground truth labels
            n_bins: Number of bins for reliability curve
            
        Returns:
            Tuple of (fraction_of_positives, mean_predicted_value)
        """
        if predictions.max() > 1.0 or predictions.min() < 0.0:
            predictions = torch.sigmoid(predictions)
            
        pred_np = predictions.detach().cpu().numpy().flatten()
        targets_np = targets.detach().cpu().numpy().flatten()
        
        fraction_pos, mean_pred = calibration_curve(targets_np, pred_np, n_bins=n_bins)
        return fraction_pos, mean_pred
    
    def precision_recall_curve_data(self, predictions: torch.Tensor, targets: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute precision-recall curve for threshold selection.
        Mentioned in APPROACH.md for operating point selection.
        
        Args:
            predictions: Model predictions
            targets: Ground truth labels
            
        Returns:
            Tuple of (precision, recall, thresholds)
        """
        if predictions.max() > 1.0 or predictions.min() < 0.0:
            predictions = torch.sigmoid(predictions)
            
        pred_np = predictions.detach().cpu().numpy().flatten()
        targets_np = targets.detach().cpu().numpy().flatten()
        
        precision, recall, thresholds = precision_recall_curve(targets_np, pred_np)
        return precision, recall, thresholds
    
    def youden_threshold(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Find optimal threshold using Youden's J statistic.
        Mentioned in APPROACH.md for threshold selection.
        
        Args:
            predictions: Model predictions
            targets: Ground truth labels
            
        Returns:
            Optimal threshold
        """
        precision, recall, thresholds = self.precision_recall_curve_data(predictions, targets)
        
        # Compute J statistic for each threshold
        j_scores = precision + recall - 1
        
        # Find threshold with maximum J score
        best_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        
        return optimal_threshold
    
    def compute_all_metrics(self, predictions: torch.Tensor, targets: torch.Tensor, threshold: Optional[float] = None) -> Dict[str, float]:
        """
        Compute all metrics for comprehensive evaluation.
        
        Args:
            predictions: Model predictions
            targets: Ground truth labels
            threshold: Probability threshold
            
        Returns:
            Dictionary of all computed metrics
        """
        if threshold is None:
            threshold = self.threshold
            
        metrics = {}
        
        # Binary classification metrics
        metrics['iou'] = self.iou_score(predictions, targets, threshold)
        metrics['f1'] = self.f1_score(predictions, targets, threshold)
        
        precision, recall, f1 = self.precision_recall_f1(predictions, targets, threshold)
        metrics['precision'] = precision
        metrics['recall'] = recall
        
        # Area under curve metrics
        metrics['aucpr'] = self.compute_aucpr(predictions, targets)
        metrics['auroc'] = self.compute_auroc(predictions, targets)
        
        # Calibration metrics
        metrics['brier_score'] = self.brier_score(predictions, targets)
        
        # Optimal threshold
        metrics['youden_threshold'] = self.youden_threshold(predictions, targets)
        
        return metrics
    
    def print_metrics_summary(self, metrics: Dict[str, float]) -> None:
        """Print formatted metrics summary."""
        logger.info("=" * 50)
        logger.info("FLOOD PREDICTION METRICS SUMMARY")
        logger.info("=" * 50)
        logger.info(f"IoU (Jaccard):        {metrics['iou']:.4f}")
        logger.info(f"F1 Score:            {metrics['f1']:.4f}")
        logger.info(f"Precision:           {metrics['precision']:.4f}")
        logger.info(f"Recall:              {metrics['recall']:.4f}")
        logger.info(f"AUCPR:               {metrics['aucpr']:.4f}")
        logger.info(f"AUROC:               {metrics['auroc']:.4f}")  
        logger.info(f"Brier Score:         {metrics['brier_score']:.4f}")
        logger.info(f"Optimal Threshold:   {metrics['youden_threshold']:.4f}")
        logger.info("=" * 50)