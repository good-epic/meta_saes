"""
Training Logger for Meta-SAE experiments.

Provides structured logging of all training metrics to JSON files,
with periodic saves and comprehensive metric tracking.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
import torch


@dataclass
class TrainingMetrics:
    """Container for a single training step's metrics."""
    step: int
    phase: str  # 'joint_primary', 'joint_meta', 'solo_primary', 'sequential_meta'
    timestamp: float

    # Core losses
    loss: float = 0.0
    l2_loss: float = 0.0
    l0_norm: float = 0.0

    # Sparsity metrics
    l1_loss: float = 0.0
    l1_norm: float = 0.0
    l0_loss: Optional[float] = None
    l0_coeff: Optional[float] = None
    l0_ema: Optional[float] = None
    target_l0: Optional[float] = None

    # Auxiliary losses
    aux_loss: Optional[float] = None
    decomp_penalty: Optional[float] = None

    # Feature health
    num_dead_features: int = 0

    # Memory (optional)
    gpu_memory_allocated_gb: Optional[float] = None
    gpu_memory_reserved_gb: Optional[float] = None


class TrainingLogger:
    """
    Logs training metrics to structured JSON files.

    Usage:
        logger = TrainingLogger(output_dir="outputs/run_001", save_freq=100)

        for step in range(num_steps):
            output = sae(batch)
            logger.log_step(step, "solo_primary", output)

        logger.save()  # Final save
    """

    def __init__(
        self,
        output_dir: str | Path,
        run_name: str = "training",
        save_freq: int = 100,
        log_memory: bool = True,
    ):
        """
        Args:
            output_dir: Directory to save log files
            run_name: Name prefix for log files
            save_freq: Save logs every N steps
            log_memory: Whether to log GPU memory usage
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self.save_freq = save_freq
        self.log_memory = log_memory

        # Metrics storage by phase
        self.metrics: Dict[str, List[Dict[str, Any]]] = {
            "joint_primary": [],
            "joint_meta": [],
            "solo_primary": [],
            "sequential_meta": [],
        }

        # Training metadata
        self.start_time = time.time()
        self.config: Dict[str, Any] = {}

        # Step counters per phase
        self.step_counts: Dict[str, int] = {
            "joint_primary": 0,
            "joint_meta": 0,
            "solo_primary": 0,
            "sequential_meta": 0,
        }

    def set_config(self, cfg: Dict[str, Any], meta_cfg: Dict[str, Any] = None, penalty_cfg: Dict[str, Any] = None):
        """Store configuration for the run."""
        self.config = {
            "primary": self._serialize_config(cfg),
            "meta": self._serialize_config(meta_cfg) if meta_cfg else None,
            "penalty": penalty_cfg,
            "start_time": datetime.now().isoformat(),
        }

    def _serialize_config(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Convert config to JSON-serializable format."""
        result = {}
        for k, v in cfg.items():
            if isinstance(v, torch.dtype):
                result[k] = str(v)
            elif isinstance(v, torch.device):
                result[k] = str(v)
            elif hasattr(v, 'item'):  # tensor scalar
                result[k] = v.item()
            elif isinstance(v, (int, float, str, bool, list, dict, type(None))):
                result[k] = v
            else:
                result[k] = str(v)
        return result

    def log_step(
        self,
        step: int,
        phase: str,
        output: Dict[str, Any],
        extra: Dict[str, Any] = None,
    ):
        """
        Log metrics from a single training step.

        Args:
            step: Global step number
            phase: Training phase ('joint_primary', 'joint_meta', 'solo_primary', 'sequential_meta')
            output: Output dictionary from SAE forward pass
            extra: Additional metrics to log
        """
        metrics = TrainingMetrics(
            step=step,
            phase=phase,
            timestamp=time.time() - self.start_time,
        )

        # Extract core metrics from output
        metrics.loss = self._to_float(output.get("loss", 0.0))
        metrics.l2_loss = self._to_float(output.get("l2_loss", 0.0))
        metrics.l0_norm = self._to_float(output.get("l0_norm", 0.0))
        metrics.l1_loss = self._to_float(output.get("l1_loss", 0.0))
        metrics.l1_norm = self._to_float(output.get("l1_norm", 0.0))

        # JumpReLU-specific metrics
        if "l0_loss" in output:
            metrics.l0_loss = self._to_float(output["l0_loss"])
        if "l0_coeff" in output:
            metrics.l0_coeff = self._to_float(output["l0_coeff"])
        if "l0_ema" in output:
            metrics.l0_ema = self._to_float(output["l0_ema"])
        if "target_l0" in output:
            metrics.target_l0 = self._to_float(output["target_l0"])

        # Auxiliary losses
        if "aux_loss" in output:
            metrics.aux_loss = self._to_float(output["aux_loss"])
        if "decomp_penalty" in output:
            metrics.decomp_penalty = self._to_float(output["decomp_penalty"])

        # Feature health
        if "num_dead_features" in output:
            metrics.num_dead_features = self._to_int(output["num_dead_features"])

        # GPU memory
        if self.log_memory and torch.cuda.is_available():
            metrics.gpu_memory_allocated_gb = torch.cuda.memory_allocated() / 1e9
            metrics.gpu_memory_reserved_gb = torch.cuda.memory_reserved() / 1e9

        # Add extra metrics
        metrics_dict = asdict(metrics)
        if extra:
            for k, v in extra.items():
                metrics_dict[k] = self._to_float(v) if isinstance(v, (torch.Tensor, float, int)) else v

        # Store
        self.metrics[phase].append(metrics_dict)
        self.step_counts[phase] += 1

        # Periodic save
        total_steps = sum(self.step_counts.values())
        if total_steps % self.save_freq == 0:
            self.save()

    def _to_float(self, v) -> float:
        """Convert value to float, handling tensors."""
        if isinstance(v, torch.Tensor):
            return v.detach().cpu().item()
        return float(v) if v is not None else 0.0

    def _to_int(self, v) -> int:
        """Convert value to int, handling tensors."""
        if isinstance(v, torch.Tensor):
            return int(v.detach().cpu().item())
        return int(v) if v is not None else 0

    def get_latest(self, phase: str) -> Optional[Dict[str, Any]]:
        """Get the most recent metrics for a phase."""
        if self.metrics[phase]:
            return self.metrics[phase][-1]
        return None

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of training metrics."""
        summary = {
            "total_time_seconds": time.time() - self.start_time,
            "step_counts": self.step_counts,
            "phases": {},
        }

        for phase, metrics_list in self.metrics.items():
            if metrics_list:
                # Get final metrics
                final = metrics_list[-1]

                # Compute averages over last 100 steps
                recent = metrics_list[-100:] if len(metrics_list) >= 100 else metrics_list

                summary["phases"][phase] = {
                    "num_steps": len(metrics_list),
                    "final": {
                        "loss": final["loss"],
                        "l2_loss": final["l2_loss"],
                        "l0_norm": final["l0_norm"],
                        "num_dead_features": final["num_dead_features"],
                    },
                    "avg_last_100": {
                        "loss": sum(m["loss"] for m in recent) / len(recent),
                        "l2_loss": sum(m["l2_loss"] for m in recent) / len(recent),
                        "l0_norm": sum(m["l0_norm"] for m in recent) / len(recent),
                    },
                }

                # Add JumpReLU-specific summary if present
                if final.get("l0_coeff") is not None:
                    summary["phases"][phase]["final"]["l0_coeff"] = final["l0_coeff"]
                if final.get("target_l0") is not None:
                    summary["phases"][phase]["final"]["target_l0"] = final["target_l0"]
                if final.get("decomp_penalty") is not None:
                    summary["phases"][phase]["final"]["decomp_penalty"] = final["decomp_penalty"]

        return summary

    def save(self):
        """Save all logs to files."""
        # Save detailed metrics per phase
        for phase, metrics_list in self.metrics.items():
            if metrics_list:
                filepath = self.output_dir / f"{self.run_name}_{phase}_metrics.json"
                with open(filepath, 'w') as f:
                    json.dump({
                        "phase": phase,
                        "num_steps": len(metrics_list),
                        "metrics": metrics_list,
                    }, f, indent=2)

        # Save summary
        summary = self.get_summary()
        summary["config"] = self.config
        summary_path = self.output_dir / f"{self.run_name}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        # Save combined metrics (for easy analysis)
        all_metrics_path = self.output_dir / f"{self.run_name}_all_metrics.json"
        with open(all_metrics_path, 'w') as f:
            json.dump({
                "config": self.config,
                "summary": summary,
                "metrics": self.metrics,
            }, f, indent=2)

    def print_summary(self):
        """Print a formatted summary to stdout."""
        summary = self.get_summary()

        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        print(f"Total time: {summary['total_time_seconds'] / 60:.1f} minutes")
        print(f"Log saved to: {self.output_dir}")

        for phase, phase_summary in summary.get("phases", {}).items():
            print(f"\n{phase.upper()}:")
            print(f"  Steps: {phase_summary['num_steps']}")
            final = phase_summary['final']
            print(f"  Final loss: {final['loss']:.4f}")
            print(f"  Final L2: {final['l2_loss']:.4f}")
            print(f"  Final L0: {final['l0_norm']:.1f}")
            print(f"  Dead features: {final['num_dead_features']}")

            if 'l0_coeff' in final:
                print(f"  Final L0 coeff: {final['l0_coeff']:.6f}")
            if 'target_l0' in final:
                print(f"  Target L0: {final['target_l0']}")
            if 'decomp_penalty' in final:
                print(f"  Decomp penalty: {final['decomp_penalty']:.4f}")

        print("=" * 60)
