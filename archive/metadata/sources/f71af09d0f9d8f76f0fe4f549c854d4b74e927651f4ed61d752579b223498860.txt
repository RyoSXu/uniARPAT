import torch



class Metrics(object):
    """
    Define metrics for evaluation, metrics include:

        - MSE, masked MSE;

        - RMSE, masked RMSE;

        - REL, masked REL;

        - MAE, masked MAE;

        - Threshold, masked threshold.
    """
    def __init__(self, epsilon = 1e-8, **kwargs):
        """
        Initialization.

        Parameters
        ----------

        epsilon: float, optional, default: 1e-8, the epsilon used in the metric calculation.
        """
        super(Metrics, self).__init__()
        self.epsilon = epsilon
    
    def MSE(self, pred, gt, data_mask, clim_time_mean_daily, data_std):
        """
        MSE metric.

        Parameters
        ----------

        pred: tensor, required, the predicted;

        gt: tensor, required, the ground-truth

        Returns
        -------

        The MSE metric.
        """
        sample_mse = torch.mean((pred - gt) ** 2)
        return sample_mse.item()
    
    def RMSE(self, pred, gt, data_mask, clim_time_mean_daily, data_std):
        """
        RMSE metric.

        Parameters
        ----------

        pred: tensor, required, the predicted;

        gt: tensor, required, the ground-truth;


        Returns
        -------

        The RMSE metric.
        """
        sample_mse = torch.mean((pred - gt) ** 2, dim = [1, 2])
        return torch.mean(torch.sqrt(sample_mse)).item()
    
    def MAE(self, pred, gt, data_mask, clim_time_mean_daily, data_std):
        """
        MAE metric.

        Parameters
        ----------
        pred: tensor, required, the predicted

        gt: tensor, required, the ground-truth

        Returns
        -------
        
        The MAE metric.
        """
        sample_mae = torch.mean(torch.abs(pred - gt))
        return sample_mae.item()


class MetricsRecorder(object):
    """
    Metrics Recorder.
    """
    def __init__(self, metrics_list, epsilon = 1e-7, **kwargs):
        """
        Initialization.

        Parameters
        ----------

        metrics_list: list of str, required, the metrics name list used in the metric calcuation.

        epsilon: float, optional, default: 1e-8, the epsilon used in the metric calculation.
        """
        super(MetricsRecorder, self).__init__()
        self.epsilon = epsilon
        self.metrics = Metrics(epsilon = epsilon)
        self.metrics_list = []
        for metric in metrics_list:
            try:
                metric_func = getattr(self.metrics, metric)
                self.metrics_list.append([metric, metric_func, {}])
            except Exception:
                raise NotImplementedError('Invalid metric type.')
    
    def evaluate_batch(self, data_dict):
        """
        Evaluate a batch of the samples.

        Parameters
        ----------

        data_dict: pred and gt


        Returns
        -------

        The metrics dict.
        """
        pred = data_dict['pred']            # (B, C, H, W)
        gt = data_dict['gt']
        data_mask = None
        clim_time_mean_daily = None
        data_std = None
        if "clim_mean" in data_dict:
            clim_time_mean_daily = data_dict['clim_mean']    #(C, H, W)
            data_std = data_dict["std"]

        losses = {}
        for metric_line in self.metrics_list:
            metric_name, metric_func, metric_kwargs = metric_line
            loss = metric_func(pred, gt, data_mask, clim_time_mean_daily, data_std)
            if isinstance(loss, torch.Tensor):
                for i in range(len(loss)):
                    losses[metric_name+str(i)] = loss[i].item()
            else:
                losses[metric_name] = loss

        return losses


def per_sample_spectral_metrics(p, t, eps=1e-8):
    """H2 hygiene: single source of truth for per-sample DOS metrics.

    p, t: [B, L] tensors in TRUE PHYSICAL space (already denormalized).
    Returns dict of [B] tensors: mae, mse, r2.
    Formula frozen to match legacy implementations in model.test_one_step
    and run_ablation_experiments.evaluate_split (bit-identical math).
    """
    mae = torch.mean(torch.abs(p - t), dim=-1)
    mse = torch.mean((p - t) ** 2, dim=-1)
    ss_res = torch.sum((t - p) ** 2, dim=-1)
    ss_tot = torch.sum((t - torch.mean(t, dim=-1, keepdim=True)) ** 2, dim=-1)
    r2 = 1.0 - (ss_res / (ss_tot + eps))
    return {'mae': mae, 'mse': mse, 'r2': r2}
