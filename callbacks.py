from pytorch_lightning.callbacks import Callback
import wandb
 
class LogPredictionsCallback(Callback):
    def __init__(self, n_samples):
        super().__init__()
        self.n_samples = n_samples

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """Called when the validation batch ends."""
 
        # `outputs` comes from `LightningModule.validation_step`
        # which corresponds to our model predictions in this case
        
        # Let's log 20 sample image predictions from first batch
        if batch_idx == 0:
            images, targets = batch['images'].cuda(), batch['targets'].cuda()
            targets = targets/100. if pl_module.classification else targets

            logits = pl_module(images)
            # we can directly use `wandb` for logging custom objects (image, video, audio, modecules and any other custom plot)
            wandb.log({'real_examples': [wandb.Image(x_i, caption=f'Ground Truth: {y_i}\nPrediction: {y_pred}')
                                       for x_i, y_i, y_pred in list(zip(images[:self.n_samples], targets[:self.n_samples], outputs['preds'][:self.n_samples]))]})