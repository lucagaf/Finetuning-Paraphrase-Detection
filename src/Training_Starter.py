from GLUEDataModule import GLUEDataModule
from GLUETransformer import GLUETransformer
import lightning as L
from lightning.pytorch.loggers import WandbLogger
import wandb

class Training_Starter:

    def __init__(self, model_name,
                 wandb_environment,
                 task_name,
                 seed,
                 wandb_project_name,
                 wandb_apikey,
                 lr,
                 batch_size,
                 warmup_steps,
                 beta1
                 ):


        self.epochs = 3  # do not change this
        wandb.login(key=wandb_apikey)
        self.logger = WandbLogger(project=wandb_project_name, group=wandb_environment, name=f'{lr=}_{batch_size=}_{warmup_steps=}_{beta1=}')

        self.model_name = model_name
        self.task_name = task_name
        self.seed = seed
        self.lr = lr
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.beta1 = beta1

    def start_single_trainingsrun(self):
        L.seed_everything(self.seed)
        dm = GLUEDataModule(
            model_name_or_path=self.model_name,
            task_name=self.task_name,
        )
        dm.setup("fit")
        model = GLUETransformer(
            model_name_or_path=self.model_name,
            num_labels=dm.num_labels,
            eval_splits=dm.eval_splits,
            task_name=dm.task_name,
            learning_rate = self.lr,
            warmup_steps = self.warmup_steps,
            train_batch_size = self.batch_size,
            beta1 = self.beta1,
        )

        trainer = L.Trainer(
            max_epochs=self.epochs,
            accelerator="auto",
            devices=1,
            logger=self.logger,
        )
        trainer.fit(model, datamodule=dm)

