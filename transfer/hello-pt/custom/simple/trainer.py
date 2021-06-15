from .lenet import LeNet

import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLConstants, ShareableKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, ShareableKey, ShareableValue
from nvflare.apis.trainer import Trainer
from nvflare.common.signal import Signal
from nvflare.utils.fed_utils import generate_failure


class SimpleTrainer(Trainer):
    def __init__(self, epochs_per_round, validation_interval, lr=1e-4):
        super().__init__()
        self.epochs_per_round = epochs_per_round
        self.validation_interval = validation_interval
        self.lr = lr
        self.logger.info(
            f"epochs_per_round: {epochs_per_round}, validation_interval: {validation_interval}"
        )

    def setup(self, fl_ctx):
        self.model = LeNet()
        self.optimizer = optim.Adadelta(self.model.parameters(), lr=self.lr)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=0.7)
        train_kwargs = {"batch_size": 10}
        test_kwargs = {"batch_size": 1}
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        self.dataset1 = datasets.MNIST(
            "../data", train=True, download=True, transform=self.transform
        )
        self.dataset2 = datasets.MNIST("../data", train=False, transform=self.transform)
        self.train_loader = torch.utils.data.DataLoader(self.dataset1, **train_kwargs)
        self.test_loader = torch.utils.data.DataLoader(self.dataset2, **test_kwargs)
        self.test_data_size = len(self.test_loader.dataset)

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self.setup(fl_ctx)
        elif event_type == EventType.END_RUN:
            self.teardown()

    def train(
        self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal
    ) -> Shareable:
        """
        This function is an extended function from the super class.
        As a supervised learning based trainer, the train function will run
        evaluate and train engines based on model weights from `shareable`.
        After fininshing training, a new `Shareable` object will be submitted
        to server for aggregation.

        Args:
            shareable: the `Shareable` object acheived from server.
            fl_ctx: the `FLContext` object achieved from server.
            abort_signal: if triggered, the training will be aborted.

        Returns:
            a new `Shareable` object to be submitted to server for aggregation.
        """
        # check abort signal
        self.logger.info(f"abort signal: {abort_signal.triggered}")
        # self.logger.info(fl_ctx)
        if abort_signal.triggered:
            shareable = generate_failure(fl_ctx=fl_ctx, reason="abort signal triggered")
            return shareable
        # retrieve model weights download from server's shareable
        model_weights = shareable[ShareableKey.MODEL_WEIGHTS]
        # load achieved model weights for the network (saved in fl_ctx)

        local_var_dict = self.model.state_dict()
        model_keys = model_weights.keys()
        for var_name in local_var_dict:
            if var_name in model_keys:
                weights = model_weights[var_name]
                try:
                    local_var_dict[var_name] = torch.as_tensor(weights)
                except Exception as e:
                    raise ValueError(
                        "Convert weight from {} failed with error: {}".format(
                            var_name, str(e)
                        )
                    )

        # replace local model weights with received weights
        self.model.load_state_dict(local_var_dict)

        # FLContext contains the CURRENT_ROUND
        starting_epochs = (
            fl_ctx.get_prop(FLConstants.CURRENT_ROUND) * self.epochs_per_round + 1
        )
        ending_epochs = starting_epochs + self.epochs_per_round + 1

        for epoch in range(starting_epochs, ending_epochs):
            # set the model to train mode
            self.model.train()
            for batch_idx, (data, target) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                self.optimizer.step()
                if batch_idx % 100 == 0:
                    self.logger.info(f"Training with batch idex= {batch_idx}")
            self.scheduler.step()
            if epoch % self.validation_interval == 0:
                self.model.eval()
                test_loss = 0
                correct = 0
                with torch.no_grad():
                    for data, target in self.test_loader:
                        output = self.model(data)
                        test_loss += F.nll_loss(
                            output, target, reduction="sum"
                        ).item()  # sum up batch loss
                        pred = output.argmax(
                            dim=1, keepdim=True
                        )  # get the index of the max log-probability
                        correct += pred.eq(target.view_as(pred)).sum().item()
                test_loss /= len(self.test_loader.dataset)
                self.logger.info(
                    f"Test set: Average loss: {test_loss:.4f},"
                    f" Accuracy: {correct}/{self.test_data_size} ({100.*correct/self.test_data_size:.0f}%)"
                )

        local_state_dict = self.model.state_dict()
        local_model_dict = {}
        for var_name in local_state_dict:
            try:
                local_model_dict[var_name] = local_state_dict[var_name].cpu().numpy()
            except Exception as e:
                raise ValueError(
                    "Convert weight from {} failed with error: {}".format(
                        var_name, str(e)
                    )
                )
        meta_data = {}
        meta_data[FLConstants.INITIAL_METRICS] = 0.1
        meta_data[FLConstants.CURRENT_LEARNING_RATE] = self.lr

        shareable = Shareable()
        shareable[ShareableKey.TYPE] = ShareableValue.TYPE_WEIGHT_DIFF
        shareable[ShareableKey.DATA_TYPE] = ShareableValue.DATA_TYPE_UNENCRYPTED
        shareable[ShareableKey.MODEL_WEIGHTS] = local_model_dict
        shareable[ShareableKey.META] = meta_data

        return shareable
