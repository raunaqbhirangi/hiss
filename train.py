import copy
import logging
import hydra
from typing import Type
import numpy as np
import torch
from hiss.tasks import Task
from hiss.models import LowFreqPredictor
from hiss.utils.data_utils import get_data_path
from hiss.utils.train_utils import (
    MaskedLoss,
    set_seed,
    create_dsets,
)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg):
    set_seed(cfg.seed)

    # Configure the logger
    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    log.setLevel(logging.DEBUG)

    # Standardized function to extract path to .h5 file containing data
    data_path = get_data_path(cfg.data_env.dataset_dir, cfg.data_env.data_suffix)

    # Instantiate the task using hydra config
    task: Type[Task] = hydra.utils.instantiate(cfg.data_env.task)

    # Creates subsets of CSPDataset object
    train_dset, val_dset = create_dsets(
        datafile=data_path,
        env_task=task,
        train_frac=cfg.train_frac,
        val_dset_path=cfg.data_env.val_dset_path
        if "val_dset_path" in cfg.data_env
        else None,
        append_input_deltas=cfg.append_input_deltas,
        train_subfrac=cfg.train_subfrac if "train_subfrac" in cfg else 1.0,
    )

    # Set up training
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = MaskedLoss(task.loss_fn)

    # Set up dataloaders
    train_data_loader = torch.utils.data.DataLoader(
        train_dset, batch_size=cfg.batch_size, shuffle=True
    )
    val_data_loader = torch.utils.data.DataLoader(
        val_dset, batch_size=cfg.batch_size, shuffle=False
    )

    # Instantiate model using hydra config
    input_dim = train_dset[0][0].shape[-1]
    output_dim = train_dset[0][1].shape[-1]

    model = hydra.utils.instantiate(cfg.model.model, _recursive_=False)(
        input_dim=input_dim, output_dim=output_dim
    ).to(device)

    # Wrapper to downsample model outputs if necessary
    if "low_freq_factor" in cfg:
        if cfg.low_freq_factor is not None:
            model = LowFreqPredictor(model, cfg.low_freq_factor)
            log.info("using low freq predictor")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
    )

    init_loss = evaluate(val_data_loader, criterion, device, model) / output_dim
    log.info(f"Init val loss: {init_loss:.4f}")
    best_loss = copy.copy(init_loss)

    for i in range(cfg.n_epochs):
        train_loss = 0.0
        total_len = 0
        model.train()
        for batch in train_data_loader:
            optimizer.zero_grad()

            inputs, targets, lens = (b.to(device) for b in batch)
            pred = model(inputs.float())

            # Compute and aggregate loss
            loss = criterion(pred, targets.float(), lens)
            train_loss += loss.item()
            total_len += torch.sum(lens).item()

            loss = loss / (output_dim * torch.sum(lens).item())
            loss.backward()
            optimizer.step()
        train_loss = train_loss / (output_dim * total_len)
        val_loss = evaluate(val_data_loader, criterion, device, model) / output_dim

        # Maintain best model
        if i == 0 or val_loss < best_loss:
            best_loss = val_loss
            best_model = copy.deepcopy(model.state_dict())
        model.train()
        log.info(f"Epoch {i}: Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")

        # Save best model every 100 epochs
        if i > 0 and i % 100 == 0:
            torch.save(best_model, "model.pt")

    model.load_state_dict(best_model)
    torch.save(best_model, "model.pt")

    log.info(f"Best Val Loss: {best_loss}")
    log_dict = {"best_loss": best_loss}

    # This can be used to log different task-specific metrics. Generally used to keep
    # track of quantities like accuracies, unnormalized errors etc. that are not part
    # of the loss function.
    if cfg.log_metrics:
        best_metrics = evaluate_metrics(val_data_loader, device, model, task)
        for mval, metric in zip(best_metrics, task.metrics):
            log.info(f"Best Val {metric}: {mval:.4f}")
            log_dict[f"best_{metric}"] = mval


def evaluate(val_loader, criterion, device, model):
    model.eval()
    val_loss = 0.0
    total_len = 0
    for batch in val_loader:
        with torch.no_grad():
            inputs, targets, lens = (b.to(device) for b in batch)
            pred = model(inputs.float())
            loss = criterion(pred, targets.float(), lens)
            val_loss += loss.item()
            total_len += torch.sum(lens).item()
    val_loss = val_loss / total_len
    model.train()
    return val_loss


def evaluate_metrics(val_loader, device, model, task):
    model.eval()
    val_metrics = []
    for batch in val_loader:
        with torch.no_grad():
            inputs, targets, lens = (b.to(device) for b in batch)
            pred = model(inputs.float())
            for p, t, l in zip(pred, targets, lens):
                val_metric = task.compute_metrics(
                    p[:l].to(torch.device("cpu")).numpy(),
                    t[:l].to(torch.device("cpu")).numpy(),
                    val_loader.dataset.dataset.target_std.to(
                        torch.device("cpu")
                    ).numpy(),
                )
                val_metrics.append(val_metric)
    val_metrics = np.concatenate(val_metrics, axis=0)
    model.train()
    return np.mean(np.abs(val_metrics), axis=0)


if __name__ == "__main__":
    main()
