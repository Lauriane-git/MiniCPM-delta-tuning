# using OpenDelta library to fine-tune the model on the GSM8K dataset
# library and tutorial: https://github.com/thunlp/OpenDelta
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from opendelta import AutoDeltaModel, AdapterModel
from bigmodelvis import Visualization
import torch as th
from gsm8k_tools import get_examples, GSMDataset
from transformers import get_scheduler
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt




def train_delta_model(num_epochs = 20):

    # load model, tokenizer

    path = "./MiniCPM-2B-sft-bf16"

    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
        local_files_only=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(path)

    # Add some delta parameters to the model

    print("before modify")
    Visualization(model).structure_graph()
    """
    The white part is the name of the module.
    The green part is the module's type.
    The blue part is the tunable parameters, i.e., the parameters that require grad computation.
    The grey part is the frozen parameters, i.e., the parameters that do not require grad computation.
    The red part is the structure that is repeated and thus folded.
    The purple part is the delta parameters inserted into the backbone model.
    """

    delta_model = AdapterModel(
        backbone_model=model, modified_modules=["mlp"], bottleneck_dim=12
    )
    print("after modify")
    delta_model.log()
    # This will visualize the backbone after modification and other information.

    delta_model.freeze_module(
        exclude=["deltas", "layernorm_embedding"], set_state_dict=True
    )
    print("after freeze")
    delta_model.log()

    # training pipeline for delta model

    train_examples = get_examples("train")
    train_dset = GSMDataset(tokenizer, train_examples)

    device = th.device("cuda")
    model.to(device)
    model.eval()

    train_loader = DataLoader(train_dset, batch_size=16, shuffle=True)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-5)

    num_training_steps = num_epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optim,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    losses = []

    pbar = tqdm(range(num_training_steps))
    for _ in range(num_epochs):
        for batch in train_loader:
            optim.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch, labels=batch["input_ids"])
            loss = outputs[0]
            loss.backward()
            optim.step()
            lr_scheduler.step()
            pbar.update(1)
            pbar.set_description(f"train_loss: {loss.item():.5f}")
            losses.append(loss.item())

    delta_model.save_finetuned("./delta_model")

    return losses

def import_delta_model(model):
    delta_model = AutoDeltaModel.from_finetuned("./delta_model", backbone_model=model)
    return delta_model

def print_graph (losses):
    plt.plot(losses)
    plt.ylabel('Loss')
    plt.xlabel('Batch')
    plt.savefig('loss.png')
    plt.show()

if __name__ == "__main__":
    num_epochs = 20
    losses = train_delta_model(num_epochs)
    print_graph (losses)