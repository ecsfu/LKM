
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

import matplotlib.pyplot as plt
 
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk
# Import from local files
class GPTDatasetV1(Dataset):
    def __init__(self, dataset_dir="train_dataset"):
        self.input_ids = []
        self.label_ids = []

        # Tokenize the entire text
        train_dataset_loaded = load_from_disk(dataset_dir)

        
        for data in train_dataset_loaded:
            input_chunk = data['input_ids']
            label_chunk = data['label_ids']
            self.input_ids.append(torch.tensor(input_chunk))
            self.label_ids.append(torch.tensor(label_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.label_ids[idx]


def create_dataloader(dataset_dir,batch_size=4,
                         shuffle=True, drop_last=True, num_workers=0):
    # Initialize the tokenizer
    

    # Create dataset
    dataset = GPTDatasetV1(dataset_dir)

    # Create dataloader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader





def calc_loss_batch(input_batch, label_batch, model, device):
    input_batch, label_batch = input_batch.to(device), label_batch.to(device)
    output = model(input_batch,labels=label_batch)
    loss = output.loss
    # loss2 = torch.nn.functional.cross_entropy(logits.flatten(0, 1), label_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

 
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss





def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = -1

    # Main training loop
    for epoch in tqdm(range(num_epochs)):
        model.train()  # Set model to training mode

        for input_batch, target_batch in tqdm(train_loader, desc="Training", unit="it"):
            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # Calculate loss gradients
            optimizer.step()  # Update model weights using loss gradients
            tokens_seen += input_batch.numel()
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
        torch.save(model.state_dict(), f"./lkm_ckpt0909/model_ep{epoch}.pt")
        # # Print a sample text after each epoch
        # generate_and_print_sample(
        #     model, tokenizer, device, start_context
        # )

    return train_losses, val_losses, track_tokens_seen


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots()

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    # plt.show()


def main(settings):
    from transformers import Qwen3ForCausalLM,Qwen3Config
    lkmconfig = Qwen3Config(vocab_size=19482,
        hidden_size=384,
        intermediate_size=1512,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_key_value_heads=6,
        head_dim=32,
        hidden_act="silu",
        max_position_embeddings=252,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=False,
        tie_word_embeddings=True,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        use_sliding_window=False,
        sliding_window=None,
        max_window_layers=None,
        layer_types=None,
        attention_dropout=0.1,)
    
    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'当前device:{device}')
    ##############################
    # Download data if necessary
    ##############################

    
    ##############################
    # Initialize model
    ##############################
    print('初始化模型...')
    # model = Qwen3ForCausalLM(lkmconfig)
    model = Qwen3ForCausalLM(lkmconfig)
    total_params = sum(p.numel() for p in model.parameters())
    total_size = total_params / (1024 ** 2)  # MB
    print(f"Total Parameters: {total_params}")
    print(f"Model Size: {total_size:.2f} M")
    lkmconfig.save_pretrained(f"./lkm_ckpt0909/LKM1_{total_size:.2f}M")
    model.to(device)  # no assignment model = model.to(device) necessary for nn.Module classes
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )

    ##############################
    # Set up dataloaders
    ##############################

    # Train/validation ratio
     
    print('加载数据集...')
    train_loader = create_dataloader(
        './train_data/train_dataset3',
        batch_size=settings["batch_size"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )

    val_loader =create_dataloader(
        './train_data/val_dataset3',
        batch_size=settings["batch_size"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )

    ##############################
    # Train model
    ##############################

    print('开始训练...')

    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=settings["num_epochs"], eval_freq=20, eval_iter=3,
    )

    return train_losses, val_losses, tokens_seen, model


if __name__ == "__main__":
    
    
    SETTINGS = {
        "learning_rate": 1e-4,
        "num_epochs": 10,
        "batch_size": 64,
        "weight_decay": 0.1
    }

    ###########################
    # Initiate training
    ###########################

    train_losses, val_losses, tokens_seen, model = main(SETTINGS)

    ###########################
    # After training
    ###########################

    # Plot results
    epochs_tensor = torch.linspace(0, SETTINGS["num_epochs"], len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
    plt.savefig("loss4.pdf")

    # Save and load model
    torch.save(model.state_dict(), "./lkm_ckpt0909/model.pt")

    # model = GPTModel(GPT_CONFIG_124M)
    # model.load_state_dict(torch.load("model.pth", weights_only=True))