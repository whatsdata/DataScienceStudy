import json
import math
import time

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import seaborn
import torch
import torch.nn as nn
import torch.optim as optim

from vae import loss_function


def hello_helper():
    print("Hello from a5_helper.py!")


def load_coco_captions(path: str = "./coco.pt"):
    """
    Download and load serialized COCO data from coco.pt
    It contains a dictionary of
    "train_images" - resized training images (112x112)
    "val_images" - resized validation images (112x112)
    "train_captions" - tokenized and numericalized training captions
    "val_captions" - tokenized and numericalized validation captions
    "vocab" - caption vocabulary, including "idx_to_token" and "token_to_idx"

    Returns: a data dictionary
  """
    data_dict = torch.load(path)
    # print out all the keys and values from the data dictionary
    for k, v in data_dict.items():
        if type(v) == torch.Tensor:
            print(k, type(v), v.shape, v.dtype)
        else:
            print(k, type(v), v.keys())

    assert data_dict["train_images"].size(0) == data_dict["train_captions"].size(
        0
    ) and data_dict["val_images"].size(0) == data_dict["val_captions"].size(
        0
    ), "shapes of data mismatch!"

    print("\nTrain images shape: ", data_dict["train_images"].shape)
    print("Train caption tokens shape: ", data_dict["train_captions"].shape)
    print("Validation images shape: ", data_dict["val_images"].shape)
    print("Validation caption tokens shape: ", data_dict["val_captions"].shape)
    print(
        "total number of caption tokens: ", len(data_dict["vocab"]["idx_to_token"])
    )
    print(
        "mappings (list) from index to caption token: ",
        data_dict["vocab"]["idx_to_token"],
    )
    print(
        "mappings (dict) from caption token to index: ",
        data_dict["vocab"]["token_to_idx"],
    )

    return data_dict


def get_toy_data(path: str = "final_data.json"):
    return json.load(open(path))


def train_captioner(
    model,
    image_data,
    caption_data,
    num_epochs,
    batch_size,
    learning_rate,
    lr_decay=1,
    verbose=True,
    device: torch.device = torch.device("cpu"),
):
    """
    Run optimization to train the model.
    """

    model = model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), learning_rate
    )
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: lr_decay ** epoch
    )

    # sample minibatch data
    iter_per_epoch = math.ceil(image_data.shape[0] // batch_size)
    loss_history = []

    for i in range(num_epochs):
        start_t = time.time()
        for j in range(iter_per_epoch):
            images, captions = (
                image_data[j * batch_size : (j + 1) * batch_size],
                caption_data[j * batch_size : (j + 1) * batch_size],
            )
            images = images.to(device)
            captions = captions.to(device)

            loss = model(images, captions)
            optimizer.zero_grad()
            loss.backward()
            loss_history.append(loss.item())
            optimizer.step()
        end_t = time.time()

        if verbose:
            print(
                "(Epoch {:2d} / {:2d}) loss: {:7.4f} time per epoch: {:.2f}s" \
                .format(
                    i, num_epochs, loss.item(), end_t - start_t
                )
            )

        lr_scheduler.step()

    # plot the training losses
    plt.plot(loss_history)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training loss history")
    plt.show()
    return model, loss_history


def decode_captions(captions, idx_to_word):
    """
    Decoding caption indexes into words.

    Args:
        captions: Caption indexes in a tensor of shape (N, T).
        idx_to_word: Mapping from the vocab index to word.

    Returns:
        decoded: A sentence (or a list of N sentences).
    """
    singleton = captions.ndim == 1
    captions = captions[None] if singleton else captions

    decoded = []
    N, T = captions.shape
    for i in range(N):
        words = []
        for t in range(T):
            word = idx_to_word[captions[i, t]]
            if word != "<NULL>":
                words.append(word)
            if word == "<END>":
                break
        decoded.append(" ".join(words))

    if singleton:
        decoded = decoded[0]
    return decoded


def train(
    model,
    train_dataloader,
    val_dataloader,
    loss_func,
    num_epochs,
    batch_size=32,
    warmup_lr=6e-6,
    warmup_interval=1000,
    lr=6e-4,
    device=torch.device("cpu"),
):
    print("Training started...")
    if warmup_interval is None:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, betas=(0.9, 0.995), eps=1e-9
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=warmup_lr, betas=(0.9, 0.995), eps=1e-9
        )
    iteration = 0
    for epoch_num in range(num_epochs):
        epoch_loss = []
        model.train()
        for it in train_dataloader:
            inp, inp_pos, out, out_pos = it
            model = model.to(device)
            inp_pos = inp_pos.to(device)
            out_pos = out_pos.to(device)
            out = out.to(device)
            inp = inp.to(device)
            gnd = out[:, 1:].contiguous().view(-1).long()
            optimizer.zero_grad()

            pred = model(inp.long(), inp_pos, out.long(), out_pos)
            loss = loss_func(pred, gnd)
            epoch_loss.append(loss.item())
            if warmup_interval is not None and iteration == warmup_interval:
                print(
                    f"End of warmup. Swapping learning rates from {warmup_lr} to {lr}"
                )
                for param_group in optimizer.param_groups:
                    warmup_lr = lr
                    param_group["lr"] = lr

            loss.backward()
            optimizer.step()
            iteration = iteration + 1
        avg_epoch_loss = sum(epoch_loss) / len(epoch_loss)
        val_loss, val_acc = val(model, val_dataloader, loss_func, batch_size)
        loss_hist = avg_epoch_loss / (batch_size * 4)
        print(
            f"[epoch: {epoch_num+1}]",
            "[loss: ",
            f"{loss_hist:.4f}",
            "]",
            "val_loss: [val_loss ",
            f"{val_loss:.4f}",
            "]",
        )

    return model


def val(model, dataloader, loss_func, batch_size, device=torch.device("cpu")):
    model.eval()
    epoch_loss = []
    num_correct = 0
    total = 0
    for it in dataloader:
        inp, inp_pos, out, out_pos = it

        model = model.to(device)
        inp_pos = inp_pos.to(device)
        out_pos = out_pos.to(device)
        out = out.to(device)
        inp = inp.to(device)
        gnd = out[:, 1:].contiguous().view(-1).long()
        pred = model(inp.long(), inp_pos, out.long(), out_pos)
        loss = loss_func(pred, gnd)

        pred_max = pred.max(1)[1]
        gnd = gnd.contiguous().view(-1)

        n_correct = pred_max.eq(gnd)
        n_correct = n_correct.sum().item()
        num_correct = num_correct + n_correct

        total = total + len(pred_max)
        epoch_loss.append(loss.item())

    avg_epoch_loss = sum(epoch_loss) / len(epoch_loss)
    return avg_epoch_loss / (batch_size * 4), n_correct / total


def inference(model, inp_exp, inp_exp_pos, out_pos_exp, out_seq_len):
    model.eval()
    y_init = torch.LongTensor([14]).unsqueeze(0).cuda().view(1, 1)

    ques_emb = model.emb_layer(inp_exp)
    q_emb_inp = ques_emb + inp_exp_pos
    enc_out = model.encoder(q_emb_inp)
    for i in range(out_seq_len - 1):
        ans_emb = model.emb_layer(y_init)
        a_emb_inp = ans_emb + out_pos_exp[:, : y_init.shape[1], :]
        dec_out = model.decoder(a_emb_inp, enc_out, None)
        _, next_word = torch.max(
            dec_out[0, y_init.shape[1] - 1 : y_init.shape[1]], dim=1
        )

        y_init = torch.cat([y_init, next_word.view(1, 1)], dim=1)
    return y_init, model


def draw(data, x, y, ax):
    seaborn.heatmap(
        data,
        xticklabels=x,
        square=True,
        yticklabels=y,
        vmin=0.0,
        vmax=1.0,
        cbar=False,
        ax=ax,
    )


### Helper Functions for VAE and GAN
def show_images(images):
    images = torch.reshape(
        images, [images.shape[0], -1]
    )  # images reshape to (batch_size, D)
    sqrtn = int(math.ceil(math.sqrt(images.shape[0])))
    sqrtimg = int(math.ceil(math.sqrt(images.shape[1])))

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis("off")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect("equal")
        plt.imshow(img.reshape([sqrtimg, sqrtimg]))
    return


def count_params(model):
    """Count the number of parameters in the model"""
    param_count = sum([p.numel() for p in model.parameters()])
    return param_count


def initialize_weights(m):
    """Initializes the weights of a torch.nn model using xavier initialization"""
    if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_uniform_(m.weight.data)


def one_hot(labels, class_size):
    """
    Create one hot label matrix of size (N, C)

    Inputs:
    - labels: Labels Tensor of shape (N,) representing a ground-truth label
    for each MNIST image
    - class_size: Scalar representing of target classes our dataset
    Outputs:
    - targets: One-hot label matrix of (N, C), where targets[i, j] = 1 when
    the ground truth label for image i is j, and targets[i, :j] &
    targets[i, j + 1:] are equal to 0
    """
    targets = torch.zeros(labels.size(0), class_size)
    for i, label in enumerate(labels):
        targets[i, label] = 1
    return targets


def train_vae(epoch, model, train_loader, cond=False):
    """
    Train a VAE or CVAE!

    Inputs:
    - epoch: Current epoch number
    - model: VAE model object
    - train_loader: PyTorch Dataloader object that contains our training data
    - cond: Boolean value representing whether we're training a VAE or
    Conditional VAE
    """
    model.train()
    train_loss = 0
    num_classes = 10
    loss = None
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for batch_idx, (data, labels) in enumerate(train_loader):
        data = data.to(device="cuda:0")
        if cond:
            one_hot_vec = one_hot(labels, num_classes).to(device="cuda")
            recon_batch, mu, logvar = model(data, one_hot_vec)
        else:
            recon_batch, mu, logvar = model(data)
        optimizer.zero_grad()
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.data
        optimizer.step()
    print("Train Epoch: {} \tLoss: {:.6f}".format(epoch, loss.data))
