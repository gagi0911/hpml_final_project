import copy
import torch
from torch import nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import time
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import random

from typing import Optional

from labml_helpers.module import Module
from labml_nn.transformers.mha import MultiHeadAttention
# from labml_nn.transformers import TransformerLayer
from labml_nn.utils import clone_module_list

# from labml_nn.experiments.cifar10 import CIFAR10Configs
# from labml_nn.transformers import TransformerConfigs
# from labml.configs import option
#
# from modelsummary import summary


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias1=True,
            bias2=True,
            drop=0.1,
    ):
        super().__init__()
        out_features = in_features
        hidden_features = hidden_features or in_features

        self.dropout = nn.Dropout(drop)
        linear_layer = nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias1)
        self.act = act_layer()

        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias2)


    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        # x = self.norm(x)
        x = self.fc2(x)
        # x = self.dropout(x)
        return x





class ParallelMLP(nn.Module):
    def __init__(self, in_features, hidden_features, drop):
        super(ParallelMLP, self).__init__()
        self.mlps = nn.ModuleList([Mlp(in_features, hidden_features, drop) for _ in range(4)])
        self.avg_mlp = Mlp(in_features, hidden_features, drop)


    def forward(self, x, mlp_idx=None):
        if mlp_idx is not None:
            # mlp_idx = random.randint(0, 3)
            return self.mlps[mlp_idx](x)
            # outputs = [mlp(x) for mlp in self.mlps]
            # return outputs[mlp_idx]
            
        else:
            
            for layer in range(len(self.avg_mlp.fc1.weight)):
                weights = torch.stack([mlp.fc1.weight for mlp in self.mlps])
                biases = torch.stack([mlp.fc1.bias for mlp in self.mlps])
                self.avg_mlp.fc1.weight.data = torch.mean(weights, dim=0)
                self.avg_mlp.fc1.bias.data = torch.mean(biases, dim=0)

            for layer in range(len(self.avg_mlp.fc2.weight)):
                weights = torch.stack([mlp.fc2.weight for mlp in self.mlps])
                biases = torch.stack([mlp.fc2.bias for mlp in self.mlps])
                self.avg_mlp.fc2.weight.data = torch.mean(weights, dim=0)
                self.avg_mlp.fc2.bias.data = torch.mean(biases, dim=0)            
            
            return self.avg_mlp(x)




class PatchEmbeddings(Module):
    """
    <a id="PatchEmbeddings"></a>

    ## Get patch embeddings

    The paper splits the image into patches of equal size and do a linear transformation
    on the flattened pixels for each patch.

    We implement the same thing through a convolution layer, because it's simpler to implement.
    """

    def __init__(self, d_model: int, patch_size: int, in_channels: int):
        """
        * `d_model` is the transformer embeddings size
        * `patch_size` is the size of the patch
        * `in_channels` is the number of channels in the input image (3 for rgb)
        """
        super().__init__()

        # We create a convolution layer with a kernel size and and stride length equal to patch size.
        # This is equivalent to splitting the image into patches and doing a linear
        # transformation on each patch.
        self.conv = nn.Conv2d(in_channels, d_model, patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor):
        """
        * `x` is the input image of shape `[batch_size, channels, height, width]`
        """
        # Apply convolution layer
        x = self.conv(x)
        # Get the shape.
        bs, c, h, w = x.shape
        # Rearrange to shape `[patches, batch_size, d_model]`
        x = x.permute(2, 3, 0, 1)
        x = x.view(h * w, bs, c)

        # Return the patch embeddings
        return x

class LearnedPositionalEmbeddings(Module):
    """
    <a id="LearnedPositionalEmbeddings"></a>

    ## Add parameterized positional encodings

    This adds learned positional embeddings to patch embeddings.
    """

    def __init__(self, d_model: int, max_len: int = 5_000):
        """
        * `d_model` is the transformer embeddings size
        * `max_len` is the maximum number of patches
        """
        super().__init__()
        # Positional embeddings for each location
        self.positional_encodings = nn.Parameter(torch.zeros(max_len, 1, d_model), requires_grad=True)

    def forward(self, x: torch.Tensor):
        """
        * `x` is the patch embeddings of shape `[patches, batch_size, d_model]`
        """
        # Get the positional embeddings for the given patches
        pe = self.positional_encodings[:x.shape[0]]
        # Add to patch embeddings and return
        return x + pe

class ClassificationHead(Module):
    """
    <a id="ClassificationHead"></a>

    ## MLP Classification Head

    This is the two layer MLP head to classify the image based on `[CLS]` token embedding.
    """
    def __init__(self, d_model: int, n_hidden: int, n_classes: int):
        """
        * `d_model` is the transformer embedding size
        * `n_hidden` is the size of the hidden layer
        * `n_classes` is the number of classes in the classification task
        """
        super().__init__()
        # First layer
        self.linear1 = nn.Linear(d_model, n_hidden)
        # Activation
        self.act = nn.GELU()
        # Second layer
        self.linear2 = nn.Linear(n_hidden, n_classes)

    def forward(self, x: torch.Tensor):
        """
        * `x` is the transformer encoding for `[CLS]` token
        """
        # First layer and activation
        x = self.act(self.linear1(x))
        # Second layer
        x = self.linear2(x)

        #
        return x



class Block(nn.Module):
    def __init__(self, d_model, self_attn, feed_forward, dropout_prob):
        super(Block, self).__init__()

        self.self_attn = self_attn

        self.feed_forward = feed_forward

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x, mlp_index=None):

        attn_output = self.self_attn(x)

        x = self.norm1(x + self.dropout(attn_output))
        

        ff_output = self.feed_forward(x, mlp_index)

            

        x = self.norm2(x + self.dropout(ff_output))
        return x





class TransformerLayer(nn.Module):
    """
    <a id="TransformerLayer"></a>

    ## Transformer Layer

    This can act as an encoder layer or a decoder layer.

    🗒 Some implementations, including the paper seem to have differences
    in where the layer-normalization is done.
    Here we do a layer normalization before attention and feed-forward networks,
    and add the original residual vectors.
    Alternative is to do a layer normalization after adding the residuals.
    But we found this to be less stable when training.
    We found a detailed discussion about this in the paper
     [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745).
    """

    def __init__(self, *,
                 d_model: int,
                 self_attn: MultiHeadAttention,
                 src_attn: MultiHeadAttention = None,
                 feed_forward: ParallelMLP,
                 dropout_prob: float):
        """
        * `d_model` is the token embedding size
        * `self_attn` is the self attention module
        * `src_attn` is the source attention module (when this is used in a decoder)
        * `feed_forward` is the feed forward module
        * `dropout_prob` is the probability of dropping out after self attention and FFN
        """
        super().__init__()
        self.size = d_model
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(dropout_prob)
        self.norm_self_attn = nn.LayerNorm([d_model])
        if self.src_attn is not None:
            self.norm_src_attn = nn.LayerNorm([d_model])
        self.norm_ff = nn.LayerNorm([d_model])
        # Whether to save input to the feed forward layer
        self.is_save_ff_input = False

    def forward(self, *,
                x: torch.Tensor,
                mlp_idx: Optional[int] = None,
                mask: torch.Tensor= None,
                src: torch.Tensor = None,
                src_mask: torch.Tensor = None):
        # Normalize the vectors before doing self attention
        z = self.norm_self_attn(x)
        # Run through self attention, i.e. keys and values are from self
        self_attn = self.self_attn(query=z, key=z, value=z, mask=mask)
        # Add the self attention results
        x = x + self.dropout(self_attn)

        # If a source is provided, get results from attention to source.
        # This is when you have a decoder layer that pays attention to 
        # encoder outputs
        if src is not None:
            # Normalize vectors
            z = self.norm_src_attn(x)
            # Attention to source. i.e. keys and values are from source
            attn_src = self.src_attn(query=z, key=src, value=src, mask=src_mask)
            # Add the source attention results
            x = x + self.dropout(attn_src)

        # Normalize for feed-forward
        z = self.norm_ff(x)
        # Save the input to the feed forward layer if specified
        if self.is_save_ff_input:
            self.ff_input = z.clone()
        # Pass through the feed-forward network
        ff = self.feed_forward(z, mlp_idx)
        # Add the feed-forward results back
        x = x + self.dropout(ff)

        return x




class VisionTransformer(Module):
    """
    ## Vision Transformer
    This combines the [patch embeddings](#PatchEmbeddings),
    [positional embeddings](#LearnedPositionalEmbeddings),
    transformer and the [classification head](#ClassificationHead).
    """
    def __init__(self, transformer_layer: TransformerLayer, n_layers: int,
                 patch_emb: PatchEmbeddings, pos_emb: LearnedPositionalEmbeddings,
                 classification: ClassificationHead):
        """
        * `transformer_layer` is a copy of a single [transformer layer](../models.html#TransformerLayer).
         We make copies of it to make the transformer with `n_layers`.
        * `n_layers` is the number of [transformer layers](../models.html#TransformerLayer).
        * `patch_emb` is the [patch embeddings layer](#PatchEmbeddings).
        * `pos_emb` is the [positional embeddings layer](#LearnedPositionalEmbeddings).
        * `classification` is the [classification head](#ClassificationHead).
        """
        super().__init__()
        # Patch embeddings
        self.patch_emb = patch_emb
        self.pos_emb = pos_emb
        # Classification head
        self.classification = classification
        # Make copies of the transformer layer
        self.transformer_layers = clone_module_list(transformer_layer, n_layers)

        # `[CLS]` token embedding
        self.cls_token_emb = nn.Parameter(torch.randn(1, 1, transformer_layer.size), requires_grad=True)
        # Final normalization layer
        self.ln = nn.LayerNorm([transformer_layer.size])

    def forward(self, x: torch.Tensor, mlp_idx=None):
        """
        * `x` is the input image of shape `[batch_size, channels, height, width]`
        """
        # Get patch embeddings. This gives a tensor of shape `[patches, batch_size, d_model]`
        
        x = self.patch_emb(x)
        # Add positional embeddings
        x = self.pos_emb(x)
        # Concatenate the `[CLS]` token embeddings before feeding the transformer
        cls_token_emb = self.cls_token_emb.expand(-1, x.shape[1], -1)
        x = torch.cat([cls_token_emb, x])

        # Pass through transformer layers with no attention masking
        for idx, layer in enumerate(self.transformer_layers):
            x = layer(x=x, mlp_idx=mlp_idx, mask=None)


 
        # Get the transformer output of the `[CLS]` token (which is the first in the sequence).
        x = x[0]

        # Layer normalization
        x = self.ln(x)

        # Classification head, to get logits
        x = self.classification(x)

        #
        return x

def main(rank, world_size, batch_size, d_model, epochs, cifar100, save_path):

    
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )


    train_transform = transforms.Compose([# Pad and crop
                                        transforms.RandomCrop(32, padding=4),
                                        # Random horizontal flip
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    test_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    if cifar100:
        train_dataset = datasets.CIFAR100(root='/scratch/hl5035/hpml/project/data/cifar100', train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR100(root='/scratch/hl5035/hpml/project/data/cifar100', train=False, download=True, transform=test_transform)
    else:
        train_dataset = datasets.CIFAR10(root='/scratch/hl5035/hpml/project/data/cifar10', train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR10(root='/scratch/hl5035/hpml/project/data/cifar10', train=False, download=True, transform=test_transform)
    batch_size = batch_size // world_size

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True, 
    )

    
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False, 
    )


    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size, 
        sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        100,
        sampler=test_sampler)

    # d_model = 300
    patch_size = 4
    in_channels = 3
    max_len = d_model
    n_hidden = 3072
    n_classes = 100 if cifar100 else 10

    patch_emb = PatchEmbeddings(d_model, patch_size, in_channels)
    pos_emb = LearnedPositionalEmbeddings(d_model, max_len)
    
    classification_head = ClassificationHead(d_model, n_hidden, n_classes)


    sff_hidden = 1024

    sff_dropout_prob = 0.1
    
  
    expert_ff = ParallelMLP(in_features=d_model, hidden_features=sff_hidden, drop=sff_dropout_prob)

    attn = MultiHeadAttention(d_model=d_model, heads=12, dropout_prob=0.1)

    dropout_prob = 0.1
  

    n_layers = 12

    block_layer  = Block(d_model=d_model, self_attn=attn,
                                      feed_forward=expert_ff,
                                     dropout_prob=dropout_prob)
    
    encoder_layer = TransformerLayer(d_model=d_model, self_attn=attn,
                                     src_attn=None, feed_forward=expert_ff,
                                     dropout_prob=dropout_prob)

    RRM_vit = VisionTransformer(encoder_layer, n_layers, patch_emb, pos_emb, classification_head)


    # device = torch.device(f'cuda:{rank}')
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    RRM_vit.to(device)
    
    
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        RRM_vit = nn.DataParallel(RRM_vit)
    
    # RRM_vit = DDP(RRM_vit, device_ids=[rank])

    # epochs = 100
    optimizer = torch.optim.Adam(lr=2.5e-4, params=RRM_vit.parameters(), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = nn.CrossEntropyLoss()
    # summary(modified_vit, (3, 32, 32))

    def get_total_params(module: torch.nn.Module):
        total_params = 0
        for param in module.parameters():
            total_params += param.numel()
        return total_params

    print ('Total parameters in model: {:,}'.format(get_total_params(RRM_vit)))

    min = 100
    for epoch in range(epochs):
        total = 0
        temp = 0
        
        ep_start = time.time()


        for i, (images, labels) in enumerate(train_loader):
            mlp_idx = random.randint(0, 3)
            start = time.time()
            images = images.to(device)
            labels = labels.to(device)
            outputs = RRM_vit.forward(images, mlp_idx)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += (time.time() - start)

            if (i + 1) % 50 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}], Time: {(total - temp):.2f},Loss: {loss.item():.4f}')
                temp = total
        
        scheduler.step()

        test_loss = 0


        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = RRM_vit.forward(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            test_loss /= len(test_loader)
            print(f'Time: {(time.time() - ep_start):.2f}, Test loss: {test_loss:.4f}, acc: {100 * correct / total}%')

        # if (epoch) % 10 == 0 and rank == 0:
        if min > test_loss:
            min = test_loss
            torch.save(RRM_vit.state_dict(), f"/scratch/hl5035/final_proj/models/{save_path}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Distributed deep learning')
    parser.add_argument('--gpu', default=1, type=int, help='no. of gpus')
    parser.add_argument('--epochs', default=32, type=int, help='no. of epochs')
    parser.add_argument('--batch', default=32, type=int, help='batch size')
    parser.add_argument('--cifar100', action="store_true", help='use cifar 100 dataset')
    parser.add_argument('--out', default='/scratch/hl5035/final_proj/models/new', type=str, help='model output path')
    parser.add_argument('--dmodel', default=300, type=int, help='d_model embedding size')
    args = parser.parse_args()
    gpu_count = args.gpu


    mp.spawn(main, args=(gpu_count, args.batch, args.dmodel, args.epochs, args.cifar100, args.out), nprocs=gpu_count, join=True)