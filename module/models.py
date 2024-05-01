import torch
import torch.nn as nn
import timm
from timm.models.layers import to_2tuple,trunc_normal_
from module.layers import CosineSimLayer, AttentionSparseMax

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class ASTModel(nn.Module):
    """
    The AST model.
    :param label_dim: the label dimension, i.e., the number of total classes, it is 527 for AudioSet, 50 for ESC-50, and 35 for speechcommands v2-35
    :param fstride: the stride of patch spliting on the frequency dimension, for 16*16 patchs, fstride=16 means no overlap, fstride=10 means overlap of 6
    :param tstride: the stride of patch spliting on the time dimension, for 16*16 patchs, tstride=16 means no overlap, tstride=10 means overlap of 6
    :param input_fdim: the number of frequency bins of the input spectrogram
    :param input_tdim: the number of time frames of the input spectrogram
    :param imagenet_pretrain: if use ImageNet pretrained model
    :param audioset_pretrain: if use full AudioSet and ImageNet pretrained model
    :param model_size: the model size of AST, should be in [tiny224, small224, base224, base384], base224 and base 384 are same model, but are trained differently during ImageNet pretraining.
    """
    def __init__(self, label_dim=527, fstride=10, tstride=10, input_fdim=128, input_tdim=1024, model_size='base384'):

        super(ASTModel, self).__init__()
        assert timm.__version__ == '0.4.5', 'Please use timm == 0.4.5, the code might not be compatible with newer versions.'

        # override timm input shape restriction
        timm.models.vision_transformer.PatchEmbed = PatchEmbed

        # if AudioSet pretraining is not used (but ImageNet pretraining may still apply)

        if model_size == 'tiny224':
            self.v = timm.create_model('vit_deit_tiny_distilled_patch16_224', pretrained=True)
        elif model_size == 'small224':
            self.v = timm.create_model('vit_deit_small_distilled_patch16_224', pretrained=True)
        elif model_size == 'base224':
            self.v = timm.create_model('vit_deit_base_distilled_patch16_224', pretrained=True)
        elif model_size == 'base384':
            self.v = timm.create_model('vit_deit_base_distilled_patch16_384', pretrained=True)
        else:
            raise Exception('Model size must be one of tiny224, small224, base224, base384.')
        self.original_num_patches = self.v.patch_embed.num_patches
        self.oringal_hw = int(self.original_num_patches ** 0.5)
        self.original_embedding_dim = self.v.pos_embed.shape[2]
        self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, label_dim))

        # automatcially get the intermediate shape
        f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
        num_patches = f_dim * t_dim
        self.v.patch_embed.num_patches = num_patches


        # the linear projection layer
        new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        new_proj.weight = torch.nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
        new_proj.bias = self.v.patch_embed.proj.bias
        self.v.patch_embed.proj = new_proj
        
        new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, self.original_num_patches, self.original_embedding_dim).transpose(1, 2).reshape(1, self.original_embedding_dim, self.oringal_hw, self.oringal_hw)
        # cut (from middle) or interpolate the second dimension of the positional embedding
        if t_dim <= self.oringal_hw:
            new_pos_embed = new_pos_embed[:, :, :, int(self.oringal_hw / 2) - int(t_dim / 2): int(self.oringal_hw / 2) - int(t_dim / 2) + t_dim]
        else:
            new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(self.oringal_hw, t_dim), align_corners=False)
        # cut (from middle) or interpolate the first dimension of the positional embedding
        if f_dim <= self.oringal_hw:
            new_pos_embed = new_pos_embed[:, :, int(self.oringal_hw / 2) - int(f_dim / 2): int(self.oringal_hw / 2) - int(f_dim / 2) + f_dim, :]
        else:
            new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), align_corners=False)
        # flatten the positional embedding
        new_pos_embed = new_pos_embed.reshape(1, self.original_embedding_dim, num_patches).transpose(1,2)
        # concatenate the above positional embedding with the cls token and distillation token of the deit model.
        self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))
        

    def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=1024):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

     
    def features(self, x):
        """
        :param x: the input spectrogram, expected shape: (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        :return: the features before the final classification layer
        """
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)

        B = x.shape[0]
        x = self.v.patch_embed(x)
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)
        x = (x[:, 0] + x[:, 1]) / 2
        return x

     
    def forward(self, x):
        x = self.features(x)
        x = self.mlp_head(x)
        return x

class ASTModelCosineSim(nn.Module):
    """
    The AST model.
    :param label_dim: the label dimension, i.e., the number of total classes, it is 527 for AudioSet, 50 for ESC-50, and 35 for speechcommands v2-35
    :param fstride: the stride of patch spliting on the frequency dimension, for 16*16 patchs, fstride=16 means no overlap, fstride=10 means overlap of 6
    :param tstride: the stride of patch spliting on the time dimension, for 16*16 patchs, tstride=16 means no overlap, tstride=10 means overlap of 6
    :param input_fdim: the number of frequency bins of the input spectrogram
    :param input_tdim: the number of time frames of the input spectrogram
    :param imagenet_pretrain: if use ImageNet pretrained model
    :param audioset_pretrain: if use full AudioSet and ImageNet pretrained model
    :param model_size: the model size of AST, should be in [tiny224, small224, base224, base384], base224 and base 384 are same model, but are trained differently during ImageNet pretraining.
    """
    def __init__(self, label_dim=527, fstride=10, tstride=10, input_fdim=128, input_tdim=1024, model_size='base384'):

        super(ASTModelCosineSim, self).__init__()
        assert timm.__version__ == '0.4.5', 'Please use timm == 0.4.5, the code might not be compatible with newer versions.'

        # override timm input shape restriction
        timm.models.vision_transformer.PatchEmbed = PatchEmbed

        # if AudioSet pretraining is not used (but ImageNet pretraining may still apply)

        if model_size == 'tiny224':
            self.v = timm.create_model('vit_deit_tiny_distilled_patch16_224', pretrained=True)
        elif model_size == 'small224':
            self.v = timm.create_model('vit_deit_small_distilled_patch16_224', pretrained=True)
        elif model_size == 'base224':
            self.v = timm.create_model('vit_deit_base_distilled_patch16_224', pretrained=True)
        elif model_size == 'base384':
            self.v = timm.create_model('vit_deit_base_distilled_patch16_384', pretrained=True)
        else:
            raise Exception('Model size must be one of tiny224, small224, base224, base384.')
        self.original_num_patches = self.v.patch_embed.num_patches
        self.oringal_hw = int(self.original_num_patches ** 0.5)
        self.original_embedding_dim = self.v.pos_embed.shape[2]
        # self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, label_dim))
        self.layer_norm_mlp = nn.LayerNorm(self.original_embedding_dim)
        self.mlp_head = CosineSimLayer(self.original_embedding_dim, label_dim)

        # automatcially get the intermediate shape
        f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
        num_patches = f_dim * t_dim
        self.v.patch_embed.num_patches = num_patches


        # the linear projection layer
        # the linear projection layer
        new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        new_proj.weight = torch.nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
        new_proj.bias = self.v.patch_embed.proj.bias
        self.v.patch_embed.proj = new_proj
        
        new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, self.original_num_patches, self.original_embedding_dim).transpose(1, 2).reshape(1, self.original_embedding_dim, self.oringal_hw, self.oringal_hw)
        # cut (from middle) or interpolate the second dimension of the positional embedding
        if t_dim <= self.oringal_hw:
            new_pos_embed = new_pos_embed[:, :, :, int(self.oringal_hw / 2) - int(t_dim / 2): int(self.oringal_hw / 2) - int(t_dim / 2) + t_dim]
        else:
            new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(self.oringal_hw, t_dim), align_corners=False)
        # cut (from middle) or interpolate the first dimension of the positional embedding
        if f_dim <= self.oringal_hw:
            new_pos_embed = new_pos_embed[:, :, int(self.oringal_hw / 2) - int(f_dim / 2): int(self.oringal_hw / 2) - int(f_dim / 2) + f_dim, :]
        else:
            new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), align_corners=False)
        # flatten the positional embedding
        new_pos_embed = new_pos_embed.reshape(1, self.original_embedding_dim, num_patches).transpose(1,2)
        # concatenate the above positional embedding with the cls token and distillation token of the deit model.
        self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))
        

    def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=1024):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

     
    def features(self, x):
        """
        :param x: the input spectrogram, expected shape: (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        :return: the features before the final classification layer
        """
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)

        B = x.shape[0]
        x = self.v.patch_embed(x)
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)
        x = (x[:, 0] + x[:, 1]) / 2
        x = self.layer_norm_mlp(x)
        return x

     
    def forward(self, x, memset, return_weights=False):
        x = self.features(x)
        memset = self.features(memset)
        output = self.mlp_head(x, memset, return_weights=return_weights)
        return output

class ASTModelAttentionSparse(nn.Module):
    """
    The AST model.
    :param label_dim: the label dimension, i.e., the number of total classes, it is 527 for AudioSet, 50 for ESC-50, and 35 for speechcommands v2-35
    :param fstride: the stride of patch spliting on the frequency dimension, for 16*16 patchs, fstride=16 means no overlap, fstride=10 means overlap of 6
    :param tstride: the stride of patch spliting on the time dimension, for 16*16 patchs, tstride=16 means no overlap, tstride=10 means overlap of 6
    :param input_fdim: the number of frequency bins of the input spectrogram
    :param input_tdim: the number of time frames of the input spectrogram
    :param imagenet_pretrain: if use ImageNet pretrained model
    :param audioset_pretrain: if use full AudioSet and ImageNet pretrained model
    :param model_size: the model size of AST, should be in [tiny224, small224, base224, base384], base224 and base 384 are same model, but are trained differently during ImageNet pretraining.
    """
    def __init__(self, label_dim=527, fstride=10, tstride=10, input_fdim=128, input_tdim=1024, model_size='base384', heads=4, sparse=True):

        super(ASTModelAttentionSparse, self).__init__()
        assert timm.__version__ == '0.4.5', 'Please use timm == 0.4.5, the code might not be compatible with newer versions.'

        # override timm input shape restriction
        timm.models.vision_transformer.PatchEmbed = PatchEmbed

        # if AudioSet pretraining is not used (but ImageNet pretraining may still apply)

        if model_size == 'tiny224':
            self.v = timm.create_model('vit_deit_tiny_distilled_patch16_224', pretrained=True)
        elif model_size == 'small224':
            self.v = timm.create_model('vit_deit_small_distilled_patch16_224', pretrained=True)
        elif model_size == 'base224':
            self.v = timm.create_model('vit_deit_base_distilled_patch16_224', pretrained=True)
        elif model_size == 'base384':
            self.v = timm.create_model('vit_deit_base_distilled_patch16_384', pretrained=True)
        else:
            raise Exception('Model size must be one of tiny224, small224, base224, base384.')
        self.original_num_patches = self.v.patch_embed.num_patches
        self.oringal_hw = int(self.original_num_patches ** 0.5)
        self.original_embedding_dim = self.v.pos_embed.shape[2]
        # self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, label_dim))
        self.layer_norm_mlp = nn.LayerNorm(self.original_embedding_dim)
        self.mlp_head = AttentionSparseMax(self.original_embedding_dim, label_dim, heads=heads, sparse=sparse)

        # automatcially get the intermediate shape
        f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
        num_patches = f_dim * t_dim
        self.v.patch_embed.num_patches = num_patches


        # the linear projection layer
        # the linear projection layer
        new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        new_proj.weight = torch.nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
        new_proj.bias = self.v.patch_embed.proj.bias
        self.v.patch_embed.proj = new_proj
        
        new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, self.original_num_patches, self.original_embedding_dim).transpose(1, 2).reshape(1, self.original_embedding_dim, self.oringal_hw, self.oringal_hw)
        # cut (from middle) or interpolate the second dimension of the positional embedding
        if t_dim <= self.oringal_hw:
            new_pos_embed = new_pos_embed[:, :, :, int(self.oringal_hw / 2) - int(t_dim / 2): int(self.oringal_hw / 2) - int(t_dim / 2) + t_dim]
        else:
            new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(self.oringal_hw, t_dim), align_corners=False)
        # cut (from middle) or interpolate the first dimension of the positional embedding
        if f_dim <= self.oringal_hw:
            new_pos_embed = new_pos_embed[:, :, int(self.oringal_hw / 2) - int(f_dim / 2): int(self.oringal_hw / 2) - int(f_dim / 2) + f_dim, :]
        else:
            new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), align_corners=False)
        # flatten the positional embedding
        new_pos_embed = new_pos_embed.reshape(1, self.original_embedding_dim, num_patches).transpose(1,2)
        # concatenate the above positional embedding with the cls token and distillation token of the deit model.
        self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))
        

    def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=1024):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

     
    def features(self, x):
        """
        :param x: the input spectrogram, expected shape: (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        :return: the features before the final classification layer
        """
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)

        B = x.shape[0]
        x = self.v.patch_embed(x)
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)
        x = (x[:, 0] + x[:, 1]) / 2
        x = self.layer_norm_mlp(x)
        return x

     
    def forward(self, x, memset, return_weights=False):
        x = self.features(x)
        memset = self.features(memset)
        
        output = self.mlp_head(x, memset, return_weights=return_weights)
        return output

def astmodel(label_dim, fstride=10, tstride=10, input_fdim=128, input_tdim=1024, model_size='base384'):
    return ASTModel(label_dim=label_dim, fstride=fstride, tstride=tstride, input_fdim=input_fdim, input_tdim=input_tdim, model_size=model_size)

def astmodel_cosinesim(label_dim, fstride=10, tstride=10, input_fdim=128, input_tdim=1024, model_size='base384'):
    return ASTModelCosineSim(label_dim=label_dim, fstride=fstride, tstride=tstride, input_fdim=input_fdim, input_tdim=input_tdim, model_size=model_size)

def astmodel_attentionsparse(label_dim, fstride=10, tstride=10, input_fdim=128, input_tdim=1024, model_size='base384', heads=4):
    return ASTModelAttentionSparse(label_dim=label_dim, fstride=fstride, tstride=tstride, input_fdim=input_fdim, input_tdim=input_tdim, model_size=model_size, heads=heads, sparse=True)

def astmodel_attention(label_dim, fstride=10, tstride=10, input_fdim=128, input_tdim=1024, model_size='base384', heads=4):
    return ASTModelAttentionSparse(label_dim=label_dim, fstride=fstride, tstride=tstride, input_fdim=input_fdim, input_tdim=input_tdim, model_size=model_size, heads=heads, sparse=False)

modelDic = {
    'astmodel': astmodel,
    'astmodel_cosinesim': astmodel_cosinesim,
    'astmodel_attentionsparse': astmodel_attentionsparse,
    'astmodel_attention': astmodel_attention
}