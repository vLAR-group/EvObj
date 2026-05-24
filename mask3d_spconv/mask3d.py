import torch
import torch.nn as nn
import spconv.pytorch as spconv
from spconv.pytorch import SparseAvgPool3d, SparseMaxPool3d
import numpy as np
import os
from torch.nn import functional as F
from mask3d_spconv.position_embedding import PositionEmbeddingCoordsSine
from pointnet2.pointnet2 import furthest_point_sample
from mask3d_spconv.modules.helpers_3detr import GenericMLP
from torch_scatter import scatter_mean, scatter_max, scatter_min
from torch.cuda.amp import autocast

SPCONV_ALGO = 'auto'    # 'auto', 'implicit_gemm', 'native'
env_spconv_algo = os.environ.get('SPCONV_ALGO')
if env_spconv_algo is not None and env_spconv_algo in ['auto', 'implicit_gemm', 'native']:
    SPCONV_ALGO = env_spconv_algo

algo = None
if SPCONV_ALGO == 'native':
    algo = spconv.ConvAlgo.Native
elif SPCONV_ALGO == 'implicit_gemm':
    algo = spconv.ConvAlgo.MaskImplicitGemm


def sorting_spconv(grids, spatial_shape):
    bs_sort = grids[:, 0] * (spatial_shape[0] + 1) + grids[:, 1] * (spatial_shape[1] + 1) + grids[:, 2] * (spatial_shape[2] + 1)
    return torch.sort(bs_sort).indices


class Mask3D(nn.Module):
    def __init__(self,
        backbone,
        hidden_dim,
        num_queries,
        num_heads,
        dim_feedforward,
        sample_sizes,
        shared_decoder,
        num_classes,
        num_decoders,
        dropout,
        pre_norm,
        positional_encoding_type,
        non_parametric_queries,
        # train_on_segments,
        normalize_pos_enc,
        use_level_embed,
        scatter_type,
        hlevels,
        use_np_features,
        voxel_size,
        max_sample_size,
        random_queries,
        gauss_scale,
        random_query_both,
        random_normal):
        super().__init__()

        self.random_normal = random_normal
        self.random_query_both = random_query_both
        self.random_queries = random_queries
        self.max_sample_size = max_sample_size
        self.gauss_scale = gauss_scale
        self.voxel_size = voxel_size
        self.scatter_type = scatter_type
        self.hlevels = hlevels
        self.use_level_embed = use_level_embed
        # self.train_on_segments = train_on_segments
        self.normalize_pos_enc = normalize_pos_enc
        self.num_decoders = num_decoders
        self.num_classes = num_classes
        self.dropout = dropout
        self.pre_norm = pre_norm
        self.shared_decoder = shared_decoder
        self.sample_sizes = sample_sizes
        self.non_parametric_queries = non_parametric_queries
        self.use_np_features = use_np_features
        self.mask_dim = hidden_dim
        self.num_heads = num_heads
        self.num_queries = num_queries
        self.pos_enc_type = positional_encoding_type

        self.backbone = backbone
        self.num_levels = len(self.hlevels)
        sizes = self.backbone.channels[-4:]

        self.mask_features_head = spconv.SubMConv3d(self.backbone.channels[7], self.mask_dim, kernel_size=1, stride=1, bias=True, algo=algo)

        if self.scatter_type == "mean":
            self.scatter_fn = scatter_mean
        elif self.scatter_type == "max":
            self.scatter_fn = lambda mask, p2s, dim: scatter_max(mask, p2s, dim=dim)[0]
        else:
            assert False, "Scatter function not known"

        assert (not use_np_features) or non_parametric_queries, "np features only with np queries"

        if self.non_parametric_queries:
            self.query_projection = GenericMLP(
                input_dim=self.mask_dim,
                hidden_dims=[self.mask_dim],
                output_dim=self.mask_dim,
                use_conv=True,
                output_use_activation=True,
                hidden_use_bias=True)

            if self.use_np_features:
                self.np_feature_projection = nn.Sequential(
                    nn.Linear(sizes[-1], hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim))
                self.query_feat = nn.Embedding(num_queries, hidden_dim)

        elif self.random_query_both:
            self.query_projection = GenericMLP(
                input_dim=2 * self.mask_dim,
                hidden_dims=[2 * self.mask_dim],
                output_dim=2 * self.mask_dim,
                use_conv=True,
                output_use_activation=True,
                hidden_use_bias=True)
        else:
            # PARAMETRIC QUERIES
            # learnable query features
            self.query_feat = nn.Embedding(num_queries, hidden_dim)
            # learnable query p.e.
            self.query_pos = nn.Embedding(num_queries, hidden_dim)

        if self.use_level_embed:
            # learnable scale-level embedding
            self.level_embed = nn.Embedding(self.num_levels, hidden_dim)

        self.mask_embed_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim))

        self.class_embed_head = nn.Linear(hidden_dim, self.num_classes)


        if self.pos_enc_type == "legacy":
            self.pos_enc = PositionalEncoding3D(channels=self.mask_dim)
        elif self.pos_enc_type == "fourier":
            self.pos_enc = PositionEmbeddingCoordsSine(
                pos_type="fourier",
                d_pos=self.mask_dim,
                gauss_scale=self.gauss_scale,
                normalize=self.normalize_pos_enc)
        elif self.pos_enc_type == "sine":
            self.pos_enc = PositionEmbeddingCoordsSine(
                pos_type="sine",
                d_pos=self.mask_dim,
                normalize=self.normalize_pos_enc)
        else:
            assert False, "pos enc type not known"

        self.pooling = SparseAvgPool3d(kernel_size=2, stride=2, algo=algo)

        self.masked_transformer_decoder = nn.ModuleList()
        self.cross_attention = nn.ModuleList()
        self.self_attention = nn.ModuleList()
        self.ffn_attention = nn.ModuleList()
        self.lin_squeeze = nn.ModuleList()

        num_shared = self.num_decoders if not self.shared_decoder else 1

        for _ in range(num_shared):
            tmp_cross_attention = nn.ModuleList()
            tmp_self_attention = nn.ModuleList()
            tmp_ffn_attention = nn.ModuleList()
            tmp_squeeze_attention = nn.ModuleList()
            for i, hlevel in enumerate(self.hlevels):
                tmp_cross_attention.append(
                    CrossAttentionLayer(
                        d_model=self.mask_dim,
                        nhead=self.num_heads,
                        dropout=self.dropout,
                        normalize_before=self.pre_norm))

                tmp_squeeze_attention.append(nn.Linear(sizes[hlevel], self.mask_dim))

                tmp_self_attention.append(SelfAttentionLayer(d_model=self.mask_dim,
                        nhead=self.num_heads,
                        dropout=self.dropout,
                        normalize_before=self.pre_norm))

                tmp_ffn_attention.append(FFNLayer(d_model=self.mask_dim,
                        dim_feedforward=dim_feedforward,
                        dropout=self.dropout,
                        normalize_before=self.pre_norm))

            self.cross_attention.append(tmp_cross_attention)
            self.self_attention.append(tmp_self_attention)
            self.ffn_attention.append(tmp_ffn_attention)
            self.lin_squeeze.append(tmp_squeeze_attention)

        self.decoder_norm = nn.LayerNorm(hidden_dim)

    def get_pos_encs(self, coords):
        pos_encodings_pcd = []
        for i in range(len(coords)):
            pos_encodings_pcd.append([]) ## why so many list?
            # for coords_batch in coords[i].features:
            for curr_coords in coords[i]:
                scene_min = curr_coords.min(dim=0)[0][None, ...]
                scene_max = curr_coords.max(dim=0)[0][None, ...]
                with autocast(enabled=False):
                    tmp = self.pos_enc(curr_coords[None, ...].float(), input_range=[scene_min, scene_max])
                pos_encodings_pcd[-1].append(tmp.squeeze(0).permute((1, 0)))
        return pos_encodings_pcd


    def forward(self, x, point2segment=None, raw_coordinates=None, train_on_segments=True, is_eval=False, anchor=None, is_datacollect=False, env_num=None):
        pcd_features, aux = self.backbone(x)
        batch_size = aux[-1].batch_size

        with torch.no_grad():
            coordinates = spconv.SparseConvTensor(features=raw_coordinates, indices=pcd_features.indices,
                                    spatial_shape=pcd_features.spatial_shape, batch_size=pcd_features.batch_size)
            coords = [coordinates]
            for _ in reversed(range(len(aux) - 1)):
                coords.append(self.pooling(coords[-1]))
            coords.reverse()

        decomposed_aux_grids, decomposed_aux_feats, decomposed_aux_coords = [], [], []
        for level in range(len(aux)):  ## from level 0 to 3, sparse to dense
            decomposed_aux_grids.append([]), decomposed_aux_feats.append([]), decomposed_aux_coords.append([])
            for bs in range(batch_size):
                bs_mask = torch.where(aux[level].indices[:, 0] == bs)[0]
                bs_grids, bs_feats = aux[level].indices[:, 1:][bs_mask], aux[level].features[bs_mask]
                bs_sort = sorting_spconv(bs_grids, aux[level].spatial_shape)
                decomposed_aux_grids[-1].append(bs_grids[bs_sort])  ## maybe useless
                decomposed_aux_feats[-1].append(bs_feats[bs_sort])

                other_bs_mask = torch.where(coords[level].indices[:, 0] == bs)[0]
                bs_coords = coords[level].features[other_bs_mask]
                bs_sort2 = sorting_spconv(coords[level].indices[:, 1:][other_bs_mask], coords[level].spatial_shape)
                decomposed_aux_coords[-1].append(bs_coords[bs_sort2])

        pos_encodings_pcd = self.get_pos_encs(decomposed_aux_coords)
        mask_features = self.mask_features_head(pcd_features)

        counts = torch.bincount(mask_features.indices[:, 0], minlength=batch_size) ## this indices is shared with coordinates
        decomposed_mask_feats = list(torch.split(mask_features.features, counts.tolist()))
        decomposed_mask_grids = list(torch.split(mask_features.indices[:, 1:], counts.tolist()))
        decomposed_mask_coords = list(torch.split(coordinates.features, counts.tolist()))

        if train_on_segments:
            mask_segments = []
            for i, mask_feature in enumerate(decomposed_mask_feats):
                if (point2segment[i]==-1).sum()>0:
                    tmp_idx = point2segment[i]!=-1
                    mask_segments.append(self.scatter_fn(mask_feature[tmp_idx], point2segment[i][tmp_idx], dim=0))
                else:
                    mask_segments.append(self.scatter_fn(mask_feature, point2segment[i], dim=0))

        sampled_coords = None

        if self.non_parametric_queries:
            fps_idx = [furthest_point_sample(decomposed_mask_grids[i][None, ...].float(), self.num_queries).squeeze(0).long()
                for i in range(len(decomposed_mask_grids))]

            if anchor is None:
                sampled_coords = torch.stack([decomposed_mask_coords[i][fps_idx[i].long(), :] for i in range(len(fps_idx))])
            else:
                sampled_coords = anchor

            mins = torch.stack([(decomposed_mask_coords[i]).min(dim=0)[0] for i in range(len(decomposed_mask_coords))])
            maxs = torch.stack([(decomposed_mask_coords[i]).max(dim=0)[0] for i in range(len(decomposed_mask_coords))])

            query_pos = self.pos_enc(sampled_coords.float(), input_range=[mins, maxs])  # Batch, Dim, queries
            ###
            query_pos = self.query_projection(query_pos)

            if not self.use_np_features:
                queries = torch.zeros_like(query_pos).permute((0, 2, 1))
            else:
                queries = self.query_feat.weight.unsqueeze(0).repeat(batch_size, 1, 1)
            query_pos = query_pos.permute((2, 0, 1))
        elif self.random_queries:
            query_pos = (torch.rand(batch_size, self.mask_dim, self.num_queries, device=x.device) - 0.5)

            queries = torch.zeros_like(query_pos).permute((0, 2, 1))
            query_pos = query_pos.permute((2, 0, 1))
        elif self.random_query_both:
            if not self.random_normal:
                query_pos_feat = (torch.rand(batch_size, 2 * self.mask_dim, self.num_queries, device=x.device) - 0.5)
            else:
                query_pos_feat = torch.randn(batch_size, 2 * self.mask_dim, self.num_queries, device=x.device)

            queries = query_pos_feat[:, : self.mask_dim, :].permute((0, 2, 1))
            query_pos = query_pos_feat[:, self.mask_dim :, :].permute((2, 0, 1))
        else:
            # PARAMETRIC QUERIES
            queries = self.query_feat.weight.unsqueeze(0).repeat(batch_size, 1, 1)
            query_pos = self.query_pos.weight.unsqueeze(1).repeat(1, batch_size, 1)
        #
        if is_datacollect:
            if env_num is not None:
                fps_idx = [furthest_point_sample(decomposed_mask_grids[i][None, ...].float(), env_num).squeeze(0).long()
                    for i in range(len(decomposed_mask_grids))]
                sampled_coords = torch.stack([decomposed_mask_coords[i][fps_idx[i].long(), :] for i in range(len(fps_idx))])
            return {"sampled_coords": sampled_coords.detach() if sampled_coords is not None else None, "mask_features": decomposed_mask_feats}
        else:
            predictions_class = []
            predictions_mask = []

            for decoder_counter in range(self.num_decoders):
                if self.shared_decoder:
                    decoder_counter = 0
                for i, hlevel in enumerate(self.hlevels):
                    '''new'''
                    if train_on_segments:
                        output_class, outputs_mask, attn_mask = self.mask_module(
                            queries,
                            mask_features,
                            mask_segments,
                            len(aux) - hlevel - 1,
                            ret_attn_mask=True,
                            point2segment=point2segment,
                            decomposed_mask_feats=decomposed_mask_feats)
                    else:
                        output_class, outputs_mask, attn_mask = self.mask_module(
                            queries,
                            mask_features,
                            None,
                            len(aux) - hlevel - 1,
                            ret_attn_mask=True,
                            point2segment=None,
                            decomposed_mask_feats=decomposed_mask_feats)
                    '''new end'''
                    decomposed_aux = decomposed_aux_feats[hlevel]
                    decomposed_attn = attn_mask

                    curr_sample_size = max([pcd.shape[0] for pcd in decomposed_aux])

                    if min([pcd.shape[0] for pcd in decomposed_aux]) == 1:
                        raise RuntimeError("only a single point gives nans in cross-attention")

                    if not (self.max_sample_size or is_eval):
                        curr_sample_size = min(curr_sample_size, self.sample_sizes[hlevel])

                    rand_idx = []
                    mask_idx = []
                    for k in range(len(decomposed_aux)):
                        pcd_size = decomposed_aux[k].shape[0]
                        if pcd_size <= curr_sample_size:
                            # we do not need to sample
                            # take all points and pad the rest with zeroes and mask it
                            idx = torch.zeros(curr_sample_size, dtype=torch.long, device=queries.device)

                            midx = torch.ones(curr_sample_size, dtype=torch.bool, device=queries.device)

                            idx[:pcd_size] = torch.arange(pcd_size, device=queries.device)

                            midx[:pcd_size] = False  # attend to first points
                        else:
                            # we have more points in pcd as we like to sample
                            # take a subset (no padding or masking needed)
                            idx = torch.randperm(decomposed_aux[k].shape[0], device=queries.device)[:curr_sample_size]
                            midx = torch.zeros(curr_sample_size, dtype=torch.bool, device=queries.device)  # attend to all

                        rand_idx.append(idx)
                        mask_idx.append(midx)

                    batched_aux = torch.stack([decomposed_aux[k][rand_idx[k], :] for k in range(len(rand_idx))])## batched_features for key and value
                    '''new'''
                    batched_attn = torch.stack([decomposed_attn[k][rand_idx[k], :] for k in range(len(rand_idx))])
                    '''new end'''

                    batched_pos_enc = torch.stack([pos_encodings_pcd[hlevel][k][rand_idx[k], :] for k in range(len(rand_idx))])## batched pos_enc for key and value
                    '''new'''
                    batched_attn.permute((0, 2, 1))[batched_attn.sum(1) == rand_idx[0].shape[0]] = False

                    m = torch.stack(mask_idx)
                    batched_attn = torch.logical_or(batched_attn, m[..., None])
                    '''new end'''
                    #
                    src_pcd = self.lin_squeeze[decoder_counter][i](batched_aux.permute((1, 0, 2)))
                    if self.use_level_embed:
                        src_pcd += self.level_embed.weight[i]

                    output = self.cross_attention[decoder_counter][i](
                        queries.permute((1, 0, 2)),
                        src_pcd, memory_mask=batched_attn.repeat_interleave(self.num_heads, dim=0).permute((0, 2, 1)),
                        memory_key_padding_mask=None,  # here we do not apply masking on padded region
                        pos=batched_pos_enc.permute((1, 0, 2)),
                        query_pos=query_pos)

                    output = self.self_attention[decoder_counter][i](output, tgt_mask=None, tgt_key_padding_mask=None, query_pos=query_pos)
                    # FFN
                    queries = self.ffn_attention[decoder_counter][i](output).permute((1, 0, 2))

                    predictions_class.append(output_class)
                    predictions_mask.append(outputs_mask)
                    '''new end'''

            if train_on_segments:
                output_class, outputs_mask = self.mask_module(queries, mask_features, mask_segments, 0, ret_attn_mask=False, point2segment=point2segment)
            else:
                output_class, outputs_mask = self.mask_module(queries, mask_features, None, 0, ret_attn_mask=False, point2segment=None)

            predictions_class.append(output_class)
            predictions_mask.append(outputs_mask)

            return {"pred_logits": predictions_class[-1],
                "pred_masks": predictions_mask[-1],
                "aux_outputs": self._set_aux_loss(predictions_class, predictions_mask),
                "sampled_coords": sampled_coords.detach() if sampled_coords is not None else None,
                "queries": queries,
                "mask_features": decomposed_mask_feats}


    def mask_module(self, query_feat, mask_features, mask_segments, num_pooling_steps, ret_attn_mask=True, point2segment=None, decomposed_mask_feats=None):

        query_feat = self.decoder_norm(query_feat)
        mask_embed = self.mask_embed_head(query_feat)
        outputs_class = self.class_embed_head(query_feat)
        output_masks = []

        if point2segment is not None:
            output_segments = []
            for i in range(len(mask_segments)):
                output_segments.append(mask_segments[i] @ mask_embed[i].T)
                output_masks.append(output_segments[-1][point2segment[i]])
        else:
            for i in range(len(decomposed_mask_feats)):
                output_masks.append(decomposed_mask_feats[i] @ mask_embed[i].T)

        output_masks = torch.cat(output_masks)

        outputs_mask = spconv.SparseConvTensor(features=output_masks, indices=mask_features.indices,
                                spatial_shape=mask_features.spatial_shape, batch_size=mask_features.batch_size)

        if ret_attn_mask:
            attn_mask = outputs_mask
            for _ in range(num_pooling_steps):
                attn_mask = self.pooling(attn_mask)

            attn_mask_list = []
            for bs in range(attn_mask.batch_size):
                bs_mask = torch.where(attn_mask.indices[:, 0] == bs)[0]
                bs_attn_mask = attn_mask.features[bs_mask]
                bs_sort = sorting_spconv(attn_mask.indices[:, 1:][bs_mask], attn_mask.spatial_shape)
                attn_mask_list.append(bs_attn_mask[bs_sort].detach().sigmoid() < 0.5)
            attn_mask = attn_mask_list

            if point2segment is not None:
                return outputs_class, output_segments, attn_mask
            else:
                counts = torch.bincount(mask_features.indices[:, 0], minlength=mask_features.batch_size)
                decomposed_outputs_mask = list(torch.split(outputs_mask.features, counts.tolist()))
                return (outputs_class, decomposed_outputs_mask, attn_mask)

        if point2segment is not None:
            return outputs_class, output_segments
        else:
            counts = torch.bincount(mask_features.indices[:, 0], minlength=mask_features.batch_size)
            decomposed_outputs_mask = list(torch.split(outputs_mask.features, counts.tolist()))
            return outputs_class, decomposed_outputs_mask

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"pred_logits": a, "pred_masks": b} for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])]


class PositionalEncoding3D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        self.orig_ch = channels
        super(PositionalEncoding3D, self).__init__()
        channels = int(np.ceil(channels / 6) * 2)
        if channels % 2:
            channels += 1
        self.channels = channels
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, channels, 2).float() / channels)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, tensor, input_range=None):
        """
        :param tensor: A 5d tensor of size (batch_size, x, y, z, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, z, ch)
        """
        pos_x, pos_y, pos_z = tensor[:, :, 0], tensor[:, :, 1], tensor[:, :, 2]
        sin_inp_x = torch.einsum("bi,j->bij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("bi,j->bij", pos_y, self.inv_freq)
        sin_inp_z = torch.einsum("bi,j->bij", pos_z, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)

        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1)
        emb_z = torch.cat((sin_inp_z.sin(), sin_inp_z.cos()), dim=-1)

        emb = torch.cat((emb_x, emb_y, emb_z), dim=-1)
        return emb[:, :, : self.orig_ch].permute((0, 2, 1))


class SelfAttentionLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.0,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self, tgt, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None
    ):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q,
            k,
            value=tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(
        self, tgt, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None
    ):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(
            q,
            k,
            value=tgt2,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(
        self, tgt, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt, tgt_mask, tgt_key_padding_mask, query_pos
            )
        return self.forward_post(
            tgt, tgt_mask, tgt_key_padding_mask, query_pos
        )


class CrossAttentionLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.0,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        memory,
        memory_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None,
    ):
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(
        self,
        tgt,
        memory,
        memory_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None,
    ):
        tgt2 = self.norm(tgt)

        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(
        self,
        tgt,
        memory,
        memory_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None,
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt,
                memory,
                memory_mask,
                memory_key_padding_mask,
                pos,
                query_pos,
            )
        return self.forward_post(
            tgt, memory, memory_mask, memory_key_padding_mask, pos, query_pos
        )


class FFNLayer(nn.Module):
    def __init__(
        self,
        d_model,
        dim_feedforward=2048,
        dropout=0.0,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")
