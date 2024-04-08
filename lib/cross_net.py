import torch
import torch.nn.functional as F
import math
import torch.nn as nn

from lib.xttn import mask_xattn_one_text


def is_sqr(n):
    a = int(math.sqrt(n))
    return a * a == n


class TokenSparse(nn.Module):
    def __init__(self, embed_dim=512, sparse_ratio=0.6):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.sparse_ratio = sparse_ratio
    
    def forward(self, tokens, attention_x, attention_y):
        
        B_v, L_v, C = tokens.size()

        # (B_v, L_v)
        score = attention_x + attention_y

        num_keep_token = math.ceil(L_v * self.sparse_ratio)
    
        # select the top-k index, (B_v, L_v)
        score_sort, score_index = torch.sort(score, dim=1, descending=True)
        
        # (B_v, L_v * token_ratio)
        keep_policy = score_index[:, :num_keep_token]

        # (B_v, L_v)
        score_mask = torch.zeros_like(score).scatter(1, keep_policy, 1)
        
        # (B_v, L_v * token_ratio, C)
        select_tokens = torch.gather(tokens, dim=1, index=keep_policy.unsqueeze(-1).expand(-1, -1, C))

        # fusion token
        # (B_v, L_v *  (1 - token_ratio) )
        non_keep_policy = score_index[:, num_keep_token:]

        # (B_v, L_v *  (1 - token_ratio), C )
        non_tokens = torch.gather(tokens, dim=1, index=non_keep_policy.unsqueeze(-1).expand(-1, -1, C))
        
        # (B_v, L_v *  (1 - token_ratio) )
        non_keep_score = score_sort[:, num_keep_token:]
        # through softmax function, (B_v, L_v *  (1 - token_ratio) ) -> (B_v, L_v *  (1 - token_ratio), 1)
        non_keep_score = F.softmax(non_keep_score, dim=1).unsqueeze(-1)

        # get fusion token (B_v, 1, C)
        extra_token = torch.sum(non_tokens * non_keep_score, dim=1, keepdim=True) 

        return select_tokens, extra_token, score_mask
                  

# dim_ratio affect GPU memory
class TokenAggregation(nn.Module):
    def __init__(self, dim=512, keeped_patches=64, dim_ratio=0.2):
        super().__init__()
        
        hidden_dim = int(dim * dim_ratio)

        self.weight = nn.Sequential(
                        nn.LayerNorm(dim),
                        nn.Linear(dim, hidden_dim),
                        nn.GELU(),
                        nn.Linear(hidden_dim, keeped_patches)
                        )
        
        self.scale = nn.Parameter(torch.ones(1, 1, 1))
        
    def forward(self, x, keep_policy=None):

        # (B, N, C) -> (B, N, N_s)
        weight = self.weight(x)

        #  (B, N, N_s) -> (B, N_s, N)
        weight = weight.transpose(2, 1) * self.scale       

        if keep_policy is not None:
            # keep_policy (B, N) -> (B, 1, N)
            keep_policy = keep_policy.unsqueeze(1)
            # increase a large number for mask patches
            weight = weight - (1 - keep_policy) * 1e10

        # learning a set of weight matrices
        weight = F.softmax(weight, dim=2)
        
        # (B, N_s, C)
        # multiply with patch features
        x = torch.bmm(weight, x)
        
        return x
    

## sparse + aggregation
class CrossSparseAggrNet_v2(nn.Module):
    def __init__(self, opt=None):
        super().__init__()

        self.opt = opt
        
        self.hidden_dim = opt.embed_size  
        self.num_patches = opt.num_patches

        self.sparse_ratio = opt.sparse_ratio 
        self.aggr_ratio = opt.aggr_ratio 

        self.attention_weight = opt.attention_weight
        self.ratio_weight = opt.ratio_weight
        
        # the number of aggregated patches
        self.keeped_patches = int(self.num_patches * self.aggr_ratio * self.sparse_ratio)

        # sparse network
        self.sparse_net = TokenSparse(embed_dim=self.hidden_dim, 
                                      sparse_ratio=self.sparse_ratio,
                                      )
        # aggregation network
        self.aggr_net= TokenAggregation(dim=self.hidden_dim, 
                                        keeped_patches=self.keeped_patches,
                                        )  

    def forward(self, img_embs, cap_embs, cap_lens):

        B_v, L_v, C = img_embs.shape
    
        # feature normalization
        # (B_v, L_v, C)
        img_embs_norm = F.normalize(img_embs, dim=-1)
        # (B_t, L_t, C)
        cap_embs_norm = F.normalize(cap_embs, dim=-1)

        self.has_cls_token = False if is_sqr(img_embs.shape[1]) else True

        #  whether it exists [cls] token
        if self.has_cls_token:
            # (B_v, 1, C)
            img_cls_emb = img_embs[:, 0:1, :]
            img_cls_emb_norm = img_embs_norm[:, 0:1, :]
            img_spatial_embs = img_embs[:, 1:, :]
            img_spatial_embs_norm = img_embs_norm[:, 1:, :]
        else:
            img_spatial_embs = img_embs
            img_spatial_embs_norm = img_embs_norm

        # compute self-attention 
        with torch.no_grad():
            # (B_v, L_v, C) ->  (B_v, 1, C)
            img_spatial_glo_norm = F.normalize(img_spatial_embs.mean(dim=1, keepdim=True), dim=-1)
            # (B_v, L_v, C) -> (B_v, L_v)
            img_spatial_self_attention = (img_spatial_glo_norm * img_spatial_embs_norm).sum(dim=-1)

        improve_sims = []
        score_mask_all = []

        # Introduce text information
        # process each text separately
        for i in range(len(cap_lens)):

            n_word = cap_lens[i]                  
            # (L_t, C)
            cap_i = cap_embs[i, :n_word, :]
    
            # (B_v, L_t, C)
            cap_i_expand = cap_embs_norm[i, :n_word, :].unsqueeze(0).repeat(B_v, 1, 1)

            ## compute cross-attention
            with torch.no_grad():               
                # (L_t, C) -> (1, C) -> (1, 1, C)
                cap_i_glo = F.normalize(cap_i.mean(0, keepdim=True).unsqueeze(0), dim=-1)
                # (B_v, L_v, C) -> (B_v, L_v)
                img_spatial_cap_i_attention = (cap_i_glo * img_spatial_embs_norm).sum(dim=-1)

            # selection
            select_tokens, extra_token, score_mask = self.sparse_net(tokens=img_spatial_embs, 
                                                                     attention_x=img_spatial_self_attention, 
                                                                    attention_y=img_spatial_cap_i_attention,
                                                                    )

            # aggregation
            aggr_tokens = self.aggr_net(select_tokens)
            # aggr_tokens = select_tokens

            # add fusion token
            keep_spatial_tokens = torch.cat([aggr_tokens, extra_token], dim=1)

            # add [cls] token
            if self.has_cls_token:
                select_tokens = torch.cat((img_cls_emb, keep_spatial_tokens), dim=1)
            else:
                select_tokens = keep_spatial_tokens

            # patch normalization
            select_tokens = F.normalize(select_tokens, dim=-1)

            # image-text similarity
            # (B_v, 1)
            sim_one_text = mask_xattn_one_text(img_embs=select_tokens, 
                                               cap_i_expand=cap_i_expand, 
                                               )
            
            improve_sims.append(sim_one_text)
            score_mask_all.append(score_mask)

        # (B_v, B_t)
        improve_sims = torch.cat(improve_sims, dim=1)
        score_mask_all = torch.stack(score_mask_all, dim=0)

        if self.training:
            return improve_sims, score_mask_all
        else:
            return improve_sims


if __name__ == '__main__':

    pass