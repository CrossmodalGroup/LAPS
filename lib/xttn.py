
import torch
import torch.nn.init
import torch.nn.functional as F


def get_padding_mask(features, lengths):

    with torch.no_grad():
        max_len = features.shape[1]

        mask = torch.arange(max_len).expand(features.shape[0], max_len).to(features.device)

        # (B, L)
        mask = (mask < lengths.long().unsqueeze(1))

    return mask


def l2norm(X, dim, eps=1e-8):

    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def func_attention(query, context, smooth=4, eps=1e-8, detach=True):

    queryT = torch.transpose(query, 1, 2)

    # (batch, sourceL, d) (batch, d, queryL) --> (batch, sourceL, queryL)
    attn = torch.bmm(context, queryT)

    attn = F.leaky_relu(attn, negative_slope=0.1)
    
    attn = l2norm(attn, 2)

    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()

    # --> (batch, queryL, sourceL)
    attn = F.softmax(attn*smooth, dim=2)

    # --> (batch, sourceL, queryL)
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)
   
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(contextT, attnT)
    
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)

    return weightedContext, attnT


# Returns cosine similarity between x1 and x2, computed along dim
def cosine_similarity(x1, x2, dim=1, eps=1e-8):

    x1 = F.normalize(x1, p=2, dim=dim)
    x2 = F.normalize(x2, p=2, dim=dim)

    w12 = torch.sum(x1 * x2, dim)

    return w12


# SCAN-t2i
def xattn_score_t2i(images, captions, cap_lens, smooth=9.0):

    similarities = []
    n_image = images.size(0)
    n_caption = captions.size(0)

    images = F.normalize(images, dim=-1)
    captions = F.normalize(captions, dim=-1)

    for i in range(n_caption):
        
        # # --> (n_image, n_word, d)
        n_word = cap_lens[i]
        cap_i_expand = captions[i, :n_word, :].unsqueeze(0).repeat(n_image, 1, 1)

        weiContext, _ = func_attention(cap_i_expand, images, smooth=smooth, )

        # (n_image, n_words)
        row_sim = cosine_similarity(cap_i_expand, weiContext, dim=2)
        row_sim = row_sim.mean(dim=1, keepdim=True)
    
        similarities.append(row_sim)

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 1)
    
    return similarities


# SCAN-i2t
def xattn_score_i2t(images, captions, cap_lens, smooth=4):

    similarities = []
    n_image = images.size(0)
    n_caption = captions.size(0)

    images = F.normalize(images, dim=-1)
    captions = F.normalize(captions, dim=-1)   

    for i in range(n_caption):
        
        # # --> (n_image, n_word, d)
        n_word = cap_lens[i]
        cap_i_expand = captions[i, :n_word, :].unsqueeze(0).repeat(n_image, 1, 1)

        weiContext, _ = func_attention(images, cap_i_expand, smooth=smooth, )

        # (n_image, n_region)
        row_sim = cosine_similarity(images, weiContext, dim=2)
        row_sim = row_sim.mean(dim=1, keepdim=True)

        similarities.append(row_sim)

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 1)

    return similarities


# SCAN bi-directional
def xattn_score_two(images, captions, cap_lens, smooth_t2i=9, smooth_i2t=4):

    similarities = []
    n_image = images.size(0)
    n_caption = captions.size(0)

    images = F.normalize(images, dim=-1)
    captions = F.normalize(captions, dim=-1)

    for i in range(n_caption):
        
        # # --> (n_image, n_word, d)
        n_word = cap_lens[i]
        cap_i_expand = captions[i, :n_word, :].unsqueeze(0).repeat(n_image, 1, 1)
        
        # t2i
        weiContext_t2i, _ = func_attention(cap_i_expand, images, smooth=smooth_t2i, )
        row_sim_t2i = cosine_similarity(cap_i_expand, weiContext_t2i, dim=2).mean(dim=1, keepdim=True)

        # i2t
        weiContext_i2t, _ = func_attention(images, cap_i_expand, smooth=smooth_i2t, )
        row_sim_i2t = cosine_similarity(images, weiContext_i2t, dim=2).mean(dim=1, keepdim=True)

        sims = (row_sim_t2i + row_sim_i2t) * 0.5
        similarities.append(sims)

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 1)
    
    return similarities


def matching_max_mean(img_regions, cap_words, cap_len, i2t=False, scan=False, bi_norm=False):
    
    similarities = []

    img_regions = F.normalize(img_regions, dim=-1)
    cap_words = F.normalize(cap_words, dim=-1)

    if len(img_regions.shape) == 4:
        n_image = img_regions.size(1)
        img_regions_context = img_regions
    else:
        n_image = img_regions.size(0)
        img_regions_context = None

    n_caption = cap_words.size(0)

    # Each text is operated separately
    for i in range(n_caption):
        
        if img_regions_context:
            img_regions = img_regions_context[i]    

        n_word = cap_len[i]
        # (n_images, cap_len, C)
        cap_i_expand = cap_words[i, :n_word, :].unsqueeze(0).repeat(n_image, 1, 1)

        # (n_images, cap_len, img_len)
        cap2img_sim = torch.bmm(cap_i_expand, img_regions.transpose(1, 2))

        if scan:
            cap2img_sim = F.leaky_relu(cap2img_sim, negative_slope=0.1)

        cap2img_sim_norm = F.normalize(cap2img_sim, dim=1) if bi_norm else cap2img_sim

        # t2i
        # (n_images, cap_len)
        row_sim = cap2img_sim_norm.max(dim=2)[0]    
        # (n_images, 1)
        row_sim_mean = row_sim.mean(dim=1, keepdim=True)

        if i2t:
            cap2img_sim_norm = F.normalize(cap2img_sim, dim=2) if bi_norm else cap2img_sim

            # (n_images, img_len)
            column_sim = cap2img_sim_norm.max(dim=1)[0]
            # (n_images, 1)
            column_sim_mean = column_sim.mean(dim=1, keepdim=True)
            
            similarities.append((row_sim_mean + column_sim_mean) * 0.5)
        else:
            similarities.append(row_sim_mean)
    
    # (n_image, n_caption)
    similarities = torch.cat(similarities, 1)

    return similarities


# Only for one text
# The required feature has been L2 regularized
# img_mask (B_v, L_v)
def mask_xattn_one_text(img_embs, cap_i_expand, img_mask=None, i2t=True, scan=True,):

    # (B_v, L_t, L_v)
    cap2img_sim = torch.bmm(cap_i_expand, img_embs.transpose(1, 2))

    if scan:
        cap2img_sim = F.leaky_relu(cap2img_sim, negative_slope=0.1)

    # t2i
    # (B_v, L_t)
    if img_mask is None:
        row_sim = cap2img_sim.max(dim=2)[0]
    else:
        # Add a low value to the similarity of the masked patch location 
        # to prevent it from being selected
        row_sim = (cap2img_sim - 1000 * (1 - img_mask).unsqueeze(1)).max(dim=2)[0]  
    
    # (B_v, 1)
    row_sim_mean = row_sim.mean(dim=1, keepdim=True)

    if i2t:
        # i2t
        # (B_v, L_v)
        column_sim = cap2img_sim.max(dim=1)[0]
        
        if img_mask is None:
            column_sim_mean = column_sim.mean(dim=1, keepdim=True)
        else:
            # (B_v, 1)
            column_sim_mean = (column_sim * img_mask).sum(dim=-1, keepdim=True) / (img_mask.sum(dim=-1, keepdim=True) + 1e-8)

        sim_one_text = row_sim_mean + column_sim_mean
    else:
        sim_one_text = row_sim_mean
    
    return sim_one_text 


# different alignment functions
def xattn_score(img_cross, cap_cross, cap_len, xattn_type='max_mean', i2t=True, scan=True):

    smooth_t2i = 9
    smooth_i2t = 4

    if xattn_type == 'scan_t2i':
        sim = xattn_score_t2i(img_cross, cap_cross, cap_len, smooth=smooth_t2i)
    elif xattn_type == 'scan_i2t':
        sim = xattn_score_i2t(img_cross, cap_cross, cap_len, smooth=smooth_i2t)
    elif xattn_type == 'scan_all':
        sim = xattn_score_two(img_cross, cap_cross, cap_len, smooth_t2i, smooth_i2t)
    else:
        sim = matching_max_mean(img_cross, cap_cross, cap_len, i2t=i2t, scan=scan)

    return sim


if __name__ == '__main__':

    pass





