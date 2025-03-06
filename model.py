from dataclasses import dataclass
import torch
from torch import nn
import matplotlib.pyplot as plt
import torch.nn.functional as F



@dataclass
class TransformerConfig:
    src_vocab_size: int = 32000
    tgt_vocab_size: int = 32000
    max_seq_length: int = 64
    d_model: int = 512
    num_heads: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dropout_p: float = 0.1
    dff: int = 2048
    device: str = 'cpu' 



# Source Embedding block
class SourceEmbedding(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.src_embedding = nn.Embedding(num_embeddings=config.src_vocab_size, embedding_dim=config.d_model)
            
    def forward(self, x):
        x = self.src_embedding(x)
        return x


# Target Embedding block     
class TargetEmbedding(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.tgt_embedding = nn.Embedding(num_embeddings=config.tgt_vocab_size, embedding_dim=config.d_model)
    
    def forward(self, x):
        x = self.tgt_embedding(x)
        return x
    
# Position Encoding (PE)

class PositionEncoding(nn.Module):
    def __init__(self, config: TransformerConfig, require_grad=False):
        super().__init__()
        self.PE = torch.zeros(config.max_seq_length, config.d_model)
        pos = torch.arange(0, config.max_seq_length).reshape(-1, 1)
        i = torch.arange(0, config.d_model, step=2)
        
        denominator = torch.pow(10000, (2*i) / config.d_model)
        self.PE[:, 0::2] = torch.sin(pos/denominator)
        self.PE[:, 1::2] = torch.cos(pos/denominator)
        
    
        self.PE = nn.Parameter(self.PE, requires_grad=require_grad)
    
    def forward(self, x):
        max_seq_length = x.shape[1]
        return x + self.PE[:max_seq_length]
        


# Muti Head Attention block for (Multi Head Attention, Masked Multi Head Attention and Cross Multi Heads Attention)
class MultiheadAttention(nn.Module):
    def __init__(self, config:TransformerConfig):
        super().__init__()
        self.config = config
        
        # check if the d_model is divided by num_heads to get the head dim
        assert config.d_model % self.config.num_heads == 0, "The d_model is not divided by the num of heads"
        self.head_dim = self.config.d_model // self.config.num_heads
        
        
        self.q_proj = nn.Linear(in_features=self.config.d_model, out_features=self.config.d_model)
        self.k_proj = nn.Linear(in_features=self.config.d_model, out_features=self.config.d_model)
        self.v_proj = nn.Linear(in_features=self.config.d_model, out_features=self.config.d_model)
        
        self.out_proj = nn.Linear(in_features=self.config.d_model, out_features=self.config.d_model)
        
    
    def forward(self, src, tgt=None, attention_mask=None, causal=False):
        batch, src_seq_length, d_model = src.shape
        if tgt is None:
            q = self.q_proj(src).reshape(batch, src_seq_length, self.config.num_heads, self.head_dim).transpose(1,2).contiguous()
            k = self.k_proj(src).reshape(batch, src_seq_length, self.config.num_heads, self.head_dim).transpose(1,2).contiguous()
            v = self.v_proj(src).reshape(batch, src_seq_length, self.config.num_heads, self.head_dim).transpose(1,2).contiguous()
            
            #MASKED MULTI HEAD ATTENTION
            if attention_mask is not None:
                attention_mask = attention_mask.bool()
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1).repeat(1,1,src_seq_length,1).to(self.config.device)
            
            if causal and attention_mask is not None:
                # compute new mask (pad mask + causal mask)
                causal_mask = ~torch.triu(torch.ones((src_seq_length, src_seq_length), dtype=torch.bool), diagonal=1)
                causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).to(self.config.device)
        
                combined_mask = causal_mask.int() * attention_mask.int()
                attention_mask = combined_mask.bool().to(self.config.device)
                # torch.set_printoptions(threshold=torch.inf)
  
        
            attention_out = F.scaled_dot_product_attention(q,k,v, 
                                                           attn_mask=attention_mask, 
                                                           dropout_p=self.config.dropout_p if self.training else 0.0, 
                                                           is_causal=False)
        
        # CROSS ATTENTION
        else:
            tgt_seq_length = tgt.shape[1]
            q = self.q_proj(tgt).reshape(batch, tgt_seq_length, self.config.num_heads, self.head_dim).transpose(1,2).contiguous()
            k = self.k_proj(src).reshape(batch, src_seq_length, self.config.num_heads, self.head_dim).transpose(1,2).contiguous()
            v = self.v_proj(src).reshape(batch, src_seq_length, self.config.num_heads, self.head_dim).transpose(1,2).contiguous()
            
            if attention_mask is not None:
                attention_mask = attention_mask.bool()
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1).repeat(1,1,tgt_seq_length,1)
            
            attention_out = F.scaled_dot_product_attention(q,k,v, 
                                                           attn_mask=attention_mask, 
                                                           dropout_p=self.config.dropout_p if self.training else 0.0, 
                                                           is_causal=False)
        
        attention_out = attention_out.transpose(1,2).flatten(2)
        attention_out = self.out_proj(attention_out)
        return attention_out
            

# Position Wise Feed Forward Network (MLP)
class FeedForward(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.hidden_layer = nn.Linear(in_features=config.d_model, out_features=config.dff) #eg: 512 -> 2048
        self.hidden_dropout = nn.Dropout(p=config.dropout_p)
        self.output_layer = nn.Linear(in_features=config.dff, out_features=config.d_model) #eg : 2048 - > 512
        self.output_dropout = nn.Dropout(p=config.dropout_p)
        
        
        
    def forward(self, x):
        x = self.hidden_layer(x)
        x = F.gelu(x)
        x = self.hidden_dropout(x)
        x = self.output_layer(x)
        x = self.output_dropout(x)
        return x
             
        
# Encoder block
class EncoderBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.multi_head_attention = MultiheadAttention(config=config)
        self.feed_forward = FeedForward(config=config)
        self.layer_norm_1 = nn.LayerNorm(config.d_model)
        self.layer_norm_2 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout_p)
    
    def forward(self, x, attention_mask=None):
        x = x + self.dropout(self.multi_head_attention(src=x, attention_mask=attention_mask))
        x = self.layer_norm_1(x)
        
        x = x + self.feed_forward(x)
        x = self.layer_norm_2(x)
        return x
        
# Decoder block

class DecoderBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.masked_multi_head_attention = MultiheadAttention(config=config)
        self.dropout_masked = nn.Dropout(config.dropout_p)
        
        self.cross_multi_head_attention = MultiheadAttention(config=config)
        self.dropout_cross = nn.Dropout(config.dropout_p)
        
        self.feed_forward = FeedForward(config=config)
        
        self.layer_norm_1 = nn.LayerNorm(config.d_model)
        self.layer_norm_2 = nn.LayerNorm(config.d_model)
        self.layer_norm_3 = nn.LayerNorm(config.d_model)
        
        
    def forward(self, src,tgt, src_attention_mask=None, tgt_attention_mask=None):
        
        tgt = tgt + self.dropout_masked(self.masked_multi_head_attention(tgt, attention_mask=tgt_attention_mask, causal=True))
        tgt = self.layer_norm_1(tgt)
        
        tgt = tgt + self.dropout_cross(self.cross_multi_head_attention(src, tgt, attention_mask=src_attention_mask))
        tgt = self.layer_norm_2(tgt)
        
        tgt = tgt + self.feed_forward(tgt)
        return tgt


# Transformer (put it all together)
class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        
        self.src_embedding = SourceEmbedding(config=config)
        self.tgt_embedding = TargetEmbedding(config=config)
        
        self.position_encoding = PositionEncoding(config=config)
        
        self.encoder = nn.ModuleList(
            [EncoderBlock(config=config) for _ in range(config.num_encoder_layers)]
        )
        
        self.decoder = nn.ModuleList(
            [DecoderBlock(config=config) for _ in range(config.num_decoder_layers)]
        )
        
        self.output = nn.Linear(config.d_model, config.tgt_vocab_size)
        
        ## Init weights
        self.apply(_init_weights_) 
    
    
    
    def forward(self, src_ids, tgt_ids, src_attention_mask=None, tgt_attention_mask=None):
        
        # embed token ids
        src_embed = self.src_embedding(src_ids)
        tgt_embed = self.tgt_embedding(tgt_ids)
        
        # add position encoding
        src_embed = self.position_encoding(src_embed)
        tgt_embed = self.position_encoding(tgt_embed)
        
        for layer in self.encoder:
            src_embed = layer(src_embed, src_attention_mask)
        
        for layer in self.decoder:
            tgt_embed = layer(src_embed, tgt_embed, src_attention_mask, tgt_attention_mask)
        
        pred = self.output(tgt_embed)

        return pred
    
    @torch.no_grad()
    def inference(self, src_ids, tgt_start_id, tgt_end_id, max_seq_length):
        tgt_ids = torch.tensor([tgt_start_id], device=src_ids.device).reshape(1,1)
        
        #Encode the source
        src_embed = self.src_embedding(src_ids)
        src_embed = self.position_encoding(src_embed)
        for layer in self.encoder:
            src_embed = layer(src_embed)
            
        #Generate Target
        for i in range(max_seq_length):
            tgt_embed = self.tgt_embedding(tgt_ids)
            tgt_embed = self.position_encoding(tgt_embed)
            for layer in self.decoder:
                tgt_embed = layer(src_embed, tgt_embed)
            
            tgt_embed = tgt_embed[:, -1]
            
            pred = self.output(tgt_embed)
            pred = pred.argmax(axis=-1).unsqueeze(0)
            tgt_ids = torch.cat([tgt_ids, pred], axis=-1)
            
            if torch.all(pred == tgt_end_id):
                break
        
        return tgt_ids.squeeze().cpu().tolist()
            
        
        

def _init_weights_(module):

    """
    Simple weight intialization taken directly from the huggingface
    `modeling_roberta.py` implementation! 
    """
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
        



if __name__ == "__main__":
    config = TransformerConfig()
    model = Transformer(config=config)
    
    english = torch.randint(low=0, high=1000, size=(1,3))
    res = model.inference(src_ids=english, tgt_start_id=1, tgt_end_id=2, max_seq_length=config.max_seq_length)
    print(res)
        
