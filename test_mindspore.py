#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的测试脚本，用于验证 MindSpore 版本的 Informer 模型是否正确迁移
"""

import sys
import numpy as np
import mindspore as ms
import mindspore.context as context
from mindspore import Tensor

# 设置MindSpore上下文
context.set_context(mode=context.GRAPH_MODE, device_target='CPU')

# 添加路径
sys.path.append('./models')
sys.path.append('./utils')

def test_basic_imports():
    """测试基本导入"""
    print("Testing basic imports...")
    try:
        from models.embed_mindspore import DataEmbedding, PositionalEmbedding, TokenEmbedding
        from models.attn_mindspore import FullAttention, ProbAttention, AttentionLayer
        from models.encoder_mindspore import Encoder, EncoderLayer, ConvLayer
        from models.decoder_mindspore import Decoder, DecoderLayer
        from models.model_mindspore import Informer, InformerStack
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_model_creation():
    """测试模型创建"""
    print("\nTesting model creation...")
    try:
        from models.model_mindspore import Informer
        
        # 创建一个简单的Informer模型
        model = Informer(
            enc_in=7,
            dec_in=7,
            c_out=7,
            seq_len=96,
            label_len=48,
            out_len=24,
            factor=5,
            d_model=64,  # 使用较小的维度以节省内存
            n_heads=4,
            e_layers=2,
            d_layers=1,
            d_ff=128,
            dropout=0.1,
            attn='prob',
            embed='timeF',
            freq='h',
            activation='gelu',
            output_attention=False,
            distil=True,
            mix=True,
            device='CPU'
        )
        print("✓ Model created successfully")
        return model
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return None

def test_forward_pass(model):
    """测试前向传播"""
    print("\nTesting forward pass...")
    try:
        # 创建测试数据
        batch_size = 2
        seq_len = 96
        label_len = 48
        pred_len = 24
        enc_in = 7
        
        # 输入数据
        x_enc = Tensor(np.random.randn(batch_size, seq_len, enc_in), ms.float32)
        x_mark_enc = Tensor(np.random.randn(batch_size, seq_len, 4), ms.float32)  # 时间特征
        x_dec = Tensor(np.random.randn(batch_size, label_len + pred_len, enc_in), ms.float32)
        x_mark_dec = Tensor(np.random.randn(batch_size, label_len + pred_len, 4), ms.float32)
        
        # 前向传播
        model.set_train(False)
        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        print(f"✓ Forward pass successful, output shape: {output.shape}")
        return True
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False

def test_attention_mechanism():
    """测试注意力机制"""
    print("\nTesting attention mechanism...")
    try:
        from models.attn_mindspore import FullAttention, ProbAttention, AttentionLayer
        
        # 测试FullAttention
        attention = FullAttention(mask_flag=False, attention_dropout=0.1)
        B, L, H, E = 2, 10, 4, 16
        
        queries = Tensor(np.random.randn(B, L, H, E), ms.float32)
        keys = Tensor(np.random.randn(B, L, H, E), ms.float32)
        values = Tensor(np.random.randn(B, L, H, E), ms.float32)
        
        output, attn = attention(queries, keys, values, None)
        print(f"✓ FullAttention test passed, output shape: {output.shape}")
        
        # 测试ProbAttention
        prob_attention = ProbAttention(mask_flag=False, attention_dropout=0.1)
        output, attn = prob_attention(queries, keys, values, None)
        print(f"✓ ProbAttention test passed, output shape: {output.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Attention mechanism test failed: {e}")
        return False

def test_embedding_layers():
    """测试嵌入层"""
    print("\nTesting embedding layers...")
    try:
        from models.embed_mindspore import DataEmbedding, PositionalEmbedding, TokenEmbedding
        
        # 测试TokenEmbedding
        token_emb = TokenEmbedding(c_in=7, d_model=64)
        x = Tensor(np.random.randn(2, 96, 7), ms.float32)
        output = token_emb(x)
        print(f"✓ TokenEmbedding test passed, output shape: {output.shape}")
        
        # 测试PositionalEmbedding
        pos_emb = PositionalEmbedding(d_model=64)
        output = pos_emb(x)
        print(f"✓ PositionalEmbedding test passed, output shape: {output.shape}")
        
        # 测试DataEmbedding
        data_emb = DataEmbedding(c_in=7, d_model=64, embed_type='timeF', freq='h')
        x_mark = Tensor(np.random.randn(2, 96, 4), ms.float32)
        output = data_emb(x, x_mark)
        print(f"✓ DataEmbedding test passed, output shape: {output.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Embedding layers test failed: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 50)
    print("Informer MindSpore 版本测试")
    print("=" * 50)
    
    # 测试基本导入
    if not test_basic_imports():
        return
    
    # 测试嵌入层
    if not test_embedding_layers():
        return
    
    # 测试注意力机制
    if not test_attention_mechanism():
        return
    
    # 测试模型创建
    model = test_model_creation()
    if model is None:
        return
    
    # 测试前向传播
    if not test_forward_pass(model):
        return
    
    print("\n" + "=" * 50)
    print("✓ 所有测试通过！MindSpore 版本迁移成功！")
    print("=" * 50)

if __name__ == "__main__":
    main()
