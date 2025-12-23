import hydra
import numpy as np
import onnxruntime as ort
import torch
import torch.onnx
import onnx
from omegaconf import DictConfig
from transformers import BertTokenizer
import onnxruntime as ort
import os

from model import LSTMClassifierLightning

_TOKENIZER_VOCAB_SIZE = None

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def convert_to_onnx(config: DictConfig):
    model = LSTMClassifierLightning.load_from_checkpoint(config.infer.ckpt_path, weights_only=False)
    model.eval()
    
    tokenizer = BertTokenizer.from_pretrained(config.module.tokenizer_name)
    _TOKENIZER_VOCAB_SIZE = tokenizer.vocab_size

    dummy_input_ids = torch.randint(0, _TOKENIZER_VOCAB_SIZE, (1, 384), dtype=torch.long)
    dummy_attention_mask = torch.ones((1, 384), dtype=torch.long)

    input_names = ["input_ids", "attention_mask"]
    output_names = ["output"]
    
    dynamic_axes = {
        'input_ids': {0: 'batch_size', 1: 'sequence_length'},
        'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
        'output': {0: 'batch_size'}
    }

    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask),
        "model.onnx",
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,      
        verbose=False,
        dynamo=False
    )
    onnx_path = os.path.join(config.model.model_local_path, "model.onnx")
    onnx_path = "model.onnx"

    print("Model successfully converted to ONNX")
    if os.path.exists(onnx_path):
        print(f"ONNX file created successfully: {os.path.getsize(onnx_path)} bytes")
        verify_onnx_model(onnx_path, vocab_size=tokenizer.vocab_size)
    else:
        print(f"ERROR: ONNX file not created at {onnx_path}")

def verify_onnx_model(onnx_path, batch_size=1, seq_len=384, vocab_size=None):
    """Проверка корректности экспортированной модели"""
    
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    
    ort_session = ort.InferenceSession(onnx_path)
    
    dummy_input_ids = np.random.randint(0, vocab_size, (batch_size, seq_len)).astype(np.int64)
    dummy_attention_mask = np.ones((batch_size, seq_len), dtype=np.int64)
    
    ort_inputs = {
        'input_ids': dummy_input_ids,
        'attention_mask': dummy_attention_mask
    }
    
    ort_outputs = ort_session.run(None, ort_inputs)
    
    print(f"ONNX модель загружена успешно")
    print(f"Выходная форма: {ort_outputs[0].shape}")
    
    return ort_outputs


if __name__ == "__main__":
    convert_to_onnx()