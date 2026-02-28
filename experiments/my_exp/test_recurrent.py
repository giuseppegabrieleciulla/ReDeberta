import torch
from DeBERTa.deberta.config import ModelConfig
from DeBERTa.apps.models.sequence_classification import SequenceClassificationModel

if __name__ == "__main__":
    import os
    os.environ['OMP_NUM_THREADS'] = '1'
    
    # Create dummy config
    config = ModelConfig()
    config.hidden_size = 32
    config.intermediate_size = 64
    config.num_attention_heads = 4
    config.num_hidden_layers = 4
    config.vocab_size = 100
    config.max_position_embeddings = 64
    config.type_vocab_size = 2
    
    # Enable recurrent settings
    config.use_recurrent = True
    config.recurrent_layer = 1
    config.ponder_penalty = 0.01
    
    kwargs = {
        'num_labels': 2,
        'drop_out': 0.1,
        'pre_trained': None
    }
    
    # Create dummy model with pre-trained=None to skip load_model_state
    print("Instantiating SequenceClassificationModel...")
    
    # We must patch apply_state so it doesn't crash since pre_trained is None but we still want to test the initialization
    old_use_recurrent = config.use_recurrent
    # We will instantiate it first with use_recurrent=False so it builds standard DeBERTa
    config.use_recurrent = False
    model = SequenceClassificationModel(config, **kwargs)
    
    # Now simulate what apps/run.py and __init__ do:
    from DeBERTa.deberta.recurrent_encoder import RecurrentBertEncoder
    
    recurrent_encoder = RecurrentBertEncoder(model.deberta.config)
    recurrent_layer_idx = 1
    recurrent_encoder.layer.load_state_dict(model.deberta.encoder.layer[recurrent_layer_idx].state_dict())
    model.deberta.encoder = recurrent_encoder
    model.config.use_recurrent = True
    
    print("Model created.")
    
    # Create dummy input
    batch_size = 2
    seq_length = 10
    input_ids = torch.randint(0, 100, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    token_type_ids = torch.zeros(batch_size, seq_length)
    labels = torch.randint(0, 2, (batch_size,))
    
    print("Running forward pass...")
    outputs = model(input_ids=input_ids, input_mask=attention_mask, type_ids=token_type_ids, labels=labels)
    
    print("Output keys:", outputs.keys())
    print("Loss:", outputs['loss'].item())
    print("Logits shape:", outputs['logits'].shape)
    
    # Test gradients / backward pass
    loss = outputs['loss']
    loss.backward()
    
    print("Backward pass successful!")
    print("Halting head weight grad:", model.deberta.encoder.halting_head.weight.grad)
