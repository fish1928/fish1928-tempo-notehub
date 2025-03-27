from core import SimpleEmbedder
from layers import SimpleEncoderLayer, SimpleDecoderLayer, SimpleTransformerStack, SimpleEncoderDecoder
from heads import SimpleEncoderHead_MLM, SimpleDecoderHead_S2S, SimpleEncoderHead_AveragePooling_SC, SimpleEncoderHead_Similarity
from components import HeadManager, Trainer

def build_model_with_mlm_v2(size_vocab, dim_hidden, dim_feedforward, n_head, n_layer):
    embedder_encoder = SimpleEmbedder(size_vocab=size_vocab, dim_hidden=dim_hidden)
    sample_encoder = SimpleEncoderLayer(dim_hidden, dim_feedforward, n_head)
    encoderstack = SimpleTransformerStack(sample_encoder, n_layer)

    model = SimpleEncoderDecoder(encoderstack, None, embedder_encoder, None)
    head_mlm = SimpleEncoderHead_MLM(model, size_vocab, dim_hidden)

    manager = HeadManager().register(head_mlm)
    trainer = Trainer(model=model, manager=manager)

    return trainer

# end

def build_model_with_s2s_v2(size_vocab, dim_hidden, dim_feedforward, n_head, n_layer):
    embedder_encoder = SimpleEmbedder(size_vocab=size_vocab, dim_hidden=dim_hidden)
    sample_encoder = SimpleEncoderLayer(dim_hidden, dim_feedforward, n_head)
    encoderstack = SimpleTransformerStack(sample_encoder, n_layer)

    embedder_decoder = SimpleEmbedder(size_vocab=size_vocab, dim_hidden=dim_hidden)
    sample_decoder = SimpleDecoderLayer(dim_hidden, dim_feedforward, n_head)
    decoderstack = SimpleTransformerStack(sample_decoder, n_layer)

    model = SimpleEncoderDecoder(encoderstack, decoderstack, embedder_encoder, embedder_decoder, pooling=True)
    head_s2s = SimpleDecoderHead_S2S(model, size_vocab, dim_hidden)

    manager = HeadManager().register(head_s2s)
    trainer = Trainer(model=model, manager=manager)

    return trainer

# end

def build_model_with_2heads(size_vocab, dim_hidden, dim_feedforward, n_head, n_layer):
    embedder_encoder = SimpleEmbedder(size_vocab=size_vocab, dim_hidden=dim_hidden)
    sample_encoder = SimpleEncoderLayer(dim_hidden, dim_feedforward, n_head)
    encoderstack = SimpleTransformerStack(sample_encoder, n_layer)

    embedder_decoder = SimpleEmbedder(size_vocab=size_vocab, dim_hidden=dim_hidden)
    sample_decoder = SimpleDecoderLayer(dim_hidden, dim_feedforward, n_head)
    decoderstack = SimpleTransformerStack(sample_decoder, n_layer)

    model = SimpleEncoderDecoder(encoderstack, decoderstack, embedder_encoder, embedder_decoder, pooling=True)
    head_s2s = SimpleDecoderHead_S2S(model, size_vocab, dim_hidden)
    head_mlm = SimpleEncoderHead_MLM(model, size_vocab, dim_hidden)

    manager = HeadManager().register(head_s2s).register(head_mlm)
    trainer = Trainer(model=model, manager=manager)

    return trainer

# end

def load_model_with_2heads(size_vocab, dim_hidden, dim_feedforward, n_head, n_layer, loader, name_checkpoint):
    embedder_encoder = SimpleEmbedder(size_vocab=size_vocab, dim_hidden=dim_hidden)
    sample_encoder = SimpleEncoderLayer(dim_hidden, dim_feedforward, n_head)
    encoderstack = SimpleTransformerStack(sample_encoder, n_layer)

    embedder_decoder = SimpleEmbedder(size_vocab=size_vocab, dim_hidden=dim_hidden)
    sample_decoder = SimpleDecoderLayer(dim_hidden, dim_feedforward, n_head)
    decoderstack = SimpleTransformerStack(sample_decoder, n_layer)

    model = SimpleEncoderDecoder(encoderstack, decoderstack, embedder_encoder, embedder_decoder, pooling=True)
    head_s2s = SimpleDecoderHead_S2S(model, size_vocab, dim_hidden)
    head_mlm = SimpleEncoderHead_MLM(model, size_vocab, dim_hidden)

    loader.load_item_state(name_checkpoint, model)
    loader.load_item_state(name_checkpoint, head_s2s)
    loader.load_item_state(name_checkpoint, head_mlm)

    manager = HeadManager().register(head_s2s).register(head_mlm)
    trainer = Trainer(model=model, manager=manager)

    return trainer

# end

def build_model_with_sim_v2(size_vocab, dim_hidden, dim_feedforward, n_head, n_layer):
    embedder_encoder = SimpleEmbedder(size_vocab=size_vocab, dim_hidden=dim_hidden)
    sample_encoder = SimpleEncoderLayer(dim_hidden, dim_feedforward, n_head)
    encoderstack = SimpleTransformerStack(sample_encoder, n_layer)

    model = SimpleEncoderDecoder(encoderstack, None, embedder_encoder, None, pooling=True)
    head_sim = SimpleEncoderHead_Similarity()

    manager = HeadManager().register(head_sim)
    trainer = Trainer(model=model, manager=manager)

    return trainer

# end

def build_model_with_mlm_sc(size_vocab, dim_hidden, dim_feedforward, n_head, n_layer, num_labels):
    embedder_encoder = SimpleEmbedder(size_vocab=size_vocab, dim_hidden=dim_hidden)
    sample_encoder = SimpleEncoderLayer(dim_hidden, dim_feedforward, n_head)
    encoderstack = SimpleTransformerStack(sample_encoder, n_layer)

    model = SimpleEncoderDecoder(encoderstack, None, embedder_encoder, None, pooling=True)
    head_mlm = SimpleEncoderHead_MLM(model, size_vocab, dim_hidden)
    head_sc = SimpleEncoderHead_AveragePooling_SC(num_labels, dim_hidden)

    manager = HeadManager().register(head_mlm).register(head_sc)
    trainer = Trainer(model=model, manager=manager)

    return trainer

# end

def build_model_with_mlm_sc_s2s(size_vocab, dim_hidden, dim_feedforward, n_head, n_layer, num_labels):
    embedder_encoder = SimpleEmbedder(size_vocab=size_vocab, dim_hidden=dim_hidden)
    sample_encoder = SimpleEncoderLayer(dim_hidden, dim_feedforward, n_head)
    encoderstack = SimpleTransformerStack(sample_encoder, n_layer)

    embedder_decoder = SimpleEmbedder(size_vocab=size_vocab, dim_hidden=dim_hidden)
    sample_decoder = SimpleDecoderLayer(dim_hidden, dim_feedforward, n_head)
    decoderstack = SimpleTransformerStack(sample_decoder, n_layer)

    model = SimpleEncoderDecoder(encoderstack, decoderstack, embedder_encoder, embedder_decoder, pooling=True)
    head_mlm = SimpleEncoderHead_MLM(model, size_vocab, dim_hidden)
    head_sc = SimpleEncoderHead_AveragePooling_SC(num_labels, dim_hidden)
    head_s2s = SimpleDecoderHead_S2S(model, size_vocab, dim_hidden)

    manager = HeadManager().register(head_mlm).register(head_sc).register(head_s2s)
    trainer = Trainer(model=model, manager=manager)

    return trainer

# end

def build_model_with_3heads(size_vocab, dim_hidden, dim_feedforward, n_head, n_layer):
    embedder_encoder = SimpleEmbedder(size_vocab=size_vocab, dim_hidden=dim_hidden)
    sample_encoder = SimpleEncoderLayer(dim_hidden, dim_feedforward, n_head)
    encoderstack = SimpleTransformerStack(sample_encoder, n_layer)

    embedder_decoder = SimpleEmbedder(size_vocab=size_vocab, dim_hidden=dim_hidden)
    sample_decoder = SimpleDecoderLayer(dim_hidden, dim_feedforward, n_head)
    decoderstack = SimpleTransformerStack(sample_decoder, n_layer)

    model = SimpleEncoderDecoder(encoderstack, decoderstack, embedder_encoder, embedder_decoder, pooling=True)
    head_s2s = SimpleDecoderHead_S2S(model, size_vocab, dim_hidden)
    head_mlm = SimpleEncoderHead_MLM(model, size_vocab, dim_hidden)
    head_sim = SimpleEncoderHead_Similarity()

    manager = HeadManager().register(head_s2s).register(head_mlm).register(head_sim)
    trainer = Trainer(model=model, manager=manager)

    return trainer
# end
# end