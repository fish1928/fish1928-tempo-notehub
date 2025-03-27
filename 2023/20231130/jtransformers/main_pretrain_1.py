import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datetime import datetime
from tqdm import tqdm

from utils import Dotdict
from pretrain_datasets import BookCorpus2000_SC
from collators import Collator_SC
from heads import SimpleDecoderHead_S2S, SimpleEncoderHead_MLM, SimpleEncoderHead_AveragePooling_SC
from components import SaverAndLoader
from builder import build_model_with_mlm_sc_s2s


def train_a_batch(batch, trainer, optimizer=None, scheduler=None):
    trainer.train()
    trainer.forward(**batch())

    loss_s2s, info_acc_s2s = trainer.manager.get_head(SimpleDecoderHead_S2S).get_loss()
    loss_mlm, info_acc_mlm = trainer.manager.get_head(SimpleEncoderHead_MLM).get_loss()
    loss_sc, info_acc_sc = trainer.manager.get_head(SimpleEncoderHead_AveragePooling_SC).get_loss()
    # loss_sim, info_acc_sim = trainer.manager.get_head(SimpleEncoderHead_Similarity).get_loss()

    # crossentropy loss

    # loss_all = loss_s2s * 5
    # loss_all = loss_mlm
    # loss_all = loss_sim
    # loss_all = (loss_s2s + loss_mlm) / 2
    # loss_all = (loss_s2s + loss_mlm + loss_sim) / 3
    # loss_all = (loss_s2s + loss_mlm + loss_sim)
    loss_all = loss_mlm + loss_sc + loss_s2s
    loss_all_value = loss_all.item()

    # print(loss_all)
    loss_all.backward()

    if optimizer:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    # end

    if scheduler:
        scheduler.step()
    # end

    trainer.clear_cache()
    return loss_all_value, Dotdict({'mlm': info_acc_mlm, 'sc': info_acc_sc, 's2s': info_acc_s2s})
    # return loss_all_value, Dotdict({'mlm': info_acc_mlm, 's2s': info_acc_s2s})
    # return loss_all_value, Dotdict({'mlm': info_acc_mlm, 's2s': info_acc_s2s, 'sim': info_acc_sim})
# end

def evaluate_a_batch(batch, trainer, *args, **kwargs):
    trainer.eval()
    with torch.no_grad():
        trainer.forward(**batch())
    # end

    loss_s2s, info_acc_s2s = trainer.manager.get_head(SimpleDecoderHead_S2S).get_loss()
    loss_sc, info_acc_sc = trainer.manager.get_head(SimpleEncoderHead_AveragePooling_SC).get_loss()
    loss_mlm, info_acc_mlm = trainer.manager.get_head(SimpleEncoderHead_MLM).get_loss()

    # crossentropy loss

    # loss_all = loss_s2s * 5
    # loss_all = loss_mlm
    # loss_all = loss_sim
    # loss_all = (loss_s2s + loss_mlm) / 2
    # loss_all = (loss_s2s + loss_mlm + loss_sim) / 3
    # loss_all = (loss_s2s + loss_mlm + loss_sim)
    loss_all = loss_mlm + loss_sc + loss_s2s
    loss_all_value = loss_all.item()

    trainer.clear_cache()
    # return loss_all_value, Dotdict({'mlm': info_acc_mlm, 's2s': info_acc_s2s})
    return loss_all_value, Dotdict({'mlm': info_acc_mlm, 'sc': info_acc_sc, 's2s': info_acc_s2s})


# end

# main start
gpu = 0
torch.cuda.set_device(gpu)

epochs = 3

# source
seq_max = 128
batch_size = 64


# model & head
dim_hidden = 512
dim_feedforward = 512
n_head = 8
n_layer = 8

# optimizer
lr_base_optimizer = 1e-4
betas_optimizer = (0.9, 0.999)
eps_optimizer = 1e-9

# scheduler
warmup = 200

# ### for bookcorpus 2 heads ###
# train_source, valid_source, _ = BookCorpus(split=0.001, used=100000)
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# collator = Collator_BERT(tokenizer, seq_max)
#### for bookcorpus sc #######
num_labels=5
train_source, valid_source, _ = BookCorpus2000_SC(split=0.1, labels=num_labels)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
collator = Collator_SC(tokenizer, seq_max)
##############################



dataloader_train = DataLoader(train_source, batch_size, shuffle=False, collate_fn=collator)
dataloader_eval = DataLoader(valid_source, 1, shuffle=False, collate_fn=collator)

# trainer = Builder.build_model_with_mlm_sc(tokenizer.vocab_size, dim_hidden, dim_feedforward, n_head, n_layer, num_labels)
trainer = build_model_with_mlm_sc_s2s(tokenizer.vocab_size, dim_hidden, dim_feedforward, n_head, n_layer, num_labels)
# trainer= Builder.load_model_with_2heads(tokenizer.vocab_size, dim_hidden, dim_feedforward, n_head, n_layer, loader, 'epoch1')
# trainer = Builder.build_model_with_2heads(tokenizer.vocab_size, dim_hidden, dim_feedforward, n_head, n_layer)

for p in trainer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
    # end
# end

trainer = trainer.to('cuda')

optimizer = torch.optim.Adam(trainer.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
decayRate = 0.96
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)


# optimizer = loader.load_item_state('epoch1', optimizer)
# lr_scheduler = loader.load_item_state('epoch1', lr_scheduler)

loader = SaverAndLoader('checkpoints')
# loader.add_item(trainer.model)
# loader.add_item(trainer.manager.get_head(SimpleEncoderHead_MLM))
# loader.add_item(trainer.manager.get_head(SimpleEncoderHead_AveragePooling_SC))
# loader.add_item(optimizer)
# loader.add_item(lr_scheduler)

name_checkpoint_current = None
name_checkpoint_last = None


for e in range(epochs):

    info_acc_heads_train = Dotdict({
        'mlm': SimpleEncoderHead_MLM.get_info_accuracy_template(),
        'sc': SimpleEncoderHead_AveragePooling_SC.get_info_accuracy_template(),
        's2s': SimpleDecoderHead_S2S.get_info_accuracy_template(),
        # 'sim': SimpleEncoderHead_Similarity.get_info_accuracy_template()
    })

    info_acc_heads_eval = Dotdict({
        'mlm': SimpleEncoderHead_MLM.get_info_accuracy_template(),
        'sc': SimpleEncoderHead_AveragePooling_SC.get_info_accuracy_template(),
        's2s': SimpleDecoderHead_S2S.get_info_accuracy_template(),
        # 'sim': SimpleEncoderHead_Similarity.get_info_accuracy_template()
    })

    # train phase
    losss_per_e = []
    for i, batch in enumerate(tqdm(dataloader_train)):
        loss_current, info_acc_heads_batch = train_a_batch(batch, trainer, optimizer, None)
        info_acc_heads_train += info_acc_heads_batch

        losss_per_e.append(loss_current)
        if i % 100 == 0:
            print('Epoch: {} Batch: {}, loss: {}, rate: {}, acc_mlm: {}, acc_sc: {}, acc_s2s: {}'.format(
                e, i, loss_current, optimizer.param_groups[0]['lr'],
                info_acc_heads_batch.mlm.corrects_masked / info_acc_heads_batch.mlm.num_masked,
                info_acc_heads_batch.sc.corrects / info_acc_heads_batch.sc.num,
                info_acc_heads_batch.s2s.corrects_segmented / info_acc_heads_batch.s2s.num_segmented,
                # sum(info_acc_heads_batch.sim.meansquares) / len(info_acc_heads_batch.sim.meansquares)
            ), end='\r')
        # end
    # end

    loss_average_per_e = sum(losss_per_e) / len(losss_per_e)
    print(
        '[{}] Epoch: {} training ends. Status: Average loss: {}, Average MLM accuracy: {}, Average SC accuracy: {}'.format(
            datetime.utcnow(), e, loss_average_per_e,
            info_acc_heads_train.mlm.corrects_masked / info_acc_heads_train.mlm.num_masked,
            info_acc_heads_train.sc.corrects / info_acc_heads_train.sc.num,
            info_acc_heads_train.s2s.corrects_segmented / info_acc_heads_train.s2s.num_segmented,
            # sum(info_acc_heads_train.sim.meansquares) / len(info_acc_heads_train.sim.meansquares)
        ))

    if e % 2 == 0:
        lr_scheduler.step()  # schedule per 2 epoch
    # end

    # eval phase
    losss_per_e = []
    for i, batch in enumerate(tqdm(dataloader_train)):
        loss_current, info_acc_heads_batch = evaluate_a_batch(batch, trainer)
        info_acc_heads_eval += info_acc_heads_batch

        losss_per_e.append(loss_current)
    # end

    loss_average_per_e = sum(losss_per_e) / len(losss_per_e)
    print(
        '[{}] Epoch: {} Evalutation ends. Status: Average loss: {}, Average MLM accuracy: {}, Average SC accuracy: {}'.format(
            datetime.utcnow(), e, loss_average_per_e,
            info_acc_heads_eval.mlm.corrects_masked / info_acc_heads_eval.mlm.num_masked,
            info_acc_heads_eval.sc.corrects / info_acc_heads_eval.sc.num,
            info_acc_heads_eval.s2s.corrects_segmented / info_acc_heads_eval.s2s.num_segmented,
            # sum(info_acc_heads_eval.sim.meansquares) / len(info_acc_heads_eval.sim.meansquares)
        ))

    name_checkpoint_current = f'epoch_{e}'
    loader.update_checkpoint(name_checkpoint_current, name_checkpoint_last)
    name_checkpoint_last = name_checkpoint_current
# end



trainer.eval()
print()



# For s2s head
def greedy_generate(model, head, tokenizer, collator, **kwargs):
    id_start = tokenizer.id_cls if hasattr(tokenizer, 'id_cls') else collator.id_cls
    id_end = tokenizer.id_sep if hasattr(tokenizer, 'id_sep') else collator.id_sep
    id_pad = tokenizer.id_pad if hasattr(tokenizer, 'id_pad') else collator.id_pad
    size_seq_max = collator.size_seq_max

    ids_encoder_twin = kwargs['ids_encoder']
    masks_encoder_twin = kwargs['masks_encoder']

    ids_decoder_all = []

    for j in range(ids_encoder_twin.shape[0]):
        ids_encoder = ids_encoder_twin[j,].unsqueeze(0)
        masks_encoder = masks_encoder_twin[j,].unsqueeze(0)

        output_encoder = model.embed_and_encode(ids_encoder=ids_encoder, masks_encoder=masks_encoder)
        ids_decoder = torch.zeros(1, 1).fill_(id_start).type_as(ids_encoder.data)

        for i in range(size_seq_max - 1):
            masks_decoder = collator.subsequent_mask(ids_decoder.size(1)).type_as(ids_encoder.data)
            output_decoder = model.embed_and_decode(ids_decoder=ids_decoder, masks_encoder=masks_encoder,
                                                    output_encoder=output_encoder, masks_decoder=masks_decoder)

            output_ffn = head.ffn(output_decoder)
            output_s2s = head.extractor(output_ffn)  # output_mlm = prediction_logits

            logits_nextword = torch.softmax(output_s2s[:, -1],
                                            dim=-1)  # mynote: select dim2=-1, remain=all; last is the next

            id_nextword = torch.argmax(logits_nextword, dim=-1)
            id_nextword = id_nextword.data[0]

            if id_nextword == id_end:
                break
            # end

            ids_decoder = torch.cat([ids_decoder, torch.zeros(1, 1).type_as(ids_encoder.data).fill_(id_nextword)],
                                    dim=1)
        # end

        ids_pad = torch.full((1, size_seq_max - ids_decoder.shape[-1]), id_pad).type_as(ids_decoder.data)

        ids_decoder_all.append(torch.cat([ids_decoder, ids_pad], dim=-1).squeeze(0))
    # end for

    return torch.stack(ids_decoder_all)
# end


# eval_source = to_map_style_dataset(valid_iter)
dataloader_eval = DataLoader(valid_source, 1, shuffle=False, collate_fn=collator)

for i, batch in enumerate(dataloader_eval):
    info_batch = batch()
    result = greedy_generate(trainer.model, trainer.manager.get_head(SimpleDecoderHead_S2S), tokenizer, collator,
                             **info_batch)

    result_cpu_list = result.cpu().tolist()
    labels_decoder_cpu_list = info_batch['labels_decoder'].cpu().tolist()

    for result_cpu, labels_decoder in zip(result_cpu_list, labels_decoder_cpu_list):
        sentence_predicted = tokenizer.decode(result_cpu).split(' [PAD]')[0]
        sentence_origin = tokenizer.decode(labels_decoder).split(' [PAD]')[0]

        print('source: {}\ntarget: {}\n\n'.format(sentence_origin, sentence_predicted))
    # end

    if i >= 5:
        break
    # end
# end 


def decode_output(out_mlm, masks_masked_perbatch, labels_mlm, ids_encoder, tokenizer):
    # print segments
    # sentence_predicts = tokenizer.decode(out_mlm.softmax(dim=-1).argmax(dim=-1).masked_select(segments_encoder[:, 0, :]).numpy().tolist())
    # sentence_labels = tokenizer.decode(labels_mlm.masked_select(segments_encoder[:, 0, :]).numpy().tolist())
    # sentence_inputs = tokenizer.decode(ids_encoder.masked_select(segments_encoder[:, 0, :]).numpy().tolist())

    # print masks
    sentence_predicts = tokenizer.decode(
        out_mlm.softmax(dim=-1).argmax(dim=-1).masked_select(masks_masked_perbatch).numpy().tolist())
    sentence_labels = tokenizer.decode(labels_mlm.masked_select(masks_masked_perbatch).numpy().tolist())
    sentence_inputs = tokenizer.decode(ids_encoder.masked_select(masks_masked_perbatch).numpy().tolist())

    #     sentence_predicts = tokenizer.decode(out_mlm.softmax(dim=-1).argmax(dim=-1).numpy().tolist()[0])
    #     sentence_labels = tokenizer.decode(labels_mlm.numpy().tolist()[0])
    #     sentence_inputs = tokenizer.decode(ids_encoder.numpy().tolist()[0])

    predicts_masked = out_mlm.softmax(dim=-1).argmax(dim=-1).masked_select(masks_masked_perbatch)
    labels_masked = labels_mlm.masked_select(masks_masked_perbatch)

    acc = torch.count_nonzero(predicts_masked == labels_masked) / labels_masked.shape[0]
    # acc = torch.count_nonzero(out_mlm.softmax(dim=-1).argmax(dim=-1).view(-1) == labels_mlm.view(-1)) / labels_mlm.view(-1).shape[0]
    return acc, sentence_labels, sentence_inputs, sentence_predicts


# end


# eval_source = to_map_style_dataset(valid_iter)
dataloader_eval = DataLoader(valid_source, 1, shuffle=False, collate_fn=collator)
# dataloader_eval = DataLoader(train_source, 1, shuffle=False, collate_fn=collator)

for i, batch in enumerate(dataloader_eval):
    info_batch = batch()
    trainer.forward(**info_batch)

    head = trainer.manager.get_head(SimpleEncoderHead_MLM)
    out_mlm = head.cache.output
    loss_mlm, _ = head.get_loss()

    out_mlm = out_mlm.cpu().detach()
    loss_mlm = loss_mlm.cpu().detach()
    labels_mlm = info_batch['labels_encoder'].cpu().detach()
    masks_encoder = info_batch['masks_encoder'].cpu().detach()
    segments_encoder = info_batch['segments_encoder'].cpu().detach()
    ids_encoder = info_batch['ids_encoder'].cpu().detach()

    masks_masked = torch.logical_xor(masks_encoder, segments_encoder) & segments_encoder  # True is masked
    masks_masked_perbatch = masks_masked[:, 0, :]

    for j in range(masks_masked_perbatch.shape[0]):
        acc, sentence_labels, sentence_inputs, sentence_predicts = decode_output(out_mlm[j,].unsqueeze(0),
                                                                                 masks_masked_perbatch[j,].unsqueeze(0),
                                                                                 labels_mlm[j,].unsqueeze(0),
                                                                                 ids_encoder[j,].unsqueeze(0),
                                                                                 tokenizer)
        print('loss: {}, acc: {}\nsource: {}\ninput: {}\npredict: {}\n\n'.format(loss_mlm.item(), acc, sentence_labels,
                                                                                 sentence_inputs, sentence_predicts))

    if i >= 5:
        break
    # end
# end