import argparse
from utils import read_pickle, Datagen_set, Datagen_deepcom, Datagen_tree, Datagen_binary, bleu4
from models import Seq2seqModel, CodennModel, ChildsumModel, MultiwayModel, NaryModel
import numpy as np
import os
import tensorflow as tf
from tqdm import tqdm
from joblib import delayed, Parallel
import json

from transformer import Transformer
from utils import rouge_n, meteor
tf.enable_eager_execution()

# parse argments

parser = argparse.ArgumentParser(description='Source Code Generation')

parser.add_argument('-m', "--method", type=str, nargs="?", required=True,
                    choices=['seq2seq', 'deepcom', 'codenn', 'childsum', 'multiway', "nary", "transformer"],
                    help='Encoder method')
parser.add_argument('-d', "--dim", type=int, nargs="?", required=False, default=512,
                    help='Representation dimension')
parser.add_argument("--embed", type=int, nargs="?", required=False, default=256,
                    help='Representation dimension')
parser.add_argument("--drop", type=float, nargs="?", required=False, default=.5,
                    help="Dropout rate")
parser.add_argument('-r', "--lr", type=float, nargs="?", required=True,
                    help='Learning rate')
parser.add_argument('-b', "--batch", type=int, nargs="?", required=True,
                    help='Mini batch size')
parser.add_argument('-e', "--epochs", type=int, nargs="?", required=True,
                    help='Epoch number')
parser.add_argument('-g', "--gpu", type=str, nargs="?", required=True,
                    help='What GPU to use')
parser.add_argument('-l', "--layer", type=int, nargs="?", required=False, default=1,
                    help='Number of layers')
parser.add_argument("--val", type=str, nargs="?", required=False, default="BLEU",
                    help='Validation method')
parser.add_argument("--crl", action="store_true")
parser.add_argument("--debug", action="store_true")

args = parser.parse_args()

name = args.method + "_dim" + str(args.dim) + "_embed" + str(args.embed)
name = name + "_drop" + str(args.drop)
name = name + "_lr" + str(args.lr) + "_batch" + str(args.batch)
name = name + "_epochs" + str(args.epochs) + "_layer" + str(args.layer) + "NEW_skip_size100"

if args.crl:
    name = name + "_crl"

checkpoint_dir = "./models/" + name


# set tf eager

tfe = tf.contrib.eager
config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(
        visible_device_list=args.gpu))
# config.gpu_options.allow_growth = True
session = tf.Session(config=config)
tf.enable_eager_execution(config=config)
os.makedirs("./logs/" + name, exist_ok=True)
writer = tf.contrib.summary.create_file_writer("./logs/" + name, flush_millis=10000)


# load data

trn_data = read_pickle("dataset/nl/train.pkl")
vld_data = read_pickle("dataset/nl/valid.pkl")
tst_data = read_pickle("dataset/nl/test.pkl")
code_i2w = read_pickle("dataset/code_i2w.pkl")
code_w2i = read_pickle("dataset/code_w2i.pkl")
nl_i2w = read_pickle("dataset/nl_i2w.pkl")
nl_w2i = read_pickle("dataset/nl_w2i.pkl")

if args.debug:
    trn_x, trn_y_raw = zip(*sorted(trn_data.items())[:1000])
    vld_x, vld_y_raw = zip(*sorted(vld_data.items())[:1000])
    tst_x, tst_y_raw = zip(*sorted(tst_data.items())[:1000])
else:
    trn_x, trn_y_raw = zip(*sorted(trn_data.items()))
    vld_x, vld_y_raw = zip(*sorted(vld_data.items()))
    tst_x, tst_y_raw = zip(*sorted(tst_data.items()))

trn_y = [[nl_w2i[t] if t in nl_w2i.keys() else nl_w2i["<UNK>"] for t in l] for l in trn_y_raw]
vld_y = [[nl_w2i[t] if t in nl_w2i.keys() else nl_w2i["<UNK>"] for t in l] for l in vld_y_raw]
tst_y = [[nl_w2i[t] if t in nl_w2i.keys() else nl_w2i["<UNK>"] for t in l] for l in tst_y_raw]


total_steps = len(trn_x) // args.batch + int(len(trn_x) % args.batch > 0) * args.epochs
warmup_steps = int(total_steps * 0.1)
print("Total steps: {}".format(total_steps))
print("Warmup steps: {}".format(warmup_steps))

# setting model

if args.method in ['seq2seq', 'deepcom']:
    Model = Seq2seqModel
elif args.method in ['codenn']:
    Model = CodennModel
elif args.method in ['childsum']:
    Model = ChildsumModel
elif args.method in ['multiway']:
    Model = MultiwayModel
elif args.method in ['nary']:
    Model = NaryModel
elif args.method in ['transformer']:
    Model = Transformer


if args.method in ['transformer']:
    model = Model(
        d_model=args.dim,
        dff=4*args.dim,
        ss_dim=args.dim,
        input_vocab_size=len(code_w2i),
        target_vocab_size=len(nl_w2i),
        dropout=args.drop,
        lr=args.lr,
        layer=args.layer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        crl=args.crl
    )
else:
    model = Model(args.dim, args.dim, args.dim, len(code_w2i), len(nl_w2i),
                dropout=args.drop, lr=args.lr, layer=args.layer)
epochs = args.epochs
batch_size = args.batch
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
root = tfe.Checkpoint(model=model)
history = {"loss": [], "loss_val": [], "bleu_val": []}


# Setting Data Generator

if args.method in ['seq2seq', 'deepcom', 'transformer']:
    Datagen = Datagen_deepcom
elif args.method in ['codenn']:
    Datagen = Datagen_set
elif args.method in ['childsum', 'multiway']:
    Datagen = Datagen_tree
elif args.method in ['nary']:
    Datagen = Datagen_binary


trn_gen = Datagen(trn_x, trn_y, batch_size, code_w2i, nl_i2w, train=True)
vld_gen = Datagen(vld_x, vld_y, batch_size, code_w2i, nl_i2w, train=False)
tst_gen = Datagen(tst_x, tst_y, batch_size, code_w2i, nl_i2w, train=False)


# training
with writer.as_default(), tf.contrib.summary.always_record_summaries():

    for epoch in range(1, epochs + 1):

        # train
        loss_tmp = []
        t = tqdm(trn_gen(0))
        for bid, (x, y, _, _) in enumerate(t):
            # if bid < 1:
                # print(len(code_i2w), list(code_i2w.values())[:10])
                # print(len(nl_i2w), list(nl_i2w.values())[:10])
                # print("**** SAMPLE ****")
                # print(nl_w2i["<s>"], nl_w2i["</s>"], nl_w2i["the"])
                # print(x[0, :])
                # print([code_i2w[n] for n in x[0, :].numpy().tolist() if n != -1])
                # print(y[0, :])
                # print([nl_i2w[n] for n in y[0, :].numpy().tolist() if n != -1])
            loss_tmp.append(model.train_on_batch(x, y))
            t.set_description("epoch:{:03d}, loss = {}".format(epoch, np.mean(loss_tmp)))
        history["loss"].append(np.sum(loss_tmp) / len(t))
        tf.contrib.summary.scalar("loss", np.sum(loss_tmp) / len(t), step=epoch)

        # validate loss
        loss_tmp = []
        t = tqdm(vld_gen(0))
        for x, y, _, _ in t:
            loss_tmp.append(model.evaluate_on_batch(x, y))
            t.set_description("epoch:{:03d}, loss_val = {}".format(epoch, np.mean(loss_tmp)))
        history["loss_val"].append(np.sum(loss_tmp) / len(t))
        tf.contrib.summary.scalar("loss_val", np.sum(loss_tmp) / len(t), step=epoch)

        # validate bleu
        preds = []
        trues = []
        bleus = []
        t = tqdm(vld_gen(0))
        for x, y, _, y_raw in t:
            res = model.translate(x, nl_i2w, nl_w2i)
            preds += res
            trues += [s[1:-1] for s in y_raw]
            bleus += [bleu4(tt, p) for tt, p in zip(trues, preds)]
            t.set_description("epoch:{:03d}, bleu_val = {}".format(epoch, np.mean(bleus)))
        history["bleu_val"].append(np.mean(bleus))
        tf.contrib.summary.scalar("bleu_val", np.mean(bleus), step=epoch)

        # checkpoint
        if history["bleu_val"][-1] == max(history["bleu_val"]):
            checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
            hoge = root.save(file_prefix=checkpoint_prefix)
            best_model = hoge
            print("Now best model is {}".format(best_model))


# load final weight

print("Restore {}".format(best_model))
root.restore(best_model)

# evaluation

preds = []
trues = []
for x, y, _, y_raw in tqdm(tst_gen(0), "Testing"):
    res = model.translate(x, nl_i2w, nl_w2i)
    res = list([
        ["<s>"] if len(r) == 0 else r
        for r in res
    ])
    preds += res
    trues += [s[1:-1] for s in y_raw]

bleus = Parallel(n_jobs=-1)(delayed(bleu4)(t, p) for t, p in (list(zip(trues, preds))))
rouge_score = rouge_n(trues, preds)
meteor_scores, macro_meteor_score = meteor(trues, preds)

history["bleus"] = sum(bleus) / len(bleus)
history["rouge-1"] = rouge_score["macro_rouge1"]
history["rouge-2"] = rouge_score["macro_rouge2"]
history["rouge-L"] = rouge_score["macro_rougeL"]
history["meteor"] = macro_meteor_score
history["preds"] = preds
history["trues"] = trues
history["numbers"] = [int(x.split("/")[-1]) for x in tst_x]

with open(os.path.join(checkpoint_dir, "history.json"), "w") as f:
    json.dump(history, f)
