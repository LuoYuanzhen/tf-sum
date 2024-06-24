import tensorflow as tf


from tensorflow import Variable

from layers import EncoderLayer, DecoderLayer
from utils import positional_encoding


class Encoder(tf.keras.layers.Layer):
    def __init__(
        self, 
        num_layers, 
        d_model, 
        num_heads, 
        dff, 
        input_vocab_size, 
        maximum_position_encoding, 
        rate=0.1
    ):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.enc_layers = [
            EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(rate)
    
    def call(self, x, mask, training=True):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = tf.nn.relu(x)
        x = self.embedding(x)
        # x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, mask, training)

        return x
    

class Decoder(tf.keras.layers.Layer):
    def __init__(
            self, 
            num_layers, 
            d_model, 
            num_heads, 
            dff, 
            target_vocab_size, 
            maximum_position_encoding,
            rate=0.1
        ):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [
            DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(rate)
    
    def call(self, x, enc_output, look_ahead_mask, padding_mask, training=True):
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = tf.nn.relu(x)
        x = self.embedding(x)
        # x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](
                x, enc_output, look_ahead_mask, padding_mask, training
            )

            attention_weights["decoder_layer{}_block1".format(i + 1)] = block1
            attention_weights["decoder_layer{}_block2".format(i + 1)] = block2
        
        return x, attention_weights
    

class Transformer(tf.keras.Model):
    def __init__(
        self,
        d_model,
        dff,
        ss_dim,
        input_vocab_size,
        target_vocab_size,
        num_heads=8,
        dropout=0.1,
        lr=1e-3,
        layer=6,
        warmup_steps=100,
        total_steps=1000,
        crl=True
    ):
        super(Transformer, self).__init__()
        self.max_lr = lr
        self.lr = Variable(0.)

        self.loss_object = tf.nn.sparse_softmax_cross_entropy_with_logits
        self.optimizer = tf.train.AdamOptimizer(self.lr)
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.global_step = 1

        pe_input, pe_target = input_vocab_size, target_vocab_size
        self.encoder = Encoder(
            layer, d_model, num_heads, dff, input_vocab_size, pe_input, dropout
        )
        self.decoder = Decoder(
            layer, d_model, num_heads, dff, target_vocab_size, pe_target, dropout
        )

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

        """ ss encoders """
        self.crl = crl
        if crl:
            self.inp_encoder = Encoder(
                1, d_model // 2, num_heads, ss_dim, input_vocab_size, pe_input, dropout
            )
            self.pred_encoder = Encoder(
                1, d_model // 2, num_heads, ss_dim, target_vocab_size, pe_target, dropout
            )
    
    def update_lr(self):
        lr = tf.where(
            self.global_step < self.warmup_steps,
            self.global_step * (self.max_lr / self.warmup_steps),
            self.max_lr - self.max_lr * min((self.global_step - self.warmup_steps) / (self.total_steps - self.warmup_steps), 1) / self.total_steps,
        )
        self.lr.assign(lr)
    
    def _create_masks(self, enc_in, dec_out):
        enc_padding_mask = tf.cast(
            tf.math.equal(enc_in, -1), tf.float32
        )[:, tf.newaxis, tf.newaxis, :]

        dec_padding_mask = tf.cast(
            tf.math.equal(enc_in, -1), tf.float32
        )[:, tf.newaxis, tf.newaxis, :]

        size = tf.shape(dec_out)[1]
        look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        dec_target_padding_mask = tf.cast(
            tf.math.equal(dec_out, -1), tf.float32
        )[:, tf.newaxis, tf.newaxis, :]
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return enc_padding_mask, combined_mask, dec_padding_mask
    
    def call(
        self,
        inp,
        tar,
        enc_padding_mask,
        look_ahead_mask,
        dec_padding_mask,
        training=True,
    ):
        enc_output = self.encoder(inp, enc_padding_mask, training)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, look_ahead_mask, dec_padding_mask, training
        )

        final_output = self.final_layer(dec_output)

        return final_output, attention_weights
    
    def translate(
        self,
        x,
        nl_i2w,
        nl_w2i,
        max_length=30,
    ):
        batch_size = len(x)
        preds = [[] for _ in range(batch_size)]

        enc_padding_mask = tf.cast(
            tf.math.equal(x, -1), tf.float32
        )[:, tf.newaxis, tf.newaxis, :]
        enc_output = self.encoder(x, enc_padding_mask, False)

        output = tf.ones((batch_size, 1), dtype=tf.int32) * nl_w2i["<s>"]
        is_decoded = [False] * batch_size
        token_i = 0

        while True:

            _, combined_mask, dec_padding_mask = self._create_masks(
                x, output
            )
            dec_output, _ = self.decoder(
                output, enc_output, combined_mask, dec_padding_mask, False
            )
            predictions = self.final_layer(dec_output)[:, -1:, :]
            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

            for idx, pred_id in enumerate(predicted_id.numpy().squeeze().tolist()):

                if pred_id == nl_w2i["</s>"]:
                    is_decoded[idx] = True
                    continue

                preds[idx].append(nl_i2w[pred_id])
            
            if all(is_decoded) or token_i >= max_length:
                break

            token_i += 1
            output = tf.concat([output, predicted_id], axis=-1)
        
        return preds
    
    def multi_task_ss_loss(self, inp, pred):
        inp_padding_mask = tf.cast(
            tf.math.equal(inp, -1), tf.float32
        )[:, tf.newaxis, tf.newaxis, :]

        pred_ids = tf.cast(tf.argmax(pred, axis=-1), tf.int32)
        pred_padding_mask = tf.cast(
            tf.math.equal(pred_ids, -1), tf.float32
        )[:, tf.newaxis, tf.newaxis, :]

        inp_rep = self.inp_encoder(
            inp, inp_padding_mask, True
        )
        inp_rep = tf.reduce_mean(inp_rep, axis=1)
        pred_rep = self.pred_encoder(
            pred_ids, pred_padding_mask, True
        )
        pred_rep = tf.reduce_mean(pred_rep, axis=1)

        loss = 1 - tf.keras.losses.CosineSimilarity(axis=-1)(
            inp_rep, pred_rep
        )
        return tf.reduce_mean(loss)

    def get_loss(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, -1))

        real = tf.boolean_mask(real, mask)
        pred = tf.boolean_mask(pred, mask)

        loss_ = self.loss_object(labels=real, logits=pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)
    
    show = True
    def train_on_batch(self, inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]
        enc_padding_mask, combined_mask, dec_padding_mask = self._create_masks(
            inp, tar_inp
        )
        if self.show:
            print(inp[0, :])
            print(tar_inp[0, :])
            print("encoder padding mask", enc_padding_mask[0, :])
            print("combined mask", combined_mask[0, :])
            self.show = False
        
        with tf.GradientTape() as tape:
            predictions, _ = self(
                inp, 
                tar_inp, 
                enc_padding_mask, 
                combined_mask, 
                dec_padding_mask,
                training=True
            )
            loss = self.get_loss(tar_real, predictions)
            if self.crl:
                ss_loss = self.multi_task_ss_loss(inp, predictions)
                loss += 0.1 * ss_loss
        
        gradients = tape.gradient(loss, self.variables)
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        self.update_lr()
        self.optimizer.apply_gradients(zip(gradients, self.variables))
        self.global_step += 1
        return loss.numpy()
    
    def evaluate_on_batch(self, inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]
        enc_padding_mask, combined_mask, dec_padding_mask = self._create_masks(
            inp, tar_inp
        )
        predictions, _ = self(
            inp, 
            tar_inp, 
            enc_padding_mask, 
            combined_mask, 
            dec_padding_mask,
            training=False
        )
        loss = self.get_loss(tar_real, predictions)
        return loss.numpy()
        