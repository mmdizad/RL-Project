import sys
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

from babyai.rl.algos.base import BaseAlgo


def NS_single_step(pred, target):
    metric = nn.MSELoss()
    bs = pred.shape[0]
    return metric(pred, target.reshape(bs, -1))


class PPOAlgo(BaseAlgo):
    """The class for the Proximal Policy Optimization algorithm
    ([Schulman et al., 2015](https://arxiv.org/abs/1707.06347))."""

    def __init__(
        self,
        envs,
        acmodel,
        num_frames_per_proc=None,
        discount=0.99,
        lr=7e-4,
        beta1=0.9,
        beta2=0.999,
        gae_lambda=0.95,
        entropy_coef=0.01,
        value_loss_coef=0.5,
        max_grad_norm=0.5,
        recurrence=4,
        adam_eps=1e-5,
        clip_eps=0.2,
        epochs=4,
        batch_size=256,
        preprocess_obss=None,
        reshape_reward=None,
        aux_info=None,
        use_compositional_split=False,
        compositional_test_splits=None,
        device=None,
    ):
        num_frames_per_proc = num_frames_per_proc or 128

        super().__init__(
            envs,
            acmodel,
            num_frames_per_proc,
            discount,
            lr,
            gae_lambda,
            entropy_coef,
            value_loss_coef,
            max_grad_norm,
            recurrence,
            preprocess_obss,
            reshape_reward,
            aux_info,
            use_compositional_split=use_compositional_split,
            compositional_test_splits=compositional_test_splits,
            device=device,
        )

        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size

        assert self.batch_size % self.recurrence == 0

        self.optimizer = torch.optim.Adam(
            self.acmodel.parameters(), lr, (beta1, beta2), eps=adam_eps
        )
        self.batch_num = 0

    def calculate_contrastive_loss(self, similarity_matrix):
        v_2_t_loss = 0
        t_2_v_loss = 0
        transposed_similarity_matrix = similarity_matrix.T
        log_softmax_row_wise = F.log_softmax(similarity_matrix, dim=1)
        log_softmax_column_wise = F.log_softmax(transposed_similarity_matrix, dim=1)

        v_2_t_loss = torch.trace(log_softmax_row_wise)

        v_2_t_loss = v_2_t_loss / -log_softmax_row_wise.shape[0]

        t_2_v_loss = torch.trace(log_softmax_column_wise)

        t_2_v_loss = t_2_v_loss / -log_softmax_column_wise.shape[0]

        total_loss = v_2_t_loss + t_2_v_loss

        return total_loss

    def Attention_Over_Similarity_Vector(self, vector, temp=1):
        vector = torch.tensor(vector, dtype=float)
        vector = vector / temp
        attn_weights = F.softmax(vector, dim=0)
        weighted_sum = torch.dot(attn_weights, vector)
        return weighted_sum

    def Attention_Over_Similarity_Matrix(self, matrix, temp=1):
        transposed_matrix = matrix.T
        weighted_row = []
        weighted_column = []
        for i in range(matrix.shape[0]):
            weighted_row.append(self.Attention_Over_Similarity_Vector(matrix[i], temp))
        for i in range(transposed_matrix.shape[0]):
            weighted_column.append(
                self.Attention_Over_Similarity_Vector(transposed_matrix[i], temp)
            )

        weighted_row = torch.tensor(weighted_row).reshape(-1)
        weighted_column = torch.tensor(weighted_column).reshape(-1)

        row_score = self.Attention_Over_Similarity_Vector(weighted_row, temp)
        column_score = self.Attention_Over_Similarity_Vector(weighted_column, temp)
        total_score = (row_score + column_score) / 2
        return total_score

    def update_parameters(self):
        # Collect experiences

        exps, logs = self.collect_experiences()
        """
        exps is a DictList with the following keys ['obs', 'memory', 'mask', 'action', 'value', 'reward',
         'advantage', 'returnn', 'log_prob'] and ['collected_info', 'extra_predictions'] if we use aux_info
        exps.obs is a DictList with the following keys ['image', 'instr']
        exps.obj.image is a (n_procs * n_frames_per_proc) x image_size 4D tensor
        exps.obs.instr is a (n_procs * n_frames_per_proc) x (max number of words in an instruction) 2D tensor
        exps.memory is a (n_procs * n_frames_per_proc) x (memory_size = 2*image_embedding_size) 2D tensor
        exps.mask is (n_procs * n_frames_per_proc) x 1 2D tensor
        if we use aux_info: exps.collected_info and exps.extra_predictions are DictLists with keys
        being the added information. They are either (n_procs * n_frames_per_proc) 1D tensors or
        (n_procs * n_frames_per_proc) x k 2D tensors where k is the number of classes for multiclass classification
        """
        for _ in range(self.epochs):
            # Initialize log values

            log_entropies = []
            log_values = []
            log_policy_losses = []
            log_value_losses = []
            log_grad_norms = []

            log_losses = []
            
            self.acmodel.reset_hiddens()
            
            ######################################################################
            # X-CLIP loss
            similarity_mx = torch.zeros((self.num_procs, self.num_procs))
            video_matrix = torch.zeros(
                (self.num_procs, self.num_frames_per_proc, self.acmodel.instr_dim)
            )
            text_matrix = torch.zeros(
                (self.num_procs, exps.obs.instr.shape[1], self.acmodel.instr_dim)
            )
            video_global_embeddings = torch.zeros(
                (self.num_procs, self.acmodel.instr_dim)
            )
            text_global_embeddings = torch.zeros(
                (self.num_procs, self.acmodel.instr_dim)
            )
            # calculate each env(video) and instruction features
            env_start_index = self._get_env_starting_indexes()
            memory = exps.memory[env_start_index]
            for i in range(self.num_frames_per_proc):
                sb = exps[env_start_index + i]
                model_results = self.acmodel(sb.obs, memory * sb.mask, mask=sb.mask)
                frame_embeddings = model_results["frame_embedding"]
                if i == 0:
                    token_embedding = model_results["token_embedding"]
                    text_global_embeddings = model_results["instr_embedding"]
                    text_matrix = token_embedding
                video_matrix[:, i, :] = frame_embeddings
            for i in range(self.num_procs):
                video_global_embeddings[i] = torch.mean(video_matrix[i], dim=0)
                # text_global_embeddings[i] = torch.mean(text_matrix[i], dim=0)
            for i in range(self.num_procs):
                for j in range(self.num_procs):
                    frame_token_similarity = self.Attention_Over_Similarity_Matrix(
                        torch.matmul(video_matrix[i], text_matrix[j].T)
                    )
                    text_frame_similarity = self.Attention_Over_Similarity_Vector(
                        torch.matmul(video_matrix[i], text_global_embeddings[j])
                    )
                    video_token_similarity = self.Attention_Over_Similarity_Vector(
                        torch.matmul(text_matrix[j], video_global_embeddings[i]).T
                    )
                    video_text_similarity = torch.matmul(
                        video_global_embeddings[i].T, text_global_embeddings[j]
                    )
                    similarity_mx[i][j] = (
                        frame_token_similarity
                        + text_frame_similarity
                        + video_token_similarity
                        + video_text_similarity
                    ) / 4
            # calculate loss function
            x_clip_loss = self.calculate_contrastive_loss(similarity_mx)
            ######################################################################

            """
            For each epoch, we create int(total_frames / batch_size + 1) batches, each of size batch_size (except
            maybe the last one). Each batch is divided into sub-batches of size recurrence (frames are contiguous in
            a sub-batch), but the position of each sub-batch in a batch and the position of each batch in the whole
            list of frames is random thanks to self._get_batches_starting_indexes().
            """
            for inds in self._get_batches_starting_indexes():
                # inds is a numpy array of indices that correspond to the beginning of a sub-batch
                # there are as many inds as there are batches
                # Initialize batch values

                batch_entropy = 0
                batch_value = 0
                batch_policy_loss = 0
                batch_value_loss = 0
                batch_loss = 0

                # update batch loss with x-clip loss
                batch_loss += x_clip_loss

                # Initialize memory

                memory = exps.memory[inds]

                self.acmodel.reset_hiddens()

                for i in range(self.recurrence):
                    # Create a sub-batch of experience
                    sb = exps[inds + i]

                    # Compute loss

                    model_results = self.acmodel(sb.obs, memory * sb.mask, mask=sb.mask)
                    dist = model_results["dist"]
                    value = model_results["value"]
                    memory = model_results["memory"]
                    extra_predictions = model_results["extra_predictions"]

                    entropy = dist.entropy().mean()

                    ratio = torch.exp(dist.log_prob(sb.action) - sb.log_prob)
                    surr1 = ratio * sb.advantage
                    surr2 = (
                        torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
                        * sb.advantage
                    )
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_clipped = sb.value + torch.clamp(
                        value - sb.value, -self.clip_eps, self.clip_eps
                    )
                    surr1 = (value - sb.returnn).pow(2)
                    surr2 = (value_clipped - sb.returnn).pow(2)
                    value_loss = torch.max(surr1, surr2).mean()

                    loss = (
                        policy_loss
                        - self.entropy_coef * entropy
                        + self.value_loss_coef * value_loss
                    )

                    # Update batch values

                    batch_entropy += entropy.item()
                    batch_value += value.mean().item()
                    batch_policy_loss += policy_loss.item()
                    batch_value_loss += value_loss.item()
                    batch_loss += loss

                    # Update memories for next epoch

                    if i < self.recurrence - 1:
                        exps.memory[inds + i + 1] = memory.detach()

                # Update batch values

                batch_entropy /= self.recurrence
                batch_value /= self.recurrence
                batch_policy_loss /= self.recurrence
                batch_value_loss /= self.recurrence
                batch_loss /= self.recurrence

                # Update actor-critic

                self.optimizer.zero_grad()
                # because we have a loop on variables in NPS and shared layers occur, set retain_graph to True
                batch_loss.backward()
                grad_norm = (
                    sum(
                        p.grad.data.norm(2) ** 2
                        for p in self.acmodel.parameters()
                        if p.grad is not None
                    )
                    ** 0.5
                )
                torch.nn.utils.clip_grad_norm_(
                    self.acmodel.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                # Update log values

                log_entropies.append(batch_entropy)
                log_values.append(batch_value)
                log_policy_losses.append(batch_policy_loss)
                log_value_losses.append(batch_value_loss)
                log_grad_norms.append(grad_norm.item())
                log_losses.append(batch_loss.item())

        # Log some values

        logs["entropy"] = numpy.mean(log_entropies)
        logs["value"] = numpy.mean(log_values)
        logs["policy_loss"] = numpy.mean(log_policy_losses)
        logs["value_loss"] = numpy.mean(log_value_losses)
        logs["grad_norm"] = numpy.mean(log_grad_norms)
        logs["loss"] = numpy.mean(log_losses)

        return logs

    def _get_batches_starting_indexes(self):
        """Gives, for each batch, the indexes of the observations given to
        the model and the experiences used to compute the loss at first.
        Returns
        -------
        batches_starting_indexes : list of list of int
            the indexes of the experiences to be used at first for each batch

        """

        indexes = numpy.arange(0, self.num_frames, self.recurrence)
        indexes = numpy.random.permutation(indexes)

        num_indexes = self.batch_size // self.recurrence
        batches_starting_indexes = [
            indexes[i : i + num_indexes] for i in range(0, len(indexes), num_indexes)
        ]

        return batches_starting_indexes

    def _get_env_starting_indexes(self):
        env_starting_indexes = [
            self.num_frames_per_proc * i for i in range(self.num_procs)
        ]
        return numpy.array(env_starting_indexes)
