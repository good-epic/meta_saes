import torch
import tqdm
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader, TensorDataset
from transformer_lens.hook_points import HookedRootModule


class ActivationsStore:
    def __init__(
        self,
        model: HookedRootModule,
        cfg: dict,
    ):
        self.model = model
        self.dataset = iter(
            load_dataset(
                cfg.dataset_path, split="train", streaming=True, trust_remote_code=True
            )
        )
        self.hook_point = cfg.hook_point
        self.context_size = min(cfg.seq_len, model.cfg.n_ctx)
        self.model_batch_size = cfg.model_batch_size
        self.device = cfg.device
        self.num_batches_in_buffer = cfg.num_batches_in_buffer
        self.tokens_column = self._get_tokens_column()
        self.cfg = cfg
        self.tokenizer = model.tokenizer

    def _get_tokens_column(self):
        sample = next(self.dataset)
        if "tokens" in sample:
            return "tokens"
        elif "input_ids" in sample:
            return "input_ids"
        elif "text" in sample:
            return "text"
        else:
            raise ValueError(
                "Dataset must have a 'tokens', 'input_ids', or 'text' column."
            )

    def get_batch_tokens(self):
        all_tokens = []
        while len(all_tokens) < self.model_batch_size * self.context_size:
            batch = next(self.dataset)
            if self.tokens_column == "text":
                tokens = self.model.to_tokens(
                    batch["text"], truncate=True, move_to_device=True, prepend_bos=True
                ).squeeze(0)
            else:
                tokens = batch[self.tokens_column]
            all_tokens.extend(tokens)
        token_tensor = torch.tensor(all_tokens, dtype=torch.long, device=self.device)[
            : self.model_batch_size * self.context_size
        ]
        return token_tensor.view(self.model_batch_size, self.context_size)

    def get_activations(self, batch_tokens: torch.Tensor):
        with torch.no_grad():
            _, cache = self.model.run_with_cache(
                batch_tokens,
                names_filter=[self.hook_point],
                stop_at_layer=self.cfg.layer + 1,
            )
        return cache[self.hook_point]

    def _fill_buffer(self):
        all_activations = []
        for _ in range(self.num_batches_in_buffer):
            batch_tokens = self.get_batch_tokens()
            activations = self.get_activations(batch_tokens).reshape(
                -1, self.cfg.act_size
            )
            all_activations.append(activations)
        return torch.cat(all_activations, dim=0)

    def _get_dataloader(self):
        return DataLoader(
            TensorDataset(self.activation_buffer),
            batch_size=self.cfg.batch_size,
            shuffle=True,
        )

    def next_batch(self):
        try:
            return next(self.dataloader_iter)[0]
        except (StopIteration, AttributeError):
            self.activation_buffer = self._fill_buffer()
            self.dataloader = self._get_dataloader()
            self.dataloader_iter = iter(self.dataloader)
            return next(self.dataloader_iter)[0]

    def get_complete_tokenized_dataset(self, add_bos=True):
        all_tokens = []
        dataset_iterator = iter(
            load_dataset(self.cfg.dataset_path, split="train", streaming=True)
        )

        for batch in tqdm.tqdm(dataset_iterator, desc="Tokenizing dataset"):
            if self.tokens_column == "text":
                tokens = self.model.to_tokens(
                    batch["text"],
                    truncate=False,
                    move_to_device=False,
                    prepend_bos=False,
                )
            else:
                tokens = torch.tensor(batch[self.tokens_column], dtype=torch.long)

            # Ensure all tokens are 1D
            if tokens.dim() > 1:
                tokens = tokens.squeeze()

            all_tokens.append(tokens)

        # Combine all tokens
        combined_tokens = torch.cat(all_tokens)

        # Reshape to -1 x seq_len
        seq_len = self.cfg.seq_len
        num_sequences = combined_tokens.size(0) // seq_len
        reshaped_tokens = combined_tokens[: num_sequences * seq_len].view(-1, seq_len)

        if add_bos:
            reshaped_tokens[:, 0] = self.model.tokenizer.bos_token_id
            final_tokens = reshaped_tokens
        else:
            final_tokens = reshaped_tokens

        return final_tokens
