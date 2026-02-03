import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformer_lens.hook_points import HookedRootModule
from datasets import Dataset, load_dataset
import tqdm


class ActivationsStore:
    def __init__(
        self,
        model: HookedRootModule,
        cfg: dict,
    ):
        self.model = model

        # Handle dataset path that might include config name
        dataset_path = cfg["dataset_path"]
        dataset_name = cfg.get("dataset_name", None)  # Optional subset/config name

        self.dataset = self._load_dataset(dataset_path, dataset_name)

        self.hook_point = cfg["hook_point"]
        self.context_size = min(cfg["seq_len"], model.cfg.n_ctx)
        self.model_batch_size = cfg["model_batch_size"]
        self.device = cfg["device"]
        self.num_batches_in_buffer = cfg["num_batches_in_buffer"]
        self.cfg = cfg
        self.tokenizer = model.tokenizer

        # Track documents processed (for reporting)
        self.documents_processed = 0

        # Skip documents if requested (for independent runs using different data)
        skip_documents = cfg.get("skip_documents", 0)
        if skip_documents > 0:
            print(f"   Skipping {skip_documents:,} documents...", flush=True)
            log_every = max(10000, skip_documents // 20)  # Log ~20 times, min every 10k
            for i in range(skip_documents):
                try:
                    next(self.dataset)
                except StopIteration:
                    print(f"   Dataset exhausted after skipping {i:,} documents", flush=True)
                    break
                if (i + 1) % log_every == 0:
                    print(f"   Skipped {i + 1:,} / {skip_documents:,} documents...", flush=True)
            print(f"   Done skipping {skip_documents:,} documents", flush=True)
            self.documents_processed = skip_documents

        self.tokens_column = self._get_tokens_column()

    def _load_dataset(self, dataset_path, dataset_name):
        """Load the streaming dataset."""
        if dataset_path in ["wikitext-2-raw-v1", "wikitext-2-v1", "wikitext-103-raw-v1", "wikitext-103-v1"]:
            ds = load_dataset("wikitext", dataset_path, split="train", streaming=True)
        elif dataset_name is not None:
            ds = load_dataset(dataset_path, name=dataset_name, split="train", streaming=True)
        else:
            ds = load_dataset(dataset_path, split="train", streaming=True)

        return iter(ds)

    def _get_tokens_column(self):
        sample = next(self.dataset)
        if "tokens" in sample:
            return "tokens"
        elif "input_ids" in sample:
            return "input_ids"
        elif "text" in sample:
            return "text"
        else:
            raise ValueError("Dataset must have a 'tokens', 'input_ids', or 'text' column.")

    def get_batch_tokens(self):
        all_tokens = []
        target_len = self.model_batch_size * self.context_size
        while len(all_tokens) < target_len:
            try:
                batch = next(self.dataset)
            except StopIteration:
                # Dataset exhausted, restart it
                print("   Dataset exhausted, restarting...")
                dataset_path = self.cfg["dataset_path"]
                dataset_name = self.cfg.get("dataset_name", None)
                self.dataset = self._load_dataset(dataset_path, dataset_name)
                batch = next(self.dataset)

            # Track documents processed
            self.documents_processed += 1

            if self.tokens_column == "text":
                # Tokenize text - this returns a GPU tensor
                tokens = self.model.to_tokens(batch["text"], truncate=True, move_to_device=True, prepend_bos=True).squeeze(0)
                all_tokens.extend(tokens.tolist())  # Convert to list for extending
            else:
                # Pre-tokenized data
                tokens = batch[self.tokens_column]
                if isinstance(tokens, torch.Tensor):
                    all_tokens.extend(tokens.tolist())
                else:
                    all_tokens.extend(tokens)

        # Create tensor directly on GPU
        token_tensor = torch.tensor(all_tokens[:target_len], dtype=torch.long, device=self.device)
        return token_tensor.view(self.model_batch_size, self.context_size)

    def get_activations(self, batch_tokens: torch.Tensor):
        with torch.no_grad():
            _, cache = self.model.run_with_cache(
                batch_tokens,
                names_filter=[self.hook_point],
                stop_at_layer=self.cfg["layer"] +1,
            )
            result = cache[self.hook_point]
            
            # Explicitly clear the cache to prevent reference leaks
            del cache
            return result

    def _fill_buffer(self):
        all_activations = []
        
        for i in range(self.num_batches_in_buffer):
            batch_tokens = self.get_batch_tokens()
            activations = self.get_activations(batch_tokens).reshape(-1, self.cfg["act_size"])
            all_activations.append(activations)
            
            # Clear intermediate variables
            del batch_tokens, activations
            
        result = torch.cat(all_activations, dim=0)
        del all_activations
        return result

    def _get_dataloader(self):
        return DataLoader(TensorDataset(self.activation_buffer), batch_size=self.cfg["batch_size"], shuffle=True)

    def next_batch(self):
        try:
            return next(self.dataloader_iter)[0]
        except (StopIteration, AttributeError):
            # Uncomment for memory debugging:
            # if torch.cuda.is_available():
            #     allocated_before = torch.cuda.memory_allocated() / 1e9
            #     reserved_before = torch.cuda.memory_reserved() / 1e9
            #     print(f"   Memory BEFORE cleanup - Allocated: {allocated_before:.1f}GB, Reserved: {reserved_before:.1f}GB")
            
            # Clear old buffer and associated objects
            if hasattr(self, 'dataloader_iter'):
                del self.dataloader_iter
            if hasattr(self, 'dataloader'):
                # Break the TensorDataset reference to the buffer (critical fix)
                if hasattr(self.dataloader, 'dataset') and hasattr(self.dataloader.dataset, 'tensors'):
                    self.dataloader.dataset.tensors = ()
                del self.dataloader  
            if hasattr(self, 'activation_buffer'):
                del self.activation_buffer
                
            ## Uncomment for memory debugging:
            #if torch.cuda.is_available():
            #    allocated_after = torch.cuda.memory_allocated() / 1e9
            #    reserved_after = torch.cuda.memory_reserved() / 1e9
            #    print(f"   Memory AFTER cleanup - Allocated: {allocated_after:.1f}GB, Reserved: {reserved_after:.1f}GB")
            #    #freed = allocated_before - allocated_after
            #    #print(f"   Freed: {freed:.1f}GB")
            #else:
            #    print("   No GPU available for memory debugging")
                
            # Create new buffer
            self.activation_buffer = self._fill_buffer()
            self.dataloader = self._get_dataloader()
            self.dataloader_iter = iter(self.dataloader)
            
            # Uncomment for memory debugging:
            # if torch.cuda.is_available():
            #     allocated_final = torch.cuda.memory_allocated() / 1e9
            #     reserved_final = torch.cuda.memory_reserved() / 1e9
            #     print(f"   Memory AFTER refill - Allocated: {allocated_final:.1f}GB, Reserved: {reserved_final:.1f}GB")
            
            return next(self.dataloader_iter)[0]

