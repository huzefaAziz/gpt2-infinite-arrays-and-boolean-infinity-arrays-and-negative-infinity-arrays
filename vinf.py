"""
Fast infinite GPT2 array using infinite_arrays library.
Uses GPT2 model to generate infinite sequences of tokens/embeddings.
"""

import sys
import os
import numpy as np
from typing import Tuple, Union, Iterator, Any, Optional, List

# Try to import torch, with helpful error if not available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: torch library not found. Install with: pip install torch")

# Add the infinite_arrays directory to the path so we can import it without installing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'infinite_arrays'))

from infinite_arrays import InfiniteDiagonal, InfiniteArray
from infinite_arrays.infinity import Infinity
from infinite_arrays._utils import get_infinity

# Try to import transformers, with helpful error if not available
try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers library not found. Install with: pip install transformers torch")

_INF = get_infinity()
globals()['∞'] = _INF


class GPT2Sequence:
    """
    GPT2-based sequence generator that generates tokens on-demand.
    Caches generated tokens for fast access.
    """
    
    def __init__(self, model_name: str = "gpt2", prompt: str = "", device: Optional[str] = None):
        """
        Initialize GPT2 model and tokenizer.
        
        Args:
            model_name: HuggingFace model name (default: "gpt2")
            prompt: Initial prompt for generation (default: "")
            device: Device to use ("cuda", "cpu", or None for auto)
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is required. Install with: pip install transformers torch")
        if not TORCH_AVAILABLE:
            raise ImportError("torch library is required. Install with: pip install torch")
        
        self.model_name = model_name
        self.prompt = prompt
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()  # Set to evaluation mode
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        # Token cache for fast access
        self._token_cache: List[int] = []
        self._text_cache: List[str] = []
        self._embedding_cache: dict = {}
        
        # Initialize with prompt if provided
        if prompt:
            self._initialize_from_prompt(prompt)
    
    def _initialize_from_prompt(self, prompt: str):
        """Initialize token cache from prompt."""
        tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        self._token_cache = tokens[0].tolist()
        self._text_cache = [self.tokenizer.decode([t]) for t in self._token_cache]
    
    def _generate_token(self, max_new_tokens: int = 1):
        """Generate next token(s) using GPT2."""
        if not self._token_cache:
            # Start with a token if cache is empty
            input_ids = torch.tensor([[self.tokenizer.bos_token_id or self.tokenizer.eos_token_id]], 
                                    device=self.device)
        else:
            input_ids = torch.tensor([self._token_cache], device=self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                pad_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1
            )
        
        # Extract new tokens
        new_tokens = outputs[0][len(input_ids[0]):].tolist()
        self._token_cache.extend(new_tokens)
        self._text_cache.extend([self.tokenizer.decode([t]) for t in new_tokens])
        
        return new_tokens
    
    def __getitem__(self, index: int) -> Union[int, str, np.ndarray]:
        """
        Get token at index.
        Generates tokens on-demand if not in cache.
        
        Args:
            index: Token index (0-based)
            
        Returns:
            Token ID, text, or embedding based on access mode
        """
        # Generate tokens up to index if needed
        while len(self._token_cache) <= index:
            self._generate_token()
        
        return self._token_cache[index]
    
    def get_token_text(self, index: int) -> str:
        """Get text representation of token at index."""
        while len(self._text_cache) <= index:
            self._generate_token()
        return self._text_cache[index]
    
    def get_embedding(self, index: int) -> np.ndarray:
        """Get embedding vector for token at index."""
        if index not in self._embedding_cache:
            token_id = self[index]
            with torch.no_grad():
                # Get embedding from model
                token_tensor = torch.tensor([[token_id]], device=self.device)
                embeddings = self.model.transformer.wte(token_tensor)
                self._embedding_cache[index] = embeddings[0][0].cpu().numpy()
        return self._embedding_cache[index]
    
    def __iter__(self) -> Iterator:
        """Iterate over tokens."""
        i = 0
        while True:
            yield self[i]
            i += 1
    
    def generate_text(self, length: int = 10) -> str:
        """Generate text of specified length."""
        while len(self._text_cache) < length:
            self._generate_token()
        return "".join(self._text_cache[:length])


class gpt2(InfiniteDiagonal):
    """
    GPT2-based infinite diagonal array.
    Each diagonal element represents a GPT2-generated token or embedding.
    Inherits from InfiniteDiagonal to create an infinite matrix.
    """
    
    def __init__(self, 
                 model_name: str = "gpt2",
                 prompt: str = "",
                 mode: str = "token",
                 device: Optional[str] = None,
                 dtype=None):
        """
        Initialize GPT2 infinite diagonal array.
        
        Args:
            model_name: HuggingFace GPT2 model name (default: "gpt2")
            prompt: Initial prompt for generation
            mode: "token" (token IDs), "text" (text strings), or "embedding" (embedding vectors)
            device: Device to use ("cuda", "cpu", or None for auto)
            dtype: Data type for array elements
        """
        # Initialize GPT2 sequence generator
        self._gpt2_seq = GPT2Sequence(model_name, prompt, device)
        self.mode = mode
        
        # Set dtype based on mode
        if dtype is None:
            if mode == "token":
                dtype = np.int64
            elif mode == "embedding":
                dtype = np.float32
            else:
                dtype = object  # For text strings
        
        # Initialize InfiniteDiagonal with the GPT2 sequence
        super().__init__(self._gpt2_seq, dtype=dtype)
        
        # Store mode for custom access
        self._mode = mode
    
    def __getitem__(self, key: Union[int, Tuple[int, int]]) -> Union[Any, 'gpt2']:
        """
        Get element from the infinite diagonal matrix.
        
        Args:
            key: Integer (diagonal index) or tuple (row, col)
            
        Returns:
            GPT2 token, text, or embedding based on mode
        """
        if isinstance(key, slice):
            return self
        
        # Get the diagonal index
        if isinstance(key, int):
            diag_idx = key
        elif isinstance(key, tuple) and len(key) == 2:
            row, col = key
            if row != col:
                return 0.0 if self._mode != "text" else ""
            diag_idx = row
        else:
            return super().__getitem__(key)
        
        # Get value based on mode
        if self._mode == "token":
            return self._gpt2_seq[diag_idx]
        elif self._mode == "text":
            return self._gpt2_seq.get_token_text(diag_idx)
        elif self._mode == "embedding":
            return self._gpt2_seq.get_embedding(diag_idx)
        else:
            return self._gpt2_seq[diag_idx]
    
    def generate(self, length: int = 10) -> str:
        """Generate text of specified length."""
        return self._gpt2_seq.generate_text(length)
    
    def get_model(self) -> GPT2LMHeadModel:
        """Get the underlying GPT2 model."""
        return self._gpt2_seq.model
    
    def get_tokenizer(self) -> GPT2Tokenizer:
        """Get the underlying tokenizer."""
        return self._gpt2_seq.tokenizer


class VInfiniteArray(gpt2):
    """
    Vector-based infinite array using GPT2 embeddings.
    Each element is an embedding vector from GPT2.
    Inherits from gpt2 (which already inherits from InfiniteDiagonal -> InfiniteArray).
    Provides 1D vector-style access while maintaining 2D diagonal matrix capabilities.
    """
    
    def __init__(self,
                 model_name: str = "gpt2",
                 prompt: str = "",
                 device: Optional[str] = None,
                 dtype=None):
        """
        Initialize GPT2-based vector infinite array.
        
        Args:
            model_name: HuggingFace GPT2 model name
            prompt: Initial prompt for generation
            device: Device to use
            dtype: Data type (default: np.float32 for embeddings)
        """
        # Initialize gpt2 with embedding mode
        # This will call InfiniteDiagonal.__init__ -> InfiniteArray.__init__ through the inheritance chain
        super().__init__(model_name=model_name, prompt=prompt, mode="embedding", device=device, dtype=dtype or np.float32)
        self._shape_1d = (_INF,)
        self._embedding_dim = None
    
    def __getitem__(self, key: Union[int, slice, Tuple]) -> np.ndarray:
        """
        Get embedding vector at index.
        Supports both 1D indexing (vector) and 2D indexing (diagonal matrix).
        """
        if isinstance(key, slice):
            return self
        
        # For 1D indexing, get embedding directly
        if isinstance(key, int):
            return self._gpt2_seq.get_embedding(key)
        # For 2D indexing (from gpt2/diagonal), use parent method
        elif isinstance(key, tuple) and len(key) == 2:
            return gpt2.__getitem__(self, key)
        else:
            return self._gpt2_seq.get_embedding(key)
    
    def __iter__(self) -> Iterator:
        """Iterate over embedding vectors."""
        i = 0
        while True:
            yield self[i]
            i += 1
    
    def shape(self) -> Tuple:
        """Return shape: (∞,) for 1D vector or (∞, ∞) for diagonal matrix."""
        # Can return 1D shape for vector access, or 2D for diagonal matrix access
        return self._shape_1d  # Default to 1D vector shape


if __name__ == "__main__":
    print("Initializing GPT2 infinite array...")
    print("Note: First run will download the model (~500MB)")
    print("=" * 60)
    
    # Create GPT2 infinite diagonal array in token mode
    print("\n1. Creating GPT2 infinite diagonal (token mode):")
    gpt2_tokens = gpt2(model_name="gpt2", prompt="The future of AI is", mode="token")
    
    print("\nFirst 10 diagonal elements (token IDs):")
    for i in range(10):
        print(f"  [{i},{i}] = {gpt2_tokens[i, i]}")
    
    # Create GPT2 infinite diagonal array in text mode
    print("\n2. Creating GPT2 infinite diagonal (text mode):")
    gpt2_text = gpt2(model_name="gpt2", prompt="The future of AI is", mode="text")
    
    print("\nFirst 10 diagonal elements (text):")
    for i in range(10):
        token_text = gpt2_text[i, i]
        print(f"  [{i},{i}] = '{token_text}'")
    
    print("\n3. Generated text sequence:")
    generated = gpt2_text.generate(30)
    print(f"  '{generated}'")
    
    print("\n4. Creating VInfiniteArray (embedding vectors):")
    vinf = VInfiniteArray(model_name="gpt2", prompt="Hello world")
    
    print("\nFirst 3 embedding vectors:")
    for i in range(3):
        emb = vinf[i]
        print(f"  Vector {i}: shape={emb.shape}, first 5 values={emb[:5]}")

