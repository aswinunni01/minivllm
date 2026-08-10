from abc import ABC, abstractmethod
from typing import Optional

import mlx.core as mx

from .attention import causal_mask


class TinyKvCache(ABC):
    @abstractmethod
    def update_and_fetch(
        self,
        key: mx.array,
        value: mx.array,
        mask_length: int | None = None,
        mask: mx.array | str | None = None,
    ) -> tuple[mx.array, mx.array, int, Optional[mx.array]]:
        """
        Update the key-value cache and fetch the updated key-value cache.

        Args:
            key: The key to update the cache with.
            value: The value to update the cache with.
            mask_length: The length of the mask (only used in batching mode)
            mask: The mask to use (only used in batching mode)

        Returns:
            The updated keys, updated values, sequence length, and mask. On
            On Week 2 Day 1, the mask is passed through unchanged. Week 3 Day 1
            uses the sequence length and mask to construct a dense batch.
        """

    def release(self):
        pass

    def materialize(self):
        """Evaluate owned K/V storage without changing its logical layout."""
        pass

    def update_and_fetch_paged(
        self,
        key: mx.array,
        value: mx.array,
        mask_length: int | None = None,
        mask: mx.array | str | None = None,
    ) -> "PagedKvMetadata":
        # Dense caches intentionally do not provide paged metadata.
        raise NotImplementedError("This KV cache does not support paged attention")

    def rewind(self, n: int):
        # Speculative decoding rewinds; caches without suffix state decline.
        raise NotImplementedError("This KV cache does not support rewind")


class BatchingKvCache(TinyKvCache):
    def __init__(self, max_active_requests: int, max_seq_len: int | None = None):
        self.max_active_requests = max_active_requests
        self.max_seq_len = max_seq_len
        # One request cache per decode slot; None means the slot is idle.
        self.kv_caches: list[TinyKvCache] = [None] * max_active_requests
        self.HD = None
        self.last_batch_bytes = 0
        self.staging_copy_bytes = 0

    def update_and_fetch(
        self,
        keys: mx.array,
        values: mx.array,
        mask_length: int | None = None,
        mask: mx.array | str | None = None,
    ) -> tuple[mx.array, mx.array, int, Optional[mx.array]]:
        B, H, S, D = keys.shape
        assert keys.shape == values.shape
        if self.max_seq_len is not None:
            assert S <= self.max_seq_len
        if self.HD is None:
            self.HD = (H, D)
        else:
            assert self.HD == (H, D), f"expect {self.HD} but got {H, D}"
        assert B == self.max_active_requests
        # Step 1: append each active row into its own request cache.
        data = []
        for b in range(B):
            if self.kv_caches[b] is None:
                data.append(None)
                continue
            key, value = keys[b : b + 1], values[b : b + 1]
            new_key, new_value, seq_len, mask_i = self.kv_caches[b].update_and_fetch(
                key, value
            )
            data.append((new_key[0], new_value[0], seq_len, mask_i))

        # Step 2: the dense batch length is the longest active sequence.
        def get_seq_len(data):
            if data is None:
                return 0
            _, _, seq_len, _ = data
            return seq_len

        seq_len = max(map(get_seq_len, data))
        # Step 3: rebuild one dense batch tensor, right-aligning each request;
        # leading positions stay zero and fully masked. True paged attention
        # replaces this with block_table/context_lens metadata later in Week 3.
        keys = mx.zeros((self.max_active_requests, H, seq_len, D), dtype=key.dtype)
        values = mx.zeros((self.max_active_requests, H, seq_len, D), dtype=value.dtype)
        masks = mx.full(
            (self.max_active_requests, mask_length, seq_len), -mx.inf, dtype=key.dtype
        )
        for b in range(B):
            if data[b] is None:
                continue
            key, value, S_i, mask_i = data[b]
            self.staging_copy_bytes += key.nbytes + value.nbytes
            keys[b, :, seq_len - S_i : seq_len, :] = key
            values[b, :, seq_len - S_i : seq_len, :] = value
            if mask_i is None or mask_i == "causal":
                masks[b, :, seq_len - S_i : seq_len] = causal_mask(
                    mask_length, S_i, dtype=key.dtype
                )
            elif isinstance(mask_i, mx.array):
                masks[b, :, seq_len - S_i : seq_len] = mask_i
            else:
                raise NotImplementedError
        self.last_batch_bytes = keys.nbytes + values.nbytes
        return keys, values, None, masks.reshape(B, 1, mask_length, seq_len)

    def update_and_fetch_paged(
        self,
        keys: mx.array,
        values: mx.array,
        mask_length: int | None = None,
        mask: mx.array | str | None = None,
    ) -> "PagedKvMetadata":
        from .paged_kv_cache import PagedKvMetadata, TinyKvPagedCache

        if len(keys.shape) != 4 or len(values.shape) != 4:
            raise ValueError("Batched K/V chunks must be 4D [B, H, S, D]")
        if keys.shape != values.shape:
            raise ValueError("Batched K/V chunks must have the same shape")
        B, H, S, D = keys.shape
        if B != self.max_active_requests:
            raise ValueError(f"Expected batch size {self.max_active_requests}, got {B}")
        if self.HD is not None and self.HD != (H, D):
            raise ValueError(f"expect {self.HD} but got {H, D}")

        # Validate the complete active set before any request or allocator is
        # mutated, so a mixed pool fails before row zero appends.
        pool = None
        active_caches = []
        for b in range(B):
            cache = self.kv_caches[b]
            if cache is None:
                continue
            if not isinstance(cache, TinyKvPagedCache):
                raise ValueError("BatchingKvCache contains a non-paged request cache")
            if pool is None:
                pool = cache.pool
            elif pool is not cache.pool:
                raise ValueError("Paged batch caches must share one page pool")
            if self.max_seq_len is not None and cache.offset + S > self.max_seq_len:
                raise ValueError("Paged batch append exceeds max_seq_len")
            cache.validate_append(keys[b : b + 1], values[b : b + 1])
            active_caches.append((b, cache))

        if pool is None:
            raise ValueError("Cannot build paged metadata without active requests")

        # Snapshot allocator/request state so a failed append can roll back.
        pool_state = pool._snapshot_state()
        cache_states = [(cache, cache._snapshot_state()) for _, cache in active_caches]
        old_hd = self.HD
        context_lens = [0] * B
        max_pages = 0
        try:
            for b, cache in active_caches:
                cache.update_and_fetch_paged(
                    keys[b : b + 1],
                    values[b : b + 1],
                    mask_length=mask_length,
                    mask=mask,
                )
                context_lens[b] = cache.offset
                max_pages = max(max_pages, cache.num_pages)
            self.HD = (H, D)
        except Exception:
            pool._restore_state(pool_state)
            for cache, state in cache_states:
                cache._restore_state(state)
            self.HD = old_hd
            raise

        self.last_batch_bytes = 0

        rows = []
        for cache in self.kv_caches:
            if cache is None:
                rows.append([-1] * max_pages)
            else:
                rows.append(cache.page_ids + [-1] * (max_pages - cache.num_pages))

        return PagedKvMetadata(
            key_pages=pool.key_pages,
            value_pages=pool.value_pages,
            block_table=mx.array(rows, dtype=mx.int32),
            context_lens=mx.array(context_lens, dtype=mx.int32),
            page_size=pool.page_size,
            mask=mask,
        )

    def add_request(self, prefilled: TinyKvCache, id: int):
        if id >= self.max_active_requests:
            raise ValueError(f"Request id {id} is out of range")
        if isinstance(prefilled, TinyKvFullCache) and prefilled.key_values is not None:
            keys, _ = prefilled.key_values
            B, H, _, D = keys.shape
            assert B == 1
            if self.HD is None:
                self.HD = (H, D)
            else:
                assert self.HD == (H, D)
        self.kv_caches[id] = prefilled

    def remove_request(self, id: int):
        if self.kv_caches[id] is None:
            raise ValueError(f"Request id {id} is not in the cache")
        self.kv_caches[id].release()
        self.kv_caches[id] = None


class TinyKvFullCache(TinyKvCache):
    def __init__(self):
        self.key_values = None
        self.offset = 0

    def update_and_fetch(
        self,
        key: mx.array,
        value: mx.array,
        mask_length: int | None = None,
        mask: mx.array | str | None = None,
    ) -> tuple[mx.array, mx.array, int, Optional[mx.array]]:
        if(self.key_values is None):
            self.key_values = (key, value)
        else :
            cached_keys, cached_values = self.key_values
            self.key_values = (mx.concat([cached_keys, key], axis=2), mx.concat([cached_values, value], axis=2))

        self.offset += key.shape[2]
        keys, values = self.key_values
        return (keys, values, self.offset, mask)



    def materialize(self):
        if self.key_values is not None:
            mx.eval(*self.key_values)

    def rewind(self, n: int):
        # Drop the newest n logical tokens (speculative decoding rejects).
        self.offset -= n
        self.key_values = (
            self.key_values[0][:, :, : self.offset],
            self.key_values[1][:, :, : self.offset],
        )
