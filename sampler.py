import mlx.core as mx
import copy


def make_sampler(temp: float, top_p: float | None, top_k: int | None):
    def sample(logprobs: mx.array):
        if temp == 0:
            return mx.argmax(logprobs, axis=-1)
        if(top_p is not None):
            sorted_idx = mx.argsort(logprobs, axis=-1)       
            sorted_idx = sorted_idx[:, ::-1] 
            sorted_logprobs = mx.take_along_axis(logprobs, sorted_idx, axis=-1)
            sorted_exp = mx.exp(sorted_logprobs)
            cumsum_exp = mx.cumsum(sorted_exp, axis=-1)
            condition = (cumsum_exp - sorted_exp) >= top_p
            mask = mx.zeros_like(condition)
            mask = mx.put_along_axis(mask, sorted_idx, condition, axis=-1)
            logprobs = mx.where(mask, float("-inf"), logprobs)

        if(top_k is not None):
            threshold = mx.sort(logprobs, axis=-1)[:, -top_k]
            condition = logprobs < threshold[:, None]
            logprobs = mx.where(condition,  float("-inf"), logprobs)
        logprobs_scaled = logprobs / temp

        return mx.random.categorical(logprobs_scaled, axis=-1)

    return sample
