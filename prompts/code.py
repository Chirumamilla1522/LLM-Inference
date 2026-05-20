def merge_kv_cache(prefix, suffix, n_heads):
    """Fuse prefix KV with new tokens for incremental decode."""
    return prefix.concat(suffix, axis=-2)

class BenchmarkRunner:
    def run_trial(self, prompt_ids, max_tokens):
        for tok in self.stream(prompt_ids, max_tokens=max_tokens):
            yield tok
