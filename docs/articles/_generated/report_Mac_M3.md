# Benchmark tables

## Article 0: Local LLMs on Apple Silicon
| Model | Run | Memory | TTFT | tok/s | Status |
|-------|-----|--------|------|-------|--------|
| llama3-8b | demo_fp16 | 16.33 | 2651 | 5.3 | ok |

## Article 1: Weight quantization
| Model | Run | Memory | TTFT | tok/s | Status |
|-------|-----|--------|------|-------|--------|
| deepseek-r1-llama-8b | fp16 | 16.33 | 2633 | 5.8 | ok |
| deepseek-r1-llama-8b | w2 | — | — | — | skipped |
| deepseek-r1-llama-8b | w4 | 5.06 | 2477 | 20.6 | ok |
| deepseek-r1-llama-8b | w8 | 8.96 | 2591 | 11.2 | ok |
| deepseek-r1-qwen-7b | fp16 | 15.49 | 2430 | 6.2 | ok |
| deepseek-r1-qwen-7b | w2 | — | — | — | skipped |
| deepseek-r1-qwen-7b | w4 | 4.72 | 2317 | 21.8 | ok |
| deepseek-r1-qwen-7b | w8 | 8.52 | 2417 | 12.0 | ok |
| gemma-2-2b | fp16 | 3.32 | 823 | 30.0 | ok |
| gemma-2-2b | w2 | — | — | — | skipped |
| gemma-2-2b | w4 | 2.12 | 800 | 54.4 | ok |
| gemma-2-2b | w8 | 3.32 | 825 | 30.3 | ok |
| gemma-9b | fp16 | 10.51 | 3276 | 8.8 | ok |
| gemma-9b | w2 | — | — | — | skipped |
| gemma-9b | w4 | 5.88 | 3168 | 15.9 | ok |
| gemma-9b | w8 | 10.51 | 3419 | 8.9 | ok |
| llama-3.2-1b | fp16 | 2.71 | 372 | 32.8 | ok |
| llama-3.2-1b | w2 | — | — | — | skipped |
| llama-3.2-1b | w4 | 1.24 | 351 | 102.9 | ok |
| llama-3.2-1b | w8 | 1.75 | 359 | 57.3 | ok |
| llama-3.2-3b | fp16 | 6.73 | 1070 | 13.6 | ok |
| llama-3.2-3b | w2 | — | — | — | skipped |
| llama-3.2-3b | w4 | 2.34 | 1019 | 45.8 | ok |
| llama-3.2-3b | w8 | 3.86 | 1056 | 25.4 | ok |
| llama3-8b | fp16 | 16.33 | 2637 | 5.8 | ok |
| llama3-8b | w2 | 3.11 | 2826 | 35.8 | ok |
| llama3-8b | w4 | 5.06 | 2738 | 20.5 | ok |
| llama3-8b | w8 | 8.96 | 2775 | 11.3 | ok |
| mistral-7b | fp16 | 14.77 | 2596 | 6.3 | ok |
| mistral-7b | w2 | — | — | — | skipped |
| mistral-7b | w4 | 4.62 | 2725 | 21.7 | ok |
| mistral-7b | w8 | 8.13 | 2742 | 11.8 | ok |
| phi-3-mini | fp16 | 4.71 | 1429 | 21.4 | ok |
| phi-3-mini | w2 | — | — | — | skipped |
| phi-3-mini | w4 | 2.93 | 1379 | 37.1 | ok |
| phi-3-mini | w8 | 4.71 | 1432 | 21.1 | ok |
| phi-3.5-mini | fp16 | 8.29 | 1455 | 11.5 | ok |
| phi-3.5-mini | w2 | — | — | — | skipped |
| phi-3.5-mini | w4 | 2.93 | 1383 | 37.0 | ok |
| phi-3.5-mini | w8 | 4.71 | 1437 | 21.3 | ok |
| qwen-0.5b | fp16 | 1.34 | 158 | 70.1 | ok |
| qwen-0.5b | w2 | — | — | — | skipped |
| qwen-0.5b | w4 | 0.64 | 147 | 215.2 | ok |
| qwen-0.5b | w8 | 0.89 | 157 | 133.1 | ok |
| qwen-1.5b | fp16 | 3.35 | 516 | 24.9 | ok |
| qwen-1.5b | w2 | — | — | — | skipped |
| qwen-1.5b | w4 | 1.43 | 489 | 89.3 | ok |
| qwen-1.5b | w8 | 2.14 | 507 | 47.1 | ok |
| qwen-3b | fp16 | 6.42 | 1059 | 14.3 | ok |
| qwen-3b | w2 | — | — | — | skipped |
| qwen-3b | w4 | 2.22 | 1013 | 48.4 | ok |
| qwen-3b | w8 | 3.74 | 1046 | 26.4 | ok |
| qwen-7b | fp16 | 15.49 | 2466 | 6.3 | ok |
| qwen-7b | w2 | — | — | — | skipped |
| qwen-7b | w4 | 4.72 | 2318 | 21.8 | ok |
| qwen-7b | w8 | 8.52 | 2418 | 11.9 | ok |

## Article 2: KV cache quantization
| Model | Run | Memory | TTFT | tok/s | Status |
|-------|-----|--------|------|-------|--------|
| llama3-8b | llama3-8b_w4 | 5.06 | 2670 | 20.7 | ok |
| llama3-8b | llama3-8b_w4_kv | 5.06 | 2773 | 20.4 | ok |
| llama3-8b | llama3-8b_w4_kv_long_g | 5.06 | 3054 | 19.8 | ok |
| mistral-7b | mistral-7b_w4 | 4.62 | 2693 | 21.6 | ok |
| mistral-7b | mistral-7b_w4_kv | 4.62 | 2807 | 21.2 | ok |
| qwen-7b | qwen-7b_w4 | 4.72 | 2560 | 21.8 | ok |
| qwen-7b | qwen-7b_w4_kv | 4.72 | 2632 | 21.4 | ok |

## Article 3: Prefill & TTFT
| Model | Run | Memory | TTFT | tok/s | Status |
|-------|-----|--------|------|-------|--------|
| llama3-8b | w4_baseline | 5.06 | 3103 | 20.6 | ok |
| llama3-8b | w4_prefill | 5.06 | 3185 | 18.7 | ok |
| llama3-8b | w4_prefill_p1024 | 5.24 | 5782 | 20.1 | ok |
| llama3-8b | w4_prefill_p256 | 4.92 | 2357 | 13.7 | ok |

## Article 4: Model size & memory ladder
| Model | Run | Memory | TTFT | tok/s | Status |
|-------|-----|--------|------|-------|--------|
| deepseek-r1-llama-8b | fp16 | 16.33 | 2639 | 5.6 | ok |
| deepseek-r1-llama-8b | w2 | — | — | — | skipped |
| deepseek-r1-llama-8b | w4 | 5.06 | 3069 | 19.4 | ok |
| deepseek-r1-llama-8b | w8 | 8.96 | 3139 | 9.8 | ok |
| deepseek-r1-qwen-7b | fp16 | 15.49 | 2518 | 6.4 | ok |
| deepseek-r1-qwen-7b | w2 | — | — | — | skipped |
| deepseek-r1-qwen-7b | w4 | 4.72 | 3559 | 18.6 | ok |
| deepseek-r1-qwen-7b | w8 | 8.52 | 2723 | 11.8 | ok |
| gemma-2-2b | fp16 | 3.32 | 823 | 30.4 | ok |
| gemma-2-2b | w2 | — | — | — | skipped |
| gemma-2-2b | w4 | 2.12 | 803 | 53.1 | ok |
| gemma-2-2b | w8 | 3.32 | 822 | 30.4 | ok |
| gemma-9b | fp16 | 10.51 | 4964 | 5.7 | ok |
| gemma-9b | w2 | — | — | — | skipped |
| gemma-9b | w4 | 5.88 | 3852 | 15.4 | ok |
| gemma-9b | w8 | 10.51 | 4451 | 8.3 | ok |
| llama-3.2-1b | fp16 | 2.71 | 361 | 36.0 | ok |
| llama-3.2-1b | w2 | — | — | — | skipped |
| llama-3.2-1b | w4 | 1.24 | 347 | 112.0 | ok |
| llama-3.2-1b | w8 | 1.75 | 355 | 63.4 | ok |
| llama-3.2-3b | fp16 | 6.73 | 1069 | 13.6 | ok |
| llama-3.2-3b | w2 | — | — | — | skipped |
| llama-3.2-3b | w4 | 2.34 | 1022 | 45.8 | ok |
| llama-3.2-3b | w8 | 3.86 | 1071 | 25.5 | ok |
| llama3-8b | fp16 | 16.33 | 3207 | 4.8 | ok |
| llama3-8b | w2 | 3.11 | 2914 | 35.3 | ok |
| llama3-8b | w4 | 5.06 | 2817 | 20.6 | ok |
| llama3-8b | w8 | 8.96 | 2854 | 11.2 | ok |
| mistral-7b | fp16 | 14.77 | 2697 | 6.2 | ok |
| mistral-7b | w2 | — | — | — | skipped |
| mistral-7b | w4 | 4.62 | 3456 | 17.4 | ok |
| mistral-7b | w8 | 8.13 | 3438 | 11.4 | ok |
| phi-3-mini | fp16 | 4.71 | 1510 | 21.3 | ok |
| phi-3-mini | w2 | — | — | — | skipped |
| phi-3-mini | w4 | 2.93 | 1505 | 35.6 | ok |
| phi-3-mini | w8 | 4.71 | 1536 | 21.2 | ok |
| phi-3.5-mini | fp16 | 8.29 | 1439 | 11.6 | ok |
| phi-3.5-mini | w2 | — | — | — | skipped |
| phi-3.5-mini | w4 | 2.93 | 1472 | 36.8 | ok |
| phi-3.5-mini | w8 | 4.71 | 1507 | 21.0 | ok |
| qwen-0.5b | fp16 | 1.34 | 149 | 83.0 | ok |
| qwen-0.5b | w2 | — | — | — | skipped |
| qwen-0.5b | w4 | 0.64 | 145 | 238.3 | ok |
| qwen-0.5b | w8 | 0.89 | 168 | 146.0 | ok |
| qwen-1.5b | fp16 | 3.35 | 507 | 28.3 | ok |
| qwen-1.5b | w2 | — | — | — | skipped |
| qwen-1.5b | w4 | 1.43 | 488 | 90.8 | ok |
| qwen-1.5b | w8 | 2.14 | 503 | 51.0 | ok |
| qwen-3b | fp16 | 6.42 | 1059 | 14.3 | ok |
| qwen-3b | w2 | — | — | — | skipped |
| qwen-3b | w4 | 2.22 | 1017 | 48.3 | ok |
| qwen-3b | w8 | 3.74 | 1056 | 26.4 | ok |
| qwen-7b | fp16 | 15.49 | 2425 | 6.4 | ok |
| qwen-7b | w2 | — | — | — | skipped |
| qwen-7b | w4 | 4.72 | 2653 | 21.6 | ok |
| qwen-7b | w8 | 8.52 | 2697 | 11.8 | ok |

## Article 5: The full optimization stack
| Model | Run | Memory | TTFT | tok/s | Status |
|-------|-----|--------|------|-------|--------|
| llama3-8b | fp16 | 16.33 | 2689 | 5.6 | ok |
| llama3-8b | optimized | 5.06 | 2746 | 19.9 | ok |
| mistral-7b | fp16_mistral | 14.77 | 4350 | 3.6 | ok |
| mistral-7b | optimized_mistral | 4.62 | 3954 | 16.0 | ok |

## Article 6: Speculative decoding
| Model | Run | Memory | TTFT | tok/s | Status |
|-------|-----|--------|------|-------|--------|
| llama3-8b | llama3-8b_w4_baseline | 5.06 | 3390 | 18.8 | ok |
| llama3-8b | w4 | — | — | — | error |
| mistral-7b | mistral-7b_w4_baseline | 4.62 | 2858 | 19.1 | ok |
| mistral-7b | w4 | — | — | — | error |
| qwen-7b | qwen-7b_w4_baseline | 4.72 | 3613 | 15.9 | ok |
| qwen-7b | qwen-7b_w4_speculative | 5.00 | 2856 | 28.3 | ok |

## Article 7: Context, generation length & prompt cache
| Model | Run | Memory | TTFT | tok/s | Status |
|-------|-----|--------|------|-------|--------|
| llama3-8b | ctx_p1024 | 5.24 | 6503 | 14.9 | ok |
| llama3-8b | ctx_p2048 | 5.35 | 15355 | 11.9 | ok |
| llama3-8b | ctx_p256 | 4.92 | 1406 | 20.3 | ok |
| llama3-8b | ctx_p512 | 5.06 | 2839 | 20.5 | ok |
| llama3-8b | gen_g256 | 5.06 | 3768 | 16.4 | ok |
| llama3-8b | gen_g512 | 5.06 | 3616 | 15.9 | ok |
| llama3-8b | gen_g64 | 5.06 | 3737 | 15.0 | ok |
| llama3-8b | prefix_cache | 4.99 | 1547 | 16.3 | ok |
| llama3-8b | wl_chat_light | 4.74 | 1518 | 13.5 | ok |
| llama3-8b | wl_chat_standard | 5.06 | 4031 | 13.4 | ok |
| llama3-8b | wl_complete_code | 4.92 | 2071 | 17.3 | ok |
| llama3-8b | wl_rag_agent | 6.10 | 31118 | 11.3 | ok |
| llama3-8b | wl_random_baseline | 5.06 | 3911 | 14.0 | ok |
| llama3-8b | wl_summarize_long | 5.35 | 15897 | 10.2 | ok |

## Article 10: Local runtimes compared
| Model | Run | Memory | TTFT | tok/s | Status |
|-------|-----|--------|------|-------|--------|
| llama3-8b | fp16 | 16.33 | 2794 | 5.5 | ok |
| llama3-8b | w4 | 5.06 | 6322 | 9.5 | ok |
| mistral-7b | w4 | 4.62 | 2613 | 20.6 | ok |
