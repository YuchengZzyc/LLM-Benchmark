# Evaluation Report — Qwen3.5-4B

> Generated at: 2026-09-10T15:43:18Z  

> Config: `configs/baseline_qwen35_4b.yaml`


---

## Model Information

| Field | Value |
| --- | --- |
| Model | Qwen3.5-4B |
| Version | `qwen35-4b-base-v0.0` |
| Type | `baseline` |
| Path | `/data/yucheng/madm-llm/models/Qwen3.5-4B` |
| Dtype | `bfloat16` |
| Trust Remote Code | `False` |

## Environment

| Field | Value |
| --- | --- |
| OS | Linux |
| OS Release | 7.0.0-29-generic |
| Python | 3.11.16 |
| PyTorch | 2.11.0+cu128 |
| transformers | 5.6.0 |
| accelerate | 1.11.0 |
| lm_eval | 0.4.13 |
| PyYAML | 6.0.3 |
| GPU | NVIDIA GeForce RTX 5090 |

## Benchmark Configuration

| Dimension | Benchmark | Metric |
| --- | --- | --- |
| knowledge | `mmlu_redux_generative` | Accuracy |
|  | `mmlu_prox_en` |  |
|  | `mmlu_prox_zh` |  |
|  | `ceval-valid` |  |
| reasoning | `gsm8k` | Exact Match / Accuracy |
|  | `leaderboard_math_hard` |  |
|  | `aime24` |  |
|  | `aime25` |  |
|  | `minerva_math500` |  |
| instruction | `ifeval` | strict / loose accuracy |
| long_context | `multifieldqa_en` | task score |
|  | `qasper` |  |
|  | `hotpotqa` |  |
|  | `2wikimqa` |  |
|  | `musique` |  |
|  | `gov_report` |  |
|  | `multi_news` |  |
|  | `trec` |  |
|  | `triviaqa` |  |
|  | `samsum` |  |
|  | `passage_count` |  |
|  | `passage_retrieval_en` |  |
| multilingual | `mgsm_native_cot_zh` | language accuracy |
|  | `mmmlu_zh_cn` |  |

| Setting | Value |
| --- | --- |
| Framework | `lm-eval` |
| Batch Size | `8` |
| Apply Chat Template | `True` |
| Max Length | `4096` |
| Num Fewshot | `None` |
| Seed | `42` |
| Device | `cuda` |

## Results

- **framework**: lm-eval
- **version**: qwen35-4b-base-v0.0
### dimensions

#### reasoning

##### gsm8k

- **sample_len**: 30.0
- **exact_match,strict-match**: 0.4
- **exact_match_stderr,strict-match**: 0.090972
- **exact_match,flexible-extract**: 0.2
- **exact_match_stderr,flexible-extract**: 0.074278
##### leaderboard_math_hard

- **sample_len**: 210.0
- **exact_match,none**: 0.152381
- **exact_match_stderr,none**: 0.023595
##### aime24

- **sample_len**: 30.0
- **exact_match,none**: 0.0
- **exact_match_stderr,none**: 0.0
##### aime25

- **sample_len**: 30.0
- **exact_match,none**: 0.0
- **exact_match_stderr,none**: 0.0
##### minerva_math500

- **sample_len**: 30.0
- **exact_match,none**: 0.066667
- **exact_match_stderr,none**: 0.046321
- **math_verify,none**: 0.4
- **math_verify_stderr,none**: 0.090972
#### long_context

##### multifieldqa_en

- **em**: 0.0
- **f1**: 0.060965
- **samples**: 30
- **category**: single_doc_qa
##### qasper

- **em**: 0.0
- **f1**: 0.025769
- **samples**: 30
- **category**: single_doc_qa
##### hotpotqa

- **em**: 0.0
- **f1**: 0.011066
- **samples**: 30
- **category**: multi_doc_qa
##### 2wikimqa

- **em**: 0.0
- **f1**: 0.018733
- **samples**: 30
- **category**: multi_doc_qa
##### musique

- **em**: 0.0
- **f1**: 0.004517
- **samples**: 30
- **category**: multi_doc_qa
##### gov_report

- **rougeL**: 0.164453
- **samples**: 30
- **category**: summarization
##### multi_news

- **rougeL**: 0.129189
- **samples**: 30
- **category**: summarization
##### trec

- **accuracy**: 0.0
- **samples**: 30
- **category**: few_shot
##### triviaqa

- **em**: 0.0
- **f1**: 0.014398
- **samples**: 30
- **category**: few_shot
##### samsum

- **rougeL**: 0.069442
- **samples**: 30
- **category**: few_shot
##### passage_count

- **accuracy**: 0.0
- **samples**: 30
- **category**: synthetic
##### passage_retrieval_en

- **accuracy**: 0.033333
- **samples**: 30
- **category**: synthetic
#### knowledge

##### mmlu_redux_generative

- **sample_len**: 1710.0
- **exact_match,default**: 0.233333
- **exact_match_stderr,default**: 0.010173
##### mmlu_prox_en

- **sample_len**: 420.0
- **exact_match,custom-extract**: 0.185714
- **exact_match_stderr,custom-extract**: 0.018453
##### mmlu_prox_zh

- **sample_len**: 420.0
- **exact_match,custom-extract**: 0.37381
- **exact_match_stderr,custom-extract**: 0.022893
##### ceval-valid

- **sample_len**: 1195.0
- **acc,none**: 0.241004
- **acc_stderr,none**: 0.01237
- **acc_norm,none**: 0.241004
- **acc_norm_stderr,none**: 0.01237
#### instruction

##### ifeval

- **sample_len**: 30.0
- **prompt_level_strict_acc,none**: 0.133333
- **prompt_level_strict_acc_stderr,none**: 0.063124
- **inst_level_strict_acc,none**: 0.333333
- **prompt_level_loose_acc,none**: 0.133333
- **prompt_level_loose_acc_stderr,none**: 0.063124
- **inst_level_loose_acc,none**: 0.333333
#### multilingual

##### mgsm_native_cot_zh

- **sample_len**: 30.0
- **exact_match,strict-match**: 0.0
- **exact_match_stderr,strict-match**: 0.0
- **exact_match,flexible-extract**: 0.6
- **exact_match_stderr,flexible-extract**: 0.090972
##### mmmlu_zh_cn

- **sample_len**: 1710.0
- **acc,none**: 0.235088
- **acc_stderr,none**: 0.010224
- **acc_norm,none**: 0.235088
- **acc_norm_stderr,none**: 0.010224
- **errors**: ['全量批次失败，按维度降级重跑: lm-eval 执行失败: _ssl.c:999: The handshake operation timed out']

## Comparison

| Version | Status |
| --- | --- |
| qwen35-4b-base-v0.0 | completed |
| qwen35-4b-smoke-v0.0 | completed |