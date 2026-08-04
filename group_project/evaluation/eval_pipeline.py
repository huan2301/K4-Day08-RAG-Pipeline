"""
RAG Evaluation Pipeline.

Sử dụng DeepEval / RAGAS / TruLens để đánh giá chất lượng RAG pipeline.
Chọn 1 framework và implement đầy đủ.

Yêu cầu:
    1. Load golden_dataset.json (≥15 Q&A pairs)
    2. Chạy RAG pipeline trên từng question
    3. Evaluate với 4 metrics: faithfulness, relevance, context_recall, context_precision
    4. So sánh A/B ít nhất 2 configs
    5. Export results ra results.md

Lưu ý rate limit nếu dùng model OpenRouter ":free": RAGAS/DeepEval gọi LLM RẤT NHIỀU LẦN
(không phải 1 lần/câu hỏi mà nhiều lần/metric/câu hỏi). Model free của OpenRouter giới hạn
50 request/ngày CHO CẢ TÀI KHOẢN (không phải theo model hay theo API key — đổi model free
khác hay tạo key mới KHÔNG reset quota). Nếu chạy full 15+ câu hỏi mà bị rate limit giữa
chừng, thử giảm xuống subset 5 câu để chạy kịp trong buổi, hoặc nạp $10 credit để mở khóa
1000 request/ngày.
"""

import json
import os
from pathlib import Path
import time

from dotenv import load_dotenv

load_dotenv()

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"

# RAGAS uses this LLM as an independent judge, not as the chatbot's generator.
# Override these values in .env when using a different OpenRouter deployment.
RAGAS_MODEL = os.getenv("RAGAS_MODEL", "nvidia/nemotron-3-ultra-30b-a3b")
RAGAS_EMBEDDING_MODEL = os.getenv(
    "RAGAS_EMBEDDING_MODEL", "openai/text-embedding-3-small"
)
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
REQUEST_DELAY_SECONDS = float(os.getenv("RAGAS_REQUEST_DELAY_SECONDS", "5"))


def load_golden_dataset() -> list[dict]:
    """Load golden dataset từ JSON file."""
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# Option 1: DeepEval
# =============================================================================

def evaluate_with_deepeval(rag_pipeline, golden_dataset: list[dict]) -> dict:
    """
    Evaluate RAG pipeline sử dụng DeepEval.

    pip install deepeval
    """
    # TODO: Implement
    #
    # from deepeval import evaluate
    # from deepeval.metrics import (
    #     FaithfulnessMetric,
    #     AnswerRelevancyMetric,
    #     ContextualRecallMetric,
    #     ContextualPrecisionMetric,
    # )
    # from deepeval.test_case import LLMTestCase
    #
    # test_cases = []
    # for item in golden_dataset:
    #     result = rag_pipeline.generate_with_citation(item["question"])
    #     test_case = LLMTestCase(
    #         input=item["question"],
    #         actual_output=result["answer"],
    #         expected_output=item["expected_answer"],
    #         retrieval_context=[c["content"] for c in result["sources"]],
    #     )
    #     test_cases.append(test_case)
    #
    # metrics = [
    #     FaithfulnessMetric(threshold=0.7),
    #     AnswerRelevancyMetric(threshold=0.7),
    #     ContextualRecallMetric(threshold=0.7),
    #     ContextualPrecisionMetric(threshold=0.7),
    # ]
    #
    # results = evaluate(test_cases, metrics)
    # return results
    raise NotImplementedError("Implement evaluate_with_deepeval")


# =============================================================================
# Option 2: RAGAS
# =============================================================================

def evaluate_with_ragas(rag_pipeline, golden_dataset: list[dict]):
    """
    Evaluate RAG pipeline sử dụng RAGAS.

    Nemotron 3 Ultra is used as the judge through OpenRouter. Requests are
    serialized and paced at five seconds by default to protect API quotas.

    Required environment variables:
        OPENROUTER_API_KEY: OpenRouter key used by the RAGAS judge.

    Optional environment variables:
        RAGAS_MODEL: Override the Nemotron 3 Ultra model ID.
        RAGAS_REQUEST_DELAY_SECONDS: Minimum interval between requests.
    """
    if not golden_dataset:
        raise ValueError("golden_dataset must contain at least one evaluation case.")
    if REQUEST_DELAY_SECONDS < 0:
        raise ValueError("RAGAS_REQUEST_DELAY_SECONDS must be non-negative.")

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required for RAGAS evaluation.")

    from datasets import Dataset
    from langchain_core.rate_limiters import InMemoryRateLimiter
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )
    from ragas.run_config import RunConfig

    generate = getattr(rag_pipeline, "generate_with_citation", rag_pipeline)
    if not callable(generate):
        raise TypeError("rag_pipeline must be callable or expose generate_with_citation().")

    eval_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for item in golden_dataset:
        result = generate(item["question"])
        if not isinstance(result, dict) or not isinstance(result.get("answer"), str):
            raise ValueError("RAG pipeline must return a dict containing a string 'answer'.")

        sources = result.get("sources", [])
        if not isinstance(sources, list):
            raise ValueError("RAG pipeline result 'sources' must be a list.")

        eval_data["question"].append(item["question"])
        eval_data["answer"].append(result["answer"])
        eval_data["contexts"].append(
            [source["content"] for source in sources if isinstance(source, dict) and source.get("content")]
        )
        eval_data["ground_truth"].append(item["expected_answer"])

        # Keep pipeline generation requests away from the first RAGAS judge call.
        time.sleep(REQUEST_DELAY_SECONDS)

    request_rate_limiter = InMemoryRateLimiter(
        requests_per_second=1 / REQUEST_DELAY_SECONDS if REQUEST_DELAY_SECONDS else 1000,
        check_every_n_seconds=0.1,
        max_bucket_size=1,
    )
    judge_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=RAGAS_MODEL,
            api_key=api_key,
            base_url=OPENROUTER_BASE_URL,
            temperature=0,
            max_retries=0,
            rate_limiter=request_rate_limiter,
        )
    )
    embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            model=RAGAS_EMBEDDING_MODEL,
            api_key=api_key,
            base_url=OPENROUTER_BASE_URL,
            max_retries=0,
        )
    )

    return evaluate(
        Dataset.from_dict(eval_data),
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=judge_llm,
        embeddings=embeddings,
        # RAGAS otherwise evaluates metric jobs concurrently, bypassing the pace
        # intended for a rate-limited API key.
        run_config=RunConfig(max_workers=1, max_retries=0),
        raise_exceptions=True,
    )


# =============================================================================
# Option 3: TruLens
# =============================================================================

def evaluate_with_trulens(rag_pipeline, golden_dataset: list[dict]) -> dict:
    """
    Evaluate RAG pipeline sử dụng TruLens.

    pip install trulens
    """
    # TODO: Implement
    #
    # from trulens.apps.custom import TruCustomApp
    # from trulens.core import Feedback
    # from trulens.providers.openai import OpenAI as TruOpenAI
    #
    # provider = TruOpenAI()
    #
    # f_faithfulness = Feedback(provider.groundedness_measure_with_cot_reasons).on_output()
    # f_relevance = Feedback(provider.relevance).on_input_output()
    # f_context_relevance = Feedback(provider.context_relevance).on_input()
    #
    # tru_rag = TruCustomApp(
    #     rag_pipeline,
    #     app_name="EcommerceSupport_RAG",
    #     feedbacks=[f_faithfulness, f_relevance, f_context_relevance],
    # )
    #
    # with tru_rag as recording:
    #     for item in golden_dataset:
    #         rag_pipeline.generate_with_citation(item["question"])
    #
    # # Dashboard: from trulens.dashboard import run_dashboard; run_dashboard()
    raise NotImplementedError("Implement evaluate_with_trulens")


# =============================================================================
# A/B Comparison
# =============================================================================

def compare_configs(rag_pipeline, golden_dataset: list[dict]):
    """
    So sánh A/B giữa ít nhất 2 configs.

    Gợi ý configs để so sánh:
    - Config A: hybrid search + reranking
    - Config B: dense-only (không reranking)
    - Config C: hybrid search + PageIndex fallback
    """
    # TODO: Implement A/B comparison
    
    configs = {
        "hybrid_rerank": {"use_reranking": True, "alpha": 0.5},
        "dense_only": {"use_reranking": False, "alpha": 1.0},
    }
    
    results = {}
    for config_name, params in configs.items():
        # Run eval with this config
        ...
        results[config_name] = scores
    
    return results
    raise NotImplementedError("Implement compare_configs")


# =============================================================================
# Export Results
# =============================================================================

def export_results(results: dict, comparison: dict):
    """Export evaluation results to results.md"""
    # TODO: Format and write results
    
    content = "# RAG Evaluation Results\n\n"
    content += "## Overall Scores\n\n"
    content += "| Metric | Score |\n|--------|-------|\n"
    ...
    content += "\n## A/B Comparison\n\n"
    ...
    content += "\n## Worst Performers\n\n"
    ...
    content += "\n## Recommendations\n\n"
    ...
    
    RESULTS_PATH.write_text(content, encoding="utf-8")
    raise NotImplementedError("Implement export_results")


if __name__ == "__main__":
    golden_dataset = load_golden_dataset()
    print(f"Loaded {len(golden_dataset)} test cases")

    from src.task10_generation import generate_with_citation

    results = evaluate_with_ragas(generate_with_citation, golden_dataset)
    print(results.to_pandas().to_string(index=False))
