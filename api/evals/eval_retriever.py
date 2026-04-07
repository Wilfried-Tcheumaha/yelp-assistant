import asyncio

from langsmith import Client

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import IDBasedContextPrecision, IDBasedContextRecall, Faithfulness, ResponseRelevancy

from api.agents.retrieval_generation import rag_pipeline

ls_client = Client()
ragas_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4.1-mini"))
ragas_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

async def ragas_faithfulness(run, example):

    sample = SingleTurnSample(
            user_input=run.outputs["question"],
            response=run.outputs["answer"],
            retrieved_contexts=run.outputs["retrieved_restaurant_names"]
        )
    scorer = Faithfulness(llm=ragas_llm)

    return await scorer.single_turn_ascore(sample)


async def ragas_response_relevancy(run, example):

    sample = SingleTurnSample(
            user_input=run.outputs["question"],
            response=run.outputs["answer"],
            retrieved_contexts=run.outputs["retrieved_restaurant_names"]
        )
    scorer = ResponseRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)

    return await scorer.single_turn_ascore(sample)


async def ragas_context_precision_id_based(run, example):

    sample = SingleTurnSample(
            retrieved_context_ids=run.outputs["retrieved_context_ids"],
            reference_context_ids=example.outputs["reference_context_ids"]
        )
    scorer = IDBasedContextPrecision()

    return await scorer.single_turn_ascore(sample)


async def ragas_context_recall_id_based(run, example):

    sample = SingleTurnSample(
            retrieved_context_ids=run.outputs["retrieved_context_ids"],
            reference_context_ids=example.outputs["reference_context_ids"]
        )
    scorer = IDBasedContextRecall()

    return await scorer.single_turn_ascore(sample)


async def predict(inputs: dict) -> dict:
    """Async target so LangSmith uses aevaluate_run on the main loop (not thread workers)."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, rag_pipeline, inputs["question"])


async def run_evaluation():
    return await ls_client.aevaluate(
        predict,
        data="yelp-rag-evaluation-dataset",
        evaluators=[
            ragas_faithfulness,
            ragas_response_relevancy,
            ragas_context_precision_id_based,
            ragas_context_recall_id_based,
        ],
        experiment_prefix="retriever",
        max_concurrency=1,
    )


if __name__ == "__main__":
    asyncio.run(run_evaluation())