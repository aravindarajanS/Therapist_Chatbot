import pandas as pd
from typing import Tuple, List
from llama_index.legacy.evaluation import (
    CorrectnessEvaluator,
    FaithfulnessEvaluator,
    RelevancyEvaluator,
    SemanticSimilarityEvaluator,
)

from modules.os_retriever_metrics import HitRate, MRR


class Evaluator:
  
    def __init__(self):
      
        self.query = None
        self.context = None
        self.generated_answer = None
        self.reference_answer = None
        self.hit_rate_metric = HitRate()
        self.mrr_metric = MRR()
      

    def evaluate_one_gen(
        self,
        query: str,  # query to evaluate
        context: str,  # context
        generated_answer: str,  # LLM generated answer
        reference_answer: str,  # Reference answer
    ) -> Tuple[float, float, float, float]:
        """
        Evaluate a single generated answer against a reference answer.

        Parameters
        ----------
        query : str
            The query to evaluate.

        context : str
            The context in which the query was generated.

        generated_answer : str
            The answer generated by the LLM (Large Language Model).

        reference_answer : str
            The reference answer against which the generated answer is evaluated.
        Returns
        -------
        faithfulness_res : float
            Faithfulness score indicating how faithful the generated answer is to the reference answer.

        correctness_res : float
            Correctness score indicating the correctness of the generated answer.

        relevancy_res : float
            Relevancy score indicating the relevancy of the generated answer to the query and context.

        ss_res : float
            Semantic similarity score indicating the semantic similarity between the generated answer and the reference answer.
        """
        faithfulness = FaithfulnessEvaluator()
        faithfulness_result = faithfulness.evaluate(
            query=query, response=generated_answer, contexts=context
        )
        correctness = CorrectnessEvaluator()
        correctness_result = correctness.evaluate(
            query=query, response=generated_answer, reference=reference_answer
        )
        relevancy = RelevancyEvaluator()
        relevancy_result = relevancy.evaluate(
            query=query,
            response=generated_answer,
            contexts=context,
        )
        semantic_similarity = SemanticSimilarityEvaluator()
        semantic_similarity_result = semantic_similarity.evaluate(
            query=query,
            response="\n".join(generated_answer),
            reference="\n".join(reference_answer),
        )
        faithfulness_res = faithfulness_result.dict()["score"]
        correctness_res = correctness_result.dict()["score"]
        relevancy_res = relevancy_result.dict()["score"]
        ss_res = semantic_similarity_result.dict()["score"]
        df = pd.DataFrame(
            {
                "metric": ["faithfulness", "correctness", "relevancy", "semantic similarity"],
                "score": [faithfulness_res, correctness_res, relevancy_res, ss_res],
            }
        )
        print(df)
        return faithfulness_res, correctness_res, relevancy_res, ss_res

    def evaluate_one_retr(
        self, retrieved_docs: List[str], reference_docs: List[str]
    ) -> Tuple[float, float]:
        """
        Evaluate a single retrieval instance.

        Parameters
        ----------
        retrieved_docs : List[str]
            The list of documents retrieved by the retrieval system.

        reference_docs : List[str]
            The list of reference documents expected to be retrieved.

        Returns
        -------
        mrr_res : float
            Mean Reciprocal Rank (MRR) score indicating the quality of the retrieval system.

        hitrate_res : float
            Hit rate score indicating the proportion of correctly retrieved documents.
        """
        hitrate_res = self.hit_rate_metric.compute(
            expected_documents=reference_docs, returned_documents=retrieved_docs
        )
        mrr_res = self.mrr_metric.compute(
            expected_documents=reference_docs, returned_documents=retrieved_docs
        )
        return mrr_res.score, hitrate_res.score
