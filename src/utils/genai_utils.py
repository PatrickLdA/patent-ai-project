"""
genai_utils.py

Utilities for extracting and managing TRIZ ontology elements from patent data using Generative AI.

This module provides:
- Data schemas for TRIZ elements and their derived relationships.
- The TRIZExtractor class for extracting, comparing, and storing TRIZ elements using LLMs and vector search.
- Integration with OpenAI through LangChain for semantic search and structured extraction.

Main Concepts:
---------------
- TRIZ Ontology: Describes patents using four key elements: kind_effect, task, object, physical_effect.
- Derived Elements: New TRIZ relationships suggested by the LLM, linked to the closest existing element.
- Vectorstore: Semantic database for storing and retrieving TRIZ elements.

Classes:
--------
- TrizSchema: Pydantic model for standard TRIZ elements.
- TrizSchemaOutput: Extends TrizSchema, adds 'derived_from' for provenance tracking.
- TRIZExtractor: Main class for extraction, comparison, and storage of TRIZ elements.

TRIZExtractor Workflow:
-----------------------
1. **extract**: Uses LLM to extract TRIZ elements from patent text.
2. **vectorstore_search**: Finds similar TRIZ elements in the vectorstore.
3. **derivative_decisor**: Compares extracted and existing elements, decides which to use.
   - If a similar element exists (score < threshold), returns it.
   - Otherwise, returns the extracted element and links to the closest existing one via 'derived_from'.
4. **insert_derived_elements**: Adds new derived elements to the vectorstore for future retrieval.
5. **run**: Orchestrates the full workflow for a given patent.

Example Usage:
--------------
extractor = TRIZExtractor(vectorstore=vectorstore, verbose=True)
result = extractor.run(title="Patent Title", abstract="Patent Abstract")

Returns:
    dict with keys: effect, task, object, physical_effect, derived_from
"""

from pydantic import BaseModel
from openai import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document


class TrizSchema(BaseModel):
    """
    Pydantic model for TRIZ ontology elements.

    Attributes:
        kind_effect (str): Type of effect (e.g., "Mechanical", "Thermal").
        task (str): Action performed (e.g., "Compress", "Heat").
        object (str): Target object (e.g., "Water", "Metal").
        physical_effect (str): Resulting physical effect (e.g., "Temperature increase").
    """
    task: str
    object: str
    physical_effect: str

class TrizSchemaOutput(TrizSchema):
    """
    Extends TrizSchema to include provenance information.

    Attributes:
        derived_from (str | None): Reference to the origin TRIZ element (page_content or None).
    """
    derived_from: str | None


class TRIZExtractor:
    """
    Extracts, compares, and manages TRIZ ontology elements from patent data using LLMs and vector search.

    Args:
        vectorstore: LangChain vectorstore instance for semantic search.
        verbose (bool): If True, prints detailed logs.

    Methods:
        extract(context: str) -> dict:
            Extracts TRIZ elements from patent text using LLM.

        vectorstore_search(query: str, k: int = 3) -> list:
            Finds k most similar TRIZ elements in the vectorstore.

        derivative_decisor(extraction, similar_samples, threshold=0.4) -> dict:
            Decides whether to use an existing TRIZ element or a new derived one.

        insert_derived_elements(result: dict):
            Adds new derived TRIZ elements to the vectorstore.

        run(title: str, abstract: str, threshold=0.4) -> dict:
            Full workflow: extract, compare, store, and return TRIZ elements.
    """
    def __init__(self, vectorstore, verbose=False, threshold=0.4, temperature=0):
        self.verbose = verbose
        self.vectorstore = vectorstore
        self.threshold = threshold
        self.temperature = temperature
        self.embedding_model = OpenAIEmbeddings()

        self.openai_client = OpenAI()
        self.prompt_extraction = """Você receberá o título e o resumo de uma patente em português.
A partir destes dados, vamos minerar elementos seguindo a **Ontologia da TRIZ**. 

Sua tarefa é identificar e retornar os seguintes elementos, respeitando as definições abaixo:

- `task`: ação potencial expressa **sempre como verbo no infinitivo**, representando a atividade realizada para se atingir um objetivo (ex: "Aquecer", "Separar", "Reduzir atrito").
- `object`: o **meio ou material alvo** sobre o qual a tarefa atua. Deve ser **sempre um substantivo** (ex: "Água", "Gás", "Superfície metálica").
- `physical_effect`: o **efeito físico, químico ou transformação observável** associado à tarefa ou ao objeto. Pode ser um princípio (ex: "Efeito Joule", "Refração da luz", "Difusão térmica") ou um resultado direto da ação (ex: "Aumento da temperatura").

Atenção:
- Se não houver informação explícita, **generalize** a partir do contexto do resumo (ex: "Transformação de energia elétrica em calor" → "Efeito Joule").
- Retorne **apenas um termo em cada campo**, evitando listas.
- Use linguagem técnica, mas clara.

# Formato de saída
Retorne o resultado em JSON no seguinte formato:

```json
{
  "task": "Separar",
  "object": "Gás",
  "physical_effect": "Difusão"
}"""


    def extract(self, context: str) -> dict:
        """
        Extracts TRIZ elements from patent text using LLM.

        Args:
            context (str): Patent title and abstract.

        Returns:
            dict: Parsed TRIZ elements.
        """
        response = self.openai_client.responses.parse(
            model="gpt-4.1",
            temperature=self.temperature,
            input=[
                {"role": "system", "content": self.prompt_extraction},
                {
                    "role": "user",
                    "content": context,
                },
            ],
            text_format=TrizSchema,
        )

        return response.output_parsed.model_dump() if response.output_parsed is not None else {}
    
    def vectorstore_search(self, query, k=3):
        """
        Finds k most similar TRIZ elements in the vectorstore.

        Args:
            query (str): Query text.
            k (int): Number of results.

        Returns:
            list: List of (Document, score) tuples.
        """
        # ref.: https://api.python.langchain.com/en/latest/vectorstores/langchain_community.vectorstores.chroma.Chroma.html#langchain_community.vectorstores.chroma.Chroma.similarity_search_by_vector_with_relevance_scores
        results = self.vectorstore.similarity_search_by_vector_with_relevance_scores(
            embedding=self.embedding_model.embed_query(query),
            k=k
        )

        return results

    def prepare_context_derivative_decisor(self, query, extraction, similar_samples):
        prompt_derivative_decisor_user = f"""{query}

        # Extração recomendada
        {extraction}

        # Amostras semelhantes já existentes na TRIZ
        """
        similar_samples_print = ""
        ids_similar_prompt = "# IDs de referência da TRIZ\n"

        for x in similar_samples:
            similar_samples_print += f"{x}\n"
            ids_similar_prompt += f"- {x.metadata['id']}\n"

        prompt_derivative_decisor_user += similar_samples_print
        prompt_derivative_decisor_user += ids_similar_prompt

        return prompt_derivative_decisor_user

    def derivative_decisor(self, extraction, similar_samples):
        """
        Decides whether to use an existing TRIZ element or a new derived one.

        Args:
            extraction: Extracted TRIZ element (TrizSchema).
            similar_samples: List of (Document, score) tuples.
            threshold (float): Similarity threshold.

        Returns:
            dict: Chosen TRIZ element with provenance.
        """
        best_doc, best_score = min(similar_samples, key=lambda x: x[1])
        best_content = best_doc.page_content

        if self.verbose:
            print(f"Best_doc: {best_doc}")
            print(f"Best content: {best_content}\nScore: {best_score}")

        if best_score < self.threshold:
            # Return the closest element
            return {
                "derived_from": None,
                "task": best_doc.metadata["task"],
                "object": best_doc.metadata["object"],
                "physical_effect": best_doc.metadata["physical_effect"]
            }
        else:
            # Returns the extracted element with the closest derived one
            return {
                "derived_from": best_content,
                "task": extraction["task"],
                "object": extraction["object"],
                "physical_effect": extraction["physical_effect"]
            }


    def insert_derived_elements(self, result):
        """
        Adds new derived TRIZ elements to the vectorstore if 'derived_from' is not None.

        Args:
            result (dict): TRIZ element with provenance.
        """
        if result.get("derived_from"):

            text = f'O/a {result["task"]} atua no/na {result["object"]} para produzir um/uma {result["physical_effect"]}.'
            metadata = {
                "task": result["task"],
                "object": result["object"],
                "physical_effect": result["physical_effect"],
                "derived_from": result["derived_from"]
            }

            doc = Document(page_content=text, metadata=metadata)
            self.vectorstore.add_documents([doc])
            if self.verbose:
                print("\nElemento derivado inserido no vectorstore:")
                print("Texto:", text)
                print("Metadados:", metadata)

    def run(self, title:str, abstract:str) -> dict:
        """
        Full workflow for extracting and storing TRIZ elements from a patent.

        Args:
            title (str): Patent title.
            abstract (str): Patent abstract.
            threshold (float): Similarity threshold.

        Returns:
            dict: Final TRIZ element with provenance.
        """
        query = f"""Título: {title}
        Resumo: {abstract}"""

        if self.verbose:
            print("Query to be processed\n", query, "\n**********\n\n")

        extraction = self.extract(query)

        if self.verbose:
            print("Extraction result\n", extraction, "\n**********\n\n")

        similar_samples = self.vectorstore_search(query=query)

        if self.verbose:
            print("Similar samples result\n", similar_samples, "\n**********\n\n")

        result = self.derivative_decisor(extraction, similar_samples)
        if self.verbose:
            print("result\n", result, "\n**********\n\n")

        self.insert_derived_elements(result)

        return result