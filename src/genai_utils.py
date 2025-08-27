from pydantic import BaseModel
from openai import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document


class TrizSchema(BaseModel):
    kind_effect: str
    task: str
    object: str
    physical_effect: str

class TrizSchemaOutput(TrizSchema):
    derived_from: str | None


class TRIZExtractor:
    def __init__(self, vectorstore, verbose=False):
        self.verbose = verbose
        self.vectorstore = vectorstore
        self.embedding_model = OpenAIEmbeddings()

        self.openai_client = OpenAI()
        self.prompt_extraction = """Você receberá o título e o resumo de uma patente em português.
            A partir destes dados, vamos buscar minerar dados desta patente seguindo a **Ontologia da TRIZ**. 
            Esta metodologia busca descrever uma patente a partir de uma série de elementos. Sua tarefa é 
            identificar e retornar os seguintes elementos:

            - `task`: ação ou tarefa (ex: "Aquecer");
            - `object`: objeto-alvo da ação (ex: "Água");
            - `kind_effect`: tipo de efeito causado no objeto a partir da tarefa (ex: "Aumento da temperatura");
            - `physical_effect`: o efeito físico utilizado ou observado (ex: "Efeito Joule").

            É importante notar que a mineração dos dados não busca extrair exatamente os componentes da patente, 
            mas sim um conjunto de termos e características suficientemente genéricos que consigam descrever a 
            patente.

            # Formato de saída
            O formato de saída deve ser um JSON com os seguintes campos:
            - `effect`: tipo de efeito (ex: "Aumento da temperatura");
            - `task`: `task` Tarefa (ex: "Aumentar a temperatura");
            - `object`: `object` Objeto (ex: "Água");
            - `physical_effect`: `physical_effect` Efeito físico (ex: "Aumento da temperatura").

            Exemplo:
            ```json
            {{
            "effect": "Efeito",
            "task": "Identificação",
            "object": "Gás",
            "physical_effect": "Adição de massa"
            }}
            ```
        """

        self.prompt_derivate_decision = """Você receberá o título e o resumo de uma patente em 
            português, bem como alguns metadados extraídos com base na **Ontologia da TRIZ**. Esta 
            metodologia busca descrever uma patente a partir de uma série de elementos.

            Além da patente e os metadados extraídos, você receberá um conjunto de dados já existentes na 
            TRIZ que podem ser usados para descrever a patente. Sua tarefa será decidir se há algum conjunto 
            de dados já existentes da TRIZ que podem ser usados para descrever a patente ou se os dados 
            minerados serão utilizados. É importante tentar priorizar um elemento que já exista na TRIZ. Só 
            traga um novo elemento se realmente nenhum elemento se aproximar do contexto.

            Em resumo, a lógica adotada será:
            1. Sabendo que devo priorizar elementos já existentes na TRIZ, devo descrever esta patente 
            com os dados extraídos ou usar algum conjunto de dados da TRIZ que recebi?;
            2. Caso eu utilize algum elemento já existente, devo retornar o mesmo com o `derived_from` nulo;
            3. Caso eu utilize o elemento que foi fornecido, devo retornar o mesmo e apontar o elemento 
            já existente na TRIZ que mais se aproxime a partir do `derived_from`.
            4. Neste caso, `derived_from` recebe o `id` dos metadados do elemento mais próximo

            Regras:
            - Todos os campos de texto são obrigatórios
            - O campo `derived_from` deve ser:
            - `null` se o elemento já existe na base TRIZ
            - Uma string com o `id` do elemento mais próximo existente se for derivado.
            - NÃO INVENTE o id. Ele deve ser algum dos ids dos metadados listados em seu input
            - Mantenha a terminologia consistente com os elementos TRIZ existentes
            - Use descrições claras e específicas para cada campo


            # Formato de saída
            O formato de saída deve ser um JSON com os seguintes campos:
            - `derived_from`: `id` do elemento mais próximo já existente na TRIZ.
            - `effect`: tipo de efeito (ex: "Aumento da temperatura");
            - `task`: `task` Tarefa (ex: "Aumentar a temperatura");
            - `object`: `object` Objeto (ex: "Água");
            - `physical_effect`: `physical_effect` Efeito físico (ex: "Aumento da temperatura").

            Exemplo:
            ```json
            {{
            "derived_from": "81071a34-d099-4dca-b07e-aaf120297f61", // Opcional: id da origem do registro
            "effect": "Efeito",                                     // Tipo do efeito (ex: "Mecânico", "Térmico")
            "task": "Identificação"                                 // Ação/tarefa (ex: "Comprimir", "Aquecer") 
            "object": "Gás",                                        // Objeto alvo (ex: "Água", "Metal")
            "physical_effect": "Adição de massa"                    // Efeito físico (ex: "Aumento de temperatura")
            }}
            ```
        """


    def extract(self, context: str) -> dict:
        # Implement the TRIZ extraction logic here
        response = self.openai_client.responses.parse(
            model="gpt-4.1",
            temperature=0,
            input=[
                {"role": "system", "content": self.prompt_extraction},
                {
                    "role": "user",
                    "content": context,
                },
            ],
            text_format=TrizSchema,
        )

        return response.output_parsed
    
    def vectorstore_search(self, query, k=3):
        results = self.vectorstore.similarity_search_by_vector_with_relevance_scores(
            embedding=self.embedding_model.embed_query(query),
            k=k
        )

        return results

    def derivative_decisor_ai(self, extraction: str) -> dict:
        # Implement the comparison logic here
        response = self.openai_client.responses.parse(
            model="gpt-4.1",
            temperature=0,
            input=[
                {"role": "system", "content": self.prompt_derivate_decision},
                {
                    "role": "user",
                    "content": extraction,
                },
            ],
            text_format=TrizSchemaOutput,
        )
        return response.output_parsed


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

    def derivative_decisor(self, extraction, similar_samples, threshold=0.4):
        """
        Decide se retorna o elemento mais próximo da TRIZ ou o elemento extraído,
        baseado no score de similaridade.
        O campo derived_from passa a ser o page_content do elemento mais próximo.
        """
        best_doc, best_score = min(similar_samples, key=lambda x: x[1])
        best_content = best_doc.page_content

        if self.verbose:
            print(f"Best_doc: {best_doc}")
            print(f"Best content: {best_content}\nScore: {best_score}")

        if best_score < threshold:
            # Retorna o elemento mais próximo da TRIZ
            return {
                "derived_from": None,
                "effect": best_doc.metadata["kind_effect"],
                "task": best_doc.metadata["task"],
                "object": best_doc.metadata["object"],
                "physical_effect": best_doc.metadata["physical_effect"]
            }
        else:
            # Retorna o elemento extraído, derivado do mais próximo (page_content)
            return {
                "derived_from": best_content,
                "effect": extraction.kind_effect,
                "task": extraction.task,
                "object": extraction.object,
                "physical_effect": extraction.physical_effect
            }


    def insert_derived_elements(self, result):
        """
        Insere elementos derivados no vectorstore se 'derived_from' não for None.
        """
        if result.get("derived_from"):
            # Monta o texto e metadados no padrão TRIZ
            text = f'O "{result["task"]}" é um {result["effect"]}, que no {result["object"]} causa {result["physical_effect"]}.'

            metadata = {
                "kind_effect": result["effect"],
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

    def run(self, title:str, abstract:str, threshold=0.4) -> dict:
        query = f"""Título: {title}
        Abstract: {abstract}"""

        if self.verbose:
            print("Query to be processed\n", query, "\n**********\n\n")

        extraction = self.extract(query)

        if self.verbose:
            print("Extraction result\n", extraction, "\n**********\n\n")

        similar_samples = self.vectorstore_search(query=query)

        if self.verbose:
            print("Similar samples result\n", similar_samples, "\n**********\n\n")

        result = self.derivative_decisor(extraction, similar_samples, threshold=threshold)
        if self.verbose:
            print("result\n", result, "\n**********\n\n")

        # Chama o método para inserir elemento derivado, se necessário
        self.insert_derived_elements(result)

        return result