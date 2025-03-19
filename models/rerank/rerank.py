import json
from typing import Optional
from dify_plugin.entities.model.rerank import RerankDocument, RerankResult
from dify_plugin.errors.model import (
    CredentialsValidateFailedError,
    InvokeError,
)
from dify_plugin.interfaces.model.rerank_model import RerankModel

from tencentcloud.common import credential
from tencentcloud.common.exception import TencentCloudSDKException
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.lkeap.v20240522 import lkeap_client, models


class LkeapRerankModel(RerankModel):
    """
    Model class for LKEAP rerank model.
    """

    def _invoke(
        self,
        model: str,
        credentials: dict,
        query: str,
        docs: list[str],
        score_threshold: Optional[float] = None,
        top_n: Optional[int] = None,
        user: Optional[str] = None,
    ) -> RerankResult:
        """
        Invoke rerank model

        :param model: model name
        :param credentials: model credentials
        :param query: search query
        :param docs: docs for reranking
        :param score_threshold: score threshold
        :param top_n: top n
        :param user: unique user id
        :return: rerank result
        """
        if len(docs) == 0:
            return RerankResult(model=model, docs=docs)

        if model != "lke-reranker-base":
            raise ValueError("Invalid model name")
        client = self._setup_lkeap_client(credentials)

        # 实例化一个请求对象,每个接口都会对应一个request对象
        req = models.RunRerankRequest()
        params = {
            "Query": query,
            "Docs": docs,
            "Model": model
        }
        req.from_json_string(json.dumps(params))

        # 返回的resp是一个RunRerankResponse的实例，与请求对象对应
        response = client.RunRerank(req)

        rerank_documents = []

        if not response.ScoreList:
            return RerankResult(model=model, docs=rerank_documents)

        ii = 0
        for _, result in enumerate(response.ScoreList):
            rerank_document = RerankDocument(
                index=ii, score=result, text=docs[ii]
            )
            ii = ii + 1
            if score_threshold is not None:
                if result >= score_threshold:
                    rerank_documents.append(rerank_document)
            else:
                rerank_documents.append(rerank_document)
        return RerankResult(model=model, docs=rerank_documents)

    def validate_credentials(self, model: str, credentials: dict) -> None:
        """
        Validate credentials
        """
        try:
            client = self._setup_lkeap_client(credentials)
            req = models.ChatCompletionsRequest()
            params = {
                "Model": model,
                "Messages": [{"Role": "user", "Content": "hello"}],
                "TopP": 1,
                "Temperature": 0,
                "Stream": False,
            }
            req.from_json_string(json.dumps(params))
            client.ChatCompletions(req)
        except Exception as e:
            raise CredentialsValidateFailedError(
                f"Credentials validation failed: {e}")

    def _setup_lkeap_client(self, credentials):
        secret_id = credentials["secret_id"]
        secret_key = credentials["secret_key"]
        cred = credential.Credential(secret_id, secret_key)
        httpProfile = HttpProfile()
        clientProfile = ClientProfile()
        clientProfile.httpProfile = httpProfile
        client = lkeap_client.LkeapClient(cred, "ap-guangzhou", clientProfile)
        return client

    @property
    def _invoke_error_mapping(self) -> dict[type[InvokeError], list[type[Exception]]]:
        """
        Map model invoke error to unified error
        The key is the error type thrown to the caller
        The value is the error type thrown by the model,
        which needs to be converted into a unified error type for the caller.

        :return: Invoke error mapping
        """
        return {InvokeError: [TencentCloudSDKException]}
