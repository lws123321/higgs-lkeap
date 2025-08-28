import json
import logging
from collections.abc import Generator
from typing import cast
from dify_plugin.entities.model.llm import LLMResult, LLMResultChunk, LLMResultChunkDelta
from dify_plugin.entities.model.message import (
    AssistantPromptMessage,
    ImagePromptMessageContent,
    PromptMessage,
    PromptMessageContentType,
    PromptMessageTool,
    SystemPromptMessage,
    TextPromptMessageContent,
    ToolPromptMessage,
    UserPromptMessage,
)
from dify_plugin.errors.model import CredentialsValidateFailedError, InvokeError
from dify_plugin.interfaces.model.large_language_model import LargeLanguageModel
from tencentcloud.common import credential
from tencentcloud.common.exception import TencentCloudSDKException
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.lkeap.v20240522 import lkeap_client, models

logger = logging.getLogger(__name__)


class LkeapLargeLanguageModel(LargeLanguageModel):
    def _invoke(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        model_parameters: dict,
        tools: list[PromptMessageTool] | None = None,
        stop: list[str] | None = None,
        stream: bool = True,
        user: str | None = None,
    ) -> LLMResult | Generator:
        client = self._setup_lkeap_client(credentials)
        request = models.ChatCompletionsRequest()
        messages_dict = self._convert_prompt_messages_to_dicts(prompt_messages)
        custom_parameters = {
            "Temperature": model_parameters.get("temperature", 0.6),
            "MaxTokens": model_parameters.get("max_tokens", 4096),
            "EnableSearch": model_parameters.get("enable_search", False),
            "Thinking": model_parameters.get("thinking", False)
        }
        params = {
            "Model": model,
            "Messages": messages_dict,
            "Stop": stop,
            "Stream": stream,
            **custom_parameters
        }

        request.from_json_string(json.dumps(params))
        response = client.ChatCompletions(request)

        if stream:
            return self._handle_stream_chat_response(model, credentials, prompt_messages, response)

        return self._handle_chat_response(credentials, model, prompt_messages, response)

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

    def _convert_prompt_messages_to_dicts(self, prompt_messages: list[PromptMessage]) -> list[dict]:
        """Convert a list of PromptMessage objects to a list of dictionaries with 'Role' and 'Content' keys."""
        dict_list = []
        for message in prompt_messages:
            if isinstance(message, AssistantPromptMessage):
                tool_calls = message.tool_calls
                if tool_calls and len(tool_calls) > 0:
                    dict_tool_calls = [
                        {
                            "Id": tool_call.id,
                            "Type": tool_call.type,
                            "Function": {
                                "Name": tool_call.function.name,
                                "Arguments": tool_call.function.arguments
                                if tool_call.function.arguments == ""
                                else "{}",
                            },
                        }
                        for tool_call in tool_calls
                    ]
                    dict_list.append(
                        {"Role": message.role.value, "Content": " ", "ToolCalls": dict_tool_calls})
                else:
                    dict_list.append(
                        {"Role": message.role.value, "Content": message.content})
            elif isinstance(message, ToolPromptMessage):
                tool_execute_result = {"result": message.content}
                content = json.dumps(tool_execute_result, ensure_ascii=False)
                dict_list.append(
                    {"Role": message.role.value, "Content": content, "ToolCallId": message.tool_call_id})
            elif isinstance(message, UserPromptMessage):
                message = cast(UserPromptMessage, message)
                if isinstance(message.content, str):
                    dict_list.append(
                        {"Role": message.role.value, "Content": message.content})
                else:
                    sub_messages = []
                    for message_content in message.content:
                        if message_content.type == PromptMessageContentType.TEXT:
                            message_content = cast(
                                TextPromptMessageContent, message_content)
                            sub_message_dict = {
                                "Type": "text", "Text": message_content.data}
                            sub_messages.append(sub_message_dict)
                        elif message_content.type == PromptMessageContentType.IMAGE:
                            message_content = cast(
                                ImagePromptMessageContent, message_content)
                            sub_message_dict = {"Type": "image_url", "ImageUrl": {
                                "Url": message_content.data}}
                            sub_messages.append(sub_message_dict)
                    dict_list.append(
                        {"Role": message.role.value, "Contents": sub_messages})
            else:
                dict_list.append({"Role": message.role.value,
                                 "Content": message.content})
        return dict_list

    def _handle_stream_chat_response(self, model, credentials, prompt_messages, resp):

        is_reasoning = False
        tool_call = None
        tool_calls = []

        for index, event in enumerate(resp):
            logging.debug("_handle_stream_chat_response, event: %s", event)
            data_str = event["data"]
            if data_str == "[DONE]":
                continue
            data = json.loads(data_str)
            choices = data.get("Choices", [])
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("Delta", {})
            message_content = delta.get("Content", "")

            message_content, is_reasoning = self._wrap_thinking_by_reasoning_content(
                delta, is_reasoning
            )

            finish_reason = choice.get("FinishReason", "")
            usage = data.get("Usage", {})
            prompt_tokens = usage.get("PromptTokens", 0)
            completion_tokens = usage.get("CompletionTokens", 0)
            response_tool_calls = delta.get("ToolCalls")
            if response_tool_calls is not None:
                new_tool_calls = self._extract_response_tool_calls(
                    response_tool_calls)
                if len(new_tool_calls) > 0:
                    new_tool_call = new_tool_calls[0]
                    if tool_call is None:
                        tool_call = new_tool_call
                    elif tool_call.id != new_tool_call.id:
                        tool_calls.append(tool_call)
                        tool_call = new_tool_call
                    else:
                        tool_call.function.name += new_tool_call.function.name
                        tool_call.function.arguments += new_tool_call.function.arguments
                if (
                    tool_call is not None
                    and len(tool_call.function.name) > 0
                    and (len(tool_call.function.arguments) > 0)
                ):
                    tool_calls.append(tool_call)
                    tool_call = None
            assistant_prompt_message = AssistantPromptMessage(
                content=message_content, tool_calls=[])
            if len(tool_calls) > 0:
                assistant_prompt_message.content = ""
            if finish_reason == "tool_calls":
                assistant_prompt_message.tool_calls = tool_calls
                tool_call = None
                tool_calls = []
            if len(finish_reason) > 0:
                usage = self._calc_response_usage(
                    model, credentials, prompt_tokens, completion_tokens)
                delta_chunk = LLMResultChunkDelta(
                    index=index,
                    role=delta.get("Role", "assistant"),
                    message=assistant_prompt_message,
                    usage=usage,
                    finish_reason=finish_reason,
                )
                tool_call = None
                tool_calls = []
            else:
                delta_chunk = LLMResultChunkDelta(
                    index=index, message=assistant_prompt_message)

            yield LLMResultChunk(model=model, prompt_messages=prompt_messages, delta=delta_chunk)

    def _handle_chat_response(self, credentials, model, prompt_messages, response):
        usage = self._calc_response_usage(
            model, credentials, response.Usage.PromptTokens, response.Usage.CompletionTokens
        )
        assistant_prompt_message = AssistantPromptMessage()
        assistant_prompt_message.content = response.Choices[0].Message.Content
        result = LLMResult(model=model, prompt_messages=prompt_messages,
                           message=assistant_prompt_message, usage=usage)
        return result

    def get_num_tokens(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        tools: list[PromptMessageTool] | None = None,
    ) -> int:
        if len(prompt_messages) == 0:
            return 0
        prompt = self._convert_messages_to_prompt(prompt_messages)
        return self._get_num_tokens_by_gpt2(prompt)

    def _convert_messages_to_prompt(self, messages: list[PromptMessage]) -> str:
        """
        Format a list of messages into a full prompt for the Anthropic model

        :param messages: List of PromptMessage to combine.
        :return: Combined string with necessary human_prompt and ai_prompt tags.
        """
        messages = messages.copy()
        text = "".join((self._convert_one_message_to_text(message)
                       for message in messages))
        return text.rstrip()

    def _convert_one_message_to_text(self, message: PromptMessage) -> str:
        """
        Convert a single message to a string.

        :param message: PromptMessage to convert.
        :return: String representation of the message.
        """
        human_prompt = "\n\nHuman:"
        ai_prompt = "\n\nAssistant:"
        tool_prompt = "\n\nTool:"
        content = message.content
        if isinstance(message, UserPromptMessage):
            message_text = f"{human_prompt} {content}"
        elif isinstance(message, AssistantPromptMessage):
            message_text = f"{ai_prompt} {content}"
        elif isinstance(message, ToolPromptMessage):
            message_text = f"{tool_prompt} {content}"
        elif isinstance(message, SystemPromptMessage):
            message_text = content
        else:
            raise ValueError(f"Got unknown type {message}")
        return message_text

    def _wrap_thinking_by_reasoning_content(self, delta: dict, is_reasoning: bool) -> tuple[str, bool]:
        """
        If the reasoning response is from delta.get("reasoning_content"), we wrap
        it with HTML think tag.
        :param delta: delta dictionary from LLM streaming response
        :param is_reasoning: is reasoning
        :return: tuple of (processed_content, is_reasoning)
        """

        content = delta.get("Content", "")
        reasoning_content = delta.get("ReasoningContent")

        if reasoning_content:
            if not is_reasoning:
                content = "<think>\n" + reasoning_content
                is_reasoning = True
            else:
                content = reasoning_content
        elif is_reasoning and content:
            content = "\n</think>" + content
            is_reasoning = False
        return content, is_reasoning

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
