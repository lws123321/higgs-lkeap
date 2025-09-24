import json
import logging
import requests
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
        """
        调用大语言模型进行对话
        :param model: 模型名称
        :param credentials: 认证信息
        :param prompt_messages: 提示消息列表
        :param model_parameters: 模型参数
        :param tools: 工具列表
        :param stop: 停止词列表
        :param stream: 是否流式输出
        :param user: 用户标识
        :return: LLM结果或生成器
        """
        # 仅对deepseek-v3.1模型使用HTTP请求逻辑，确保Thinking参数正确传递
        if model == "deepseek-v3.1":
            return self._invoke_with_http(model, credentials, prompt_messages, model_parameters, tools, stop, stream, user)
        else:
            return self._invoke_with_sdk(model, credentials, prompt_messages, model_parameters, tools, stop, stream, user)

    def _invoke_with_http(
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
        """
        使用HTTP请求调用deepseek-v3.1模型，确保Thinking参数正确传递
        :param model: 模型名称
        :param credentials: 认证信息
        :param prompt_messages: 提示消息列表
        :param model_parameters: 模型参数
        :param tools: 工具列表
        :param stop: 停止词列表
        :param stream: 是否流式输出
        :param user: 用户标识
        :return: LLM结果或生成器
        """
        messages_dict = self._convert_prompt_messages_to_openai_format(prompt_messages)
        thinking_type = "enabled" if model_parameters.get("thinking", False) else "disabled"
        
        # 构造请求参数，遵循OpenAI标准格式（小写参数名）
        params = {
            "model": model,
            "messages": messages_dict,
            "stream": stream,
            "temperature": model_parameters.get("temperature", 0.6),
            "max_tokens": model_parameters.get("max_tokens", 4096),
            "enable_search": model_parameters.get("enable_search", False),
            "thinking": {"type": thinking_type}
        }
        
        if stop:
            params["stop"] = stop

        # 使用直接HTTP请求
        response = self._make_http_request(credentials, params)

        if stream:
            return self._handle_stream_http_response(model, credentials, prompt_messages, response)

        return self._handle_http_response(credentials, model, prompt_messages, response)

    def _invoke_with_sdk(
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
        """
        使用SDK调用其他模型（原有逻辑）
        :param model: 模型名称
        :param credentials: 认证信息
        :param prompt_messages: 提示消息列表
        :param model_parameters: 模型参数
        :param tools: 工具列表
        :param stop: 停止词列表
        :param stream: 是否流式输出
        :param user: 用户标识
        :return: LLM结果或生成器
        """
        client = self._setup_lkeap_client(credentials)
        request = models.ChatCompletionsRequest()
        messages_dict = self._convert_prompt_messages_to_dicts(prompt_messages)
        thinking_type = "enabled" if model_parameters.get("thinking", False) else "disabled"
        custom_parameters = {
            "Temperature": model_parameters.get("temperature", 0.6),
            "MaxTokens": model_parameters.get("max_tokens", 4096),
            "EnableSearch": model_parameters.get("enable_search", False),
            "Thinking": {"type": thinking_type}
        }
        params = {
            "Model": model,
            "Messages": messages_dict,
            "Stop": stop,
            "Stream": stream,
            **custom_parameters
        }

        request.from_json_string(json.dumps(params, ensure_ascii=False))
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
                "Thinking": {"type": "disabled"}
            }
            req.from_json_string(json.dumps(params, ensure_ascii=False))
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

    def _make_http_request(self, credentials: dict, params: dict):
        """
        使用HTTP请求直接调用LKEAP API，遵循腾讯云OpenAI兼容接口规范
        :param credentials: 认证信息，包含secret_key作为API key
        :param params: 请求参数
        :return: 响应对象
        """
        url = "https://api.lkeap.cloud.tencent.com/v1/chat/completions"
        
        # 根据腾讯云文档，使用secret_key作为Bearer token
        # secret_key = credentials.get("secret_key")
        secret_key = "sk-lETsTqLHSwaYwj5qOUpewmoqLVuoba82THLeOgoefaqWRsUA"
        if not secret_key:
            raise InvokeError("Missing secret_key in credentials")
        
        headers = {
            'Authorization': f'Bearer {secret_key}',
            'Content-Type': 'application/json; charset=utf-8',
            'Accept': 'application/json',
            'Accept-Charset': 'utf-8'
        }
        
        try:
            response = requests.post(
                url,
                headers=headers,
                json=params,
                stream=params.get("stream", False),
                timeout=120
            )
            response.raise_for_status()
            # 确保响应使用UTF-8编码
            response.encoding = 'utf-8'
            return response
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP request failed with status {response.status_code}: {e}")
            if response.status_code == 401:
                raise CredentialsValidateFailedError("Authentication failed: Invalid secret_key")
            elif response.status_code == 403:
                raise InvokeError("Access denied: Please check your permissions")
            else:
                raise InvokeError(f"API request failed: {e}")
        except Exception as e:
            logger.error(f"HTTP request failed: {e}")
            raise InvokeError(f"API request failed: {e}")

    def _handle_http_response(self, credentials, model, prompt_messages, response):
        """
        处理非流式HTTP响应
        :param credentials: 认证信息
        :param model: 模型名称  
        :param prompt_messages: 提示消息列表
        :param response: HTTP响应对象
        :return: LLM结果
        """
        try:
            # 确保JSON正确解析UTF-8编码的内容
            response.encoding = 'utf-8'
            data = response.json()
            choices = data.get("choices", [])
            if not choices:
                raise InvokeError("No choices in response")
            
            choice = choices[0]
            message = choice.get("message", {})
            content = message.get("content", "")
            
            usage_data = data.get("usage", {})
            usage = self._calc_response_usage(
                model, credentials, 
                usage_data.get("prompt_tokens", 0), 
                usage_data.get("completion_tokens", 0)
            )
            
            assistant_prompt_message = AssistantPromptMessage(content=content)
            return LLMResult(
                model=model, 
                prompt_messages=prompt_messages,
                message=assistant_prompt_message, 
                usage=usage
            )
        except Exception as e:
            logger.error(f"Failed to parse response: {e}")
            raise InvokeError(f"Failed to parse response: {e}")

    def _handle_stream_http_response(self, model, credentials, prompt_messages, response):
        """
        处理流式HTTP响应
        :param model: 模型名称
        :param credentials: 认证信息
        :param prompt_messages: 提示消息列表
        :param response: HTTP响应对象
        :return: 生成器
        """
        is_reasoning = False
        tool_call = None
        tool_calls = []
        index = 0

        try:
            # 确保流式响应正确处理UTF-8编码
            for line in response.iter_lines(decode_unicode=True, chunk_size=None):
                if not line or not line.strip():
                    continue
                line = line.strip()
                if not line.startswith('data: '):
                    continue
                
                data_str = line[6:]  # 移除 'data: ' 前缀
                if data_str == "[DONE]":
                    continue
                
                try:
                    # 解析JSON数据
                    data = json.loads(data_str)
                    choices = data.get("choices", [])
                    if not choices:
                        continue
                    
                    choice = choices[0]
                    delta = choice.get("delta", {})
                    message_content = delta.get("content", "")

                    message_content, is_reasoning = self._wrap_thinking_by_reasoning_content(
                        delta, is_reasoning
                    )

                    finish_reason = choice.get("finish_reason", "")
                    usage = data.get("usage", {})
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                    
                    response_tool_calls = delta.get("tool_calls")
                    if response_tool_calls is not None:
                        new_tool_calls = self._extract_response_tool_calls(response_tool_calls)
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
                        
                        if (tool_call is not None and len(tool_call.function.name) > 0 
                            and len(tool_call.function.arguments) > 0):
                            tool_calls.append(tool_call)
                            tool_call = None
                    
                    assistant_prompt_message = AssistantPromptMessage(
                        content=message_content, tool_calls=[]
                    )
                    
                    if len(tool_calls) > 0:
                        assistant_prompt_message.content = ""
                    
                    if finish_reason == "tool_calls":
                        assistant_prompt_message.tool_calls = tool_calls
                        tool_call = None
                        tool_calls = []
                    
                    if len(finish_reason) > 0:
                        usage = self._calc_response_usage(
                            model, credentials, prompt_tokens, completion_tokens
                        )
                        delta_chunk = LLMResultChunkDelta(
                            index=index,
                            role=delta.get("role", "assistant"),
                            message=assistant_prompt_message,
                            usage=usage,
                            finish_reason=finish_reason,
                        )
                        tool_call = None
                        tool_calls = []
                    else:
                        delta_chunk = LLMResultChunkDelta(
                            index=index, message=assistant_prompt_message
                        )

                    yield LLMResultChunk(
                        model=model, 
                        prompt_messages=prompt_messages, 
                        delta=delta_chunk
                    )
                    index += 1
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON: {e}, line: {data_str}")
                    continue
                    
        except Exception as e:
            logger.error(f"Stream processing failed: {e}")
            raise InvokeError(f"Stream processing failed: {e}")

    def _convert_prompt_messages_to_openai_format(self, prompt_messages: list[PromptMessage]) -> list[dict]:
        """
        将PromptMessage对象列表转换为OpenAI标准格式的字典列表
        使用小写的'role'和'content'键，符合腾讯云OpenAI兼容接口规范
        """
        message_list = []
        for message in prompt_messages:
            if isinstance(message, AssistantPromptMessage):
                tool_calls = message.tool_calls
                if tool_calls and len(tool_calls) > 0:
                    dict_tool_calls = [
                        {
                            "id": tool_call.id,
                            "type": tool_call.type,
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments
                                if tool_call.function.arguments != ""
                                else "{}",
                            },
                        }
                        for tool_call in tool_calls
                    ]
                    message_list.append(
                        {"role": message.role.value, "content": " ", "tool_calls": dict_tool_calls}
                    )
                else:
                    message_list.append(
                        {"role": message.role.value, "content": message.content}
                    )
            elif isinstance(message, ToolPromptMessage):
                tool_execute_result = {"result": message.content}
                content = json.dumps(tool_execute_result, ensure_ascii=False, separators=(',', ':'))
                message_list.append(
                    {"role": message.role.value, "content": content, "tool_call_id": message.tool_call_id}
                )
            elif isinstance(message, UserPromptMessage):
                message = cast(UserPromptMessage, message)
                if isinstance(message.content, str):
                    message_list.append(
                        {"role": message.role.value, "content": message.content}
                    )
                else:
                    sub_messages = []
                    for message_content in message.content:
                        if message_content.type == PromptMessageContentType.TEXT:
                            message_content = cast(TextPromptMessageContent, message_content)
                            sub_message_dict = {
                                "type": "text", 
                                "text": message_content.data
                            }
                            sub_messages.append(sub_message_dict)
                        elif message_content.type == PromptMessageContentType.IMAGE:
                            message_content = cast(ImagePromptMessageContent, message_content)
                            sub_message_dict = {
                                "type": "image_url", 
                                "image_url": {"url": message_content.data}
                            }
                            sub_messages.append(sub_message_dict)
                    message_list.append(
                        {"role": message.role.value, "content": sub_messages}
                    )
            else:
                message_list.append(
                    {"role": message.role.value, "content": message.content}
                )
        return message_list

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
                content = json.dumps(tool_execute_result, ensure_ascii=False, separators=(',', ':'))
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

        content = delta.get("content", "")
        reasoning_content = delta.get("reasoning_content")

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
