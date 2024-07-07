from mistral_common.protocol.instruct.tool_calls import Function, Tool
from mistral_inference.model import Transformer
from mistral_inference.generate import generate
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
import gradio as gr
import torch
import json
import requests

from dotenv import load_dotenv
import os

load_dotenv()


mistral_models_path = "/home/team/mistral_models/7B-Instruct-v0.3"
tokenizer = MistralTokenizer.from_file(f"{mistral_models_path}/tokenizer.model.v3")
model = Transformer.from_folder(mistral_models_path, dtype=torch.float16)


git_token = os.getenv("GITHUB_TOKEN")
git_org_token = os.getenv("GITHUB_ORGANIZATION_TOKEN")


def get_repositories(username):
    """Get repository names and links from GitHub based on username"""
    url = f"https://api.github.com/users/{username}/repos"
    response = requests.get(url)
    if response.status_code == 200:
        repos = response.json()
        repo_info = [{"name": repo["name"], "link": repo["html_url"]} for repo in repos]
        return str(repo_info)
    else:
        return "No repositories available"

def add_collaborator(owner, repo, usernames):
    """Adding multiple repository collaborators"""
    headers = {"Authorization": f"token {git_token}", "Accept": "application/vnd.github.v3+json"}
    results = {}
    if isinstance(usernames, str):
        usernames = [usernames]
    for username in usernames:
        url = f"https://api.github.com/repos/{owner}/{repo}/collaborators/{username}"
        print("url : ", url)

        response = requests.put(url, headers=headers)
        if response.status_code == 201:
            result = f"{username} was added as a collaborator to the {owner}/{repo} repository successfully"
        elif response.status_code == 204:
            result = f"{username} is already a collaborator."
        else:
            result = f"Failed to add {username}. Status code: {response.status_code}"

        results[username] = result
    return str(results)

def add_organization_collaborator(organization, repo, usernames):
    """Get repository names and links from GitHub based on username"""
    headers = {"Authorization": f"token {git_org_token}", "Accept": "application/vnd.github.v3+json"}
    results = {}
    if isinstance(usernames, str):
        usernames = [usernames]
    for username in usernames:
        url = f"https://api.github.com/repos/{organization}/{repo}/collaborators/{username}"
        print(url)
        data = {"permission":"triage"}
        response = requests.put(url, headers=headers, json= data)
        if response.status_code == 201:
            result = f"{username} was added as a outside collaborator"
        elif response.status_code == 204:
            result = f"{username} was already a collaborator or an organization member"
        else:
            result = response.status_code
        results[username] = result
    return str(results)

def remove_collaborator(owner, repo, usernames):
    """Remove a repository collaborator"""
    headers = {"Authorization": f"token {git_token}", "Accept": "application/vnd.github.v3+json"}
    results = {}

    if isinstance(usernames, str):
        usernames = [usernames]
    for username in usernames:
        url = f" https://api.github.com/repos/{owner}/{repo}/collaborators/{username}"
        print("url : ", url)
        response = requests.delete(url, headers=headers)
        if response.status_code == 204:
            result = f"The collaborator {username} was removed from the {repo}"
        else:
            result = f"Failed to remove {response.status_code}"
        results[username] = result
    return str(results)

def remove_organization_collaborator(organization, repo, usernames):
    """Remove a repository collaborator"""
    headers = {"Authorization": f"token {git_org_token}", "Accept": "application/vnd.github.v3+json"}
    results = {}

    if isinstance(usernames, str):
        usernames = [usernames]
    for username in usernames:
        url = f" https://api.github.com/repos/{organization}/{repo}/collaborators/{username}"
        print("url : ", url)
        response = requests.delete(url, headers=headers)
        if response.status_code == 204:
            result = f"The collaborator {username} was removed from the {repo}"
        else:
            result = f"Failed to remove {response.status_code}"
        results[username] = result
    return str(results)

def search_repositories(username, repo_names):
    headers = {"Authorization": f"token {git_token}", "Accept": "application/vnd.github.v3+json"}
    repositories = {}
    if isinstance(repo_names, str):
        repo_names = [repo_names]

    for repo_name in repo_names:
        url = f"https://api.github.com/repos/{username}/{repo_name}"
        print("url", url)
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            repos = response.json()
            result = repos["html_url"]
            repositories[repo_name] = result
        elif response.status_code == 301:
            repositories[repo_name] = "Moved permanently"
        else : 
            repositories[repo_name] = f"{repo_name} is not found"

    return str(repositories)


def run_conversation(input, history):
    completion_request = ChatCompletionRequest(
        tools=[
            Tool(
                function=Function(
                    name="get_repositories",
                    description="Get repository names and links from GitHub based on username",
                    parameters={
                        "type": "object",
                        "properties": {
                            "username": {
                                "type": "string",
                                "description": "The GitHub username of the repository owner.",
                            },
                        },
                        "required": ["username"],
                    },
                ),
            ),
            Tool(
                function=Function(
                    name="remove_collaborator",
                    description="Remove collaborators from a GitHub repository",
                    parameters={
                        "type": "object",
                        "properties": {
                            "owner": {
                                "type": "string",
                                "description": "The GitHub username of the repository owner.",
                            },
                            "repo": {"type": "string", "description": "The name of the repository from which the collaborator will be removed."},
                            "usernames": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "The GitHub username of the collaborator to be removed.",
                            },
                        },
                        "required": ["owner", "repo", "usernames"],
                    },
                ),
            ),
            Tool(
                function=Function(
                    name="remove_organization_collaborator",
                    description="Remove collaborators from a GitHub repository",
                    parameters={
                        "type": "object",
                        "properties": {
                            "organization": {
                                "type": "string",
                                "description": "The GitHub username of the organization.",
                            },
                            "repo": {"type": "string", "description": "The name of the repository from which the collaborator will be removed."},
                            "usernames": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "The GitHub username of the collaborator to be removed from organization.",
                            },
                        },
                        "required": ["organization", "repo", "usernames"],
                    },
                ),
            ),
            Tool(
                function=Function(
                    name="add_collaborator",
                    description="Add collaborators to a GitHub repository",
                    parameters={
                        "type": "object",
                        "properties": {
                            "owner": {
                                "type": "string",
                                "description": "The GitHub username of the repository owner.",
                            },
                            "repo": {
                                "type": "string", 
                                "description": "The name of the repository to which the collaborator will be added"
                            },
                            "usernames": {
                                "type": "array", 
                                "items": {"type": "string"}, 
                                "description": "The GitHub username of the collaborator to be added."
                            },
                        },
                        "required": ["owner", "repo", "usernames"],
                    },
                ),
            ),
            Tool(
                function=Function(
                    name="add_organization_collaborator",
                    description="Add organization collaborators to a GitHub repository",
                    parameters={
                        "type": "object",
                        "properties": {
                            "organization": {
                                "type": "string",
                                "description": "The GitHub username of the organization.",
                            },
                            "repo": {
                                "type": "string", 
                                "description": "The name of the repository to which the collaborator will be added"
                            },
                            "usernames": {
                                "type": "array", 
                                "items": {"type": "string"}, 
                                "description": "The GitHub username of the collaborator to be added."
                            },
                        },
                        "required": ["organization", "repo", "usernames"],
                    },
                ),
            ),
            Tool(
                function=Function(
                    name="search_repositories",
                    description="Searches the repositories.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "username": {
                                "type": "string",
                                "description": "The GitHub username of the repository owner.",
                            },
                            "repo_names": {
                                "type": "array", 
                                "items": {"type": "string"}, 
                                "description": "The name of the repositories to be searched."
                            },
                        },
                        "required": ["username", "repo_names"],
                    },
                ),
            ),
        ],
        messages=[UserMessage(content=input)],
    )

    tokens = tokenizer.encode_chat_completion(completion_request).tokens

    out_tokens, _ = generate([tokens], model, max_tokens=500, temperature=0.0, eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id)
    tool_calls = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])
    print(type(tool_calls))
    print(tool_calls)
    check_change = tool_calls
    print(type(check_change))
    tool_calls_lst = []
    for tool_call in tool_calls.split('\n'):
        tool_call = tool_call.strip()
        if tool_call:
            try:
                parsed_call = json.loads(tool_call)
                if isinstance(parsed_call, list):
                    tool_calls_lst.extend(parsed_call)
                else:
                    tool_calls_lst.append(parsed_call)
            except json.JSONDecodeError as e:
                print(f"JSONDecodeError: {e}")
                print(f"Problematic JSON string: {tool_call}")

    if not tool_calls_lst:
        return []

    available_functions = {
        "get_repositories": get_repositories,
        "add_collaborator": add_collaborator,
        "add_organization_collaborator": add_organization_collaborator,
        "remove_collaborator": remove_collaborator,
        "remove_organization_collaborator": remove_organization_collaborator,
        "search_repositories": search_repositories,
    }

    for tool_call in tool_calls_lst:
        function_name = tool_call["name"]
        print(f"{function_name=}")
        function_to_call = available_functions[function_name]
        print(f"{function_to_call=}")
        function_args = tool_call["arguments"]
        print(f"{function_args=}")
        function_response = function_to_call(**function_args)
        print(function_response)
    return function_response


iface = gr.ChatInterface(fn=run_conversation, title="Github_Manager", description="Enter your message and get a response")

iface.launch(server_name="0.0.0.0")