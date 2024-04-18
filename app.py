import solara
from typing import Any, Callable, Optional, TypeVar, Union, cast, overload, List
from typing_extensions import TypedDict
import time
import ipyvue
import reacton
from solara.alias import rv as v
import os
import anthropic
from pydantic import BaseModel, Field
from graphviz import Digraph
import json
from datasets import load_dataset

# NEEDED FOR INPUT TEXT AREA INSTEAD OF INPUT TEXT
def use_change(el: reacton.core.Element, on_value: Callable[[Any], Any], enabled=True):
    """Trigger a callback when a blur events occurs or the enter key is pressed."""
    on_value_ref = solara.use_ref(on_value)
    on_value_ref.current = on_value
    def add_events():
        def on_change(widget, event, data):
            if enabled:
                on_value_ref.current(widget.v_model)
        widget = cast(ipyvue.VueWidget, solara.get_widget(el))
        if enabled:
            widget.on_event("blur", on_change)
            widget.on_event("keyup.enter", on_change)
        def cleanup():
            if enabled:
                widget.on_event("blur", on_change, remove=True)
                widget.on_event("keyup.enter", on_change, remove=True)
        return cleanup
    solara.use_effect(add_events, [enabled])

@solara.component
def InputTextarea(
    label: str,
    value: Union[str, solara.Reactive[str]] = "",
    on_value: Callable[[str], None] = None,
    disabled: bool = False,
    password: bool = False,
    continuous_update: bool = False,
    error: Union[bool, str] = False,
    message: Optional[str] = None,
):
    reactive_value = solara.use_reactive(value, on_value)
    del value, on_value
    def set_value_cast(value):
        reactive_value.value = str(value)
    def on_v_model(value):
        if continuous_update:
            set_value_cast(value)
    messages = []
    if error and isinstance(error, str):
        messages.append(error)
    elif message:
        messages.append(message)
    text_area = v.Textarea(
        v_model=reactive_value.value,
        on_v_model=on_v_model,
        label=label,
        disabled=disabled,
        type="password" if password else None,
        error=bool(error),
        messages=messages,
        solo=True,
        hide_details=True,
        outlined=True,
        rows=1,
        auto_grow=True,
    )
    use_change(text_area, set_value_cast, enabled=not continuous_update)
    return text_area

# EXTRACTION
client = anthropic.Anthropic()

class Node(BaseModel):
    id: int
    label: str
    color: str

class Edge(BaseModel):
    source: int
    target: int
    label: str
    color: str = "black"

class KnowledgeGraph(BaseModel):
    nodes: List[Node] = Field(description="Nodes in the knowledge graph")
    edges: List[Edge] = Field(description="Edges in the knowledge graph")

class MessageDict(TypedDict):
    role: str
    content: str

def add_chunk_to_ai_message(chunk: str):
    messages.value = [
        *messages.value[:-1],
        {
            "role": "assistant",
            "content": chunk,
        },
    ]

import ast

def is_valid_graph(graph):
    # Check if the graph has at least one node
    if not graph.get("nodes"):
        return False

    # Check if each node has a non-empty label
    for node in graph["nodes"]:
        if not node.get("label"):
            return False

    # Check if each edge has a source, target, and non-empty label
    for edge in graph.get("edges", []):
        if not edge.get("source") or not edge.get("target") or not edge.get("label"):
            return False

    return True

# DISPLAYED OUTPUT
@solara.component
def ChatInterface():
    with solara.lab.ChatBox():
        if len(messages.value) > 0:
            if messages.value[-1]["role"] != "user":
                obj = messages.value[-1]["content"]
                if f"{obj}" != "":
                    obj = ast.literal_eval(f"{obj}")
                    if is_valid_graph(obj):
                        dot = Digraph(comment="Knowledge Graph")
                        for node in obj["nodes"]:
                            dot.node(
                                name=str(node["id"]),
                                label=node["label"],
                                color=node["color"]
                            )
                        for edge in obj.get("edges", []):
                            dot.edge(
                                tail_name=str(edge["source"]),
                                head_name=str(edge["target"]),
                                label=edge["label"],
                                color=edge["color"]
                            )
                        with solara.Card():
                            solara.display(dot)
                    else:
                        solara.Markdown("The generated graph is not valid.")

messages: solara.Reactive[List[MessageDict]] = solara.reactive([])
aux = solara.reactive("")
text_block = solara.reactive("")
notes = solara.reactive("")
dataset_repo = solara.reactive("")
dataset_column = solara.reactive("")
keep_conv_history = solara.reactive(False)
stop_processing = solara.reactive(False)

@solara.component
def Page():
    title = "Knowledge Graph Generator"
    with solara.Head():
        solara.Title(f"{title}")
    with solara.Column(style={"width": "100%", "padding": "50px"}):
        solara.Markdown(f"#{title}")
        solara.Markdown("Enter a dataset repository and column, and the language model will process each example in the dataset. Done with :heart: by [alonsosilva](https://twitter.com/alonsosilva)")
        user_message_count = len([m for m in messages.value if m["role"] == "user"])
        def send():
            if not stop_processing.value:
                if dataset_repo.value and dataset_column.value:
                    dataset = load_dataset(dataset_repo.value)
                    for example in dataset["train"]:
                        if stop_processing.value:
                            break
                        text_block.value = example[dataset_column.value]
                        messages.value = [*messages.value, {"role": "user", "content": text_block.value}]
                        solara.sleep(1)  # Wait for 1 second before processing the next example
        def response(message):
            response_text = ""
            if keep_conv_history.value:
                conv_history = messages.value[-10:] if len(messages.value) > 10 else messages.value
            else:
                conv_history = messages.value[-1:]
            
            response = client.completions.create(
                prompt=f"Here are the notes from the previous conversation:\n{notes.value}\n\n{message}",
                model="claude-3-opus-20240229",
                max_tokens_to_sample=1024,
                stop_sequences=[anthropic.HUMAN_PROMPT],
                stream=True,
            )
            
            for data in response:
                response_text += data.completion
                solara.Markdown(data.completion)
            
            messages.value = [*messages.value, {"role": "assistant", "content": response_text}]
            
            graph_response = client.completions.create(
                prompt=f"Create a highly complex knowledge graph demonstrating all of the interrelationships and logical structures of the following prompt and response:\n\nPrompt: {text_block.value}\n\nResponse: {response_text}\n\nThe response should be in the following JSON format:\n```json\n{{\n  \"nodes\": [\n    {{\n      \"id\": 1,\n      \"label\": \"Node 1\",\n      \"color\": \"red\"\n    }},\n    {{\n      \"id\": 2,\n      \"label\": \"Node 2\",\n      \"color\": \"blue\"\n    }}\n  ],\n  \"edges\": [\n    {{\n      \"source\": 1,\n      \"target\": 2,\n      \"label\": \"Edge 1\",\n      \"color\": \"black\"\n    }}\n  ]\n}}\n```",
                model="claude-3-opus-20240229",
                max_tokens_to_sample=1024,
                stop_sequences=[anthropic.HUMAN_PROMPT],
                stream=True,
            )
            
            graph_code = ""
            for data in graph_response:
                graph_code += data.completion
                obj = data.completion.strip()
                if f"{obj}" != aux.value:
                    add_chunk_to_ai_message(f"{obj}")
                    aux.value = f"{obj}"
            
            notes_response = client.completions.create(
                prompt=f"Here are your notes from the previous conversation:\n{notes.value}\n\nBased on the following prompt and response, update your notes with important details and add any relevant tasks to the todo list:\n\nPrompt: {text_block.value}\n\nResponse: {response_text}",
                model="claude-3-opus-20240229",
                max_tokens_to_sample=1024,
                stop_sequences=[anthropic.HUMAN_PROMPT],
            )
            
            notes.value = notes_response.completion
            
            with open("conversation_log.jsonl", "a") as f:
                json.dump({"input": text_block.value, "response": response_text, "graph_code": graph_code, "notes": notes.value}, f)
                f.write("\n")
        def result():
            if messages.value != []:
                if messages.value[-1]["role"] == "user":
                    response(messages.value[-1]["content"])
        result = solara.lab.use_task(result, dependencies=[user_message_count])
        solara.TextField("Dataset Repository:", value=dataset_repo)
        solara.TextField("Dataset Column:", value=dataset_column)
        solara.Checkbox("Keep Conversation History", value=keep_conv_history)
        solara.Button(label="Start Processing", on_click=send)
        solara.Button(label="Stop Processing", on_click=lambda: stop_processing.set(True))
        ChatInterface()

Page()
