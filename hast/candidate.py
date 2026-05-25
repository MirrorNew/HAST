# -*- coding: utf-8 -*-
"""Candidate code extraction, validation, and execution."""

from __future__ import annotations

import ast
import hashlib
import math
import re
import textwrap
from dataclasses import dataclass
from typing import Any, Callable

import networkx as nx
import numpy as np

from baselines.algorithms import complete_order

ALLOWED_IMPORT_ROOTS = {
    "math",
    "heapq",
    "random",
    "itertools",
    "collections",
    "networkx",
    "numpy",
}

FORBIDDEN_TOKENS = [
    "__import__",
    "open(",
    "exec(",
    "eval(",
    "compile(",
    "input(",
    "globals(",
    "locals(",
    "subprocess",
    "socket",
    "requests",
    "urllib",
    "shutil",
    "pathlib",
    "pickle",
    "marshal",
    "ctypes",
    "multiprocessing",
    "threading",
    "os.",
    "sys.",
    "write(",
    "rmdir(",
    "unlink(",
]


@dataclass
class CandidateProgram:
    candidate_id: str
    code: str
    family: str = "unknown"
    source_stage: str = "unknown"


def stable_hash(text: str, n: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def extract_code(response_text: str) -> str:
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", response_text, flags=re.DOTALL | re.IGNORECASE)
    for block in blocks:
        if "def degree_order" in block:
            return block.strip()
    idx = response_text.find("def degree_order")
    if idx >= 0:
        return response_text[idx:].strip()
    return response_text.strip()


def validate_code(code: str) -> str:
    code = textwrap.dedent(code).strip()
    if "def degree_order" not in code:
        raise ValueError("missing degree_order(G)")
    lowered = code.lower()
    for token in FORBIDDEN_TOKENS:
        if token.lower() in lowered:
            raise ValueError(f"forbidden token: {token}")
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for name in names:
                root = name.split(".")[0]
                if root not in ALLOWED_IMPORT_ROOTS:
                    raise ValueError(f"import not allowed: {name}")
    return code


def compile_candidate(program: CandidateProgram) -> Callable[[nx.Graph], list[Any]]:
    code = validate_code(program.code)

    def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        del globals, locals, level
        root = name.split(".")[0]
        if root not in ALLOWED_IMPORT_ROOTS:
            raise ImportError(f"import not allowed: {name}")
        return __import__(name, fromlist=fromlist)

    namespace: dict[str, Any] = {
        "math": math,
        "np": np,
        "numpy": np,
        "nx": nx,
        "networkx": nx,
        "__builtins__": {
            "abs": abs,
            "all": all,
            "any": any,
            "bool": bool,
            "dict": dict,
            "enumerate": enumerate,
            "float": float,
            "int": int,
            "iter": iter,
            "len": len,
            "list": list,
            "max": max,
            "min": min,
            "next": next,
            "range": range,
            "reversed": reversed,
            "round": round,
            "set": set,
            "sorted": sorted,
            "str": str,
            "sum": sum,
            "tuple": tuple,
            "zip": zip,
            "__import__": safe_import,
        },
    }
    exec(compile(code, f"<candidate:{program.candidate_id}>", "exec"), namespace, namespace)
    fn = namespace.get("degree_order")
    if not callable(fn):
        raise ValueError("degree_order is not callable")

    def runner(graph: nx.Graph) -> list[Any]:
        return complete_order(graph, fn(graph.copy()))

    return runner


def make_program(code: str, family: str = "unknown", source_stage: str = "unknown") -> CandidateProgram:
    clean = validate_code(code)
    return CandidateProgram(candidate_id=stable_hash(clean), code=clean, family=family, source_stage=source_stage)
