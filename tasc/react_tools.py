"""
Safe arithmetic calculator: + - * / ** and parentheses only.
No names, no imports, no function calls — ``ast``-validated.
"""

from __future__ import annotations

import ast
import operator


_ALLOWED_BINOPS: dict[type[ast.operator], object] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}

_ALLOWED_UNARY: dict[type[ast.unaryop], object] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        v = node.value
        if isinstance(v, bool):
            raise ValueError("only numeric constants allowed")
        if isinstance(v, (int, float)):
            return float(v)
        raise ValueError("only numeric constants allowed")

    # Python <3.8 style numbers in some trees
    if isinstance(node, ast.Num):  # pragma: no cover
        return float(node.n)

    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
        fn = _ALLOWED_UNARY[type(node.op)]
        return float(fn(_eval_node(node.operand)))

    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        fn = _ALLOWED_BINOPS[type(node.op)]
        return float(fn(_eval_node(node.left), _eval_node(node.right)))

    raise ValueError(f"disallowed syntax: {type(node).__name__}")


def _sanitize_expression(expr: str) -> str:
    """
    Clean up common LLM formatting in calculator expressions so they parse
    as valid Python arithmetic: strip currency symbols, commas inside numbers,
    Unicode operators, whitespace-only junk, etc.
    """
    expr = expr.replace("$", "")
    expr = expr.replace("÷", "/")
    expr = expr.replace("×", "*")
    expr = expr.replace("−", "-")       # Unicode minus
    expr = expr.replace("\u2013", "-")   # en-dash
    expr = expr.replace("\u2014", "-")   # em-dash
    # Strip commas that are thousands-separators (digit,digit pattern)
    import re
    expr = re.sub(r"(\d),(\d)", r"\1\2", expr)
    # Remove stray text like trailing "= 123" that some models append
    expr = re.split(r"\s*=\s*(?=\s*-?\d|$)", expr)[0]
    return expr.strip()


def safe_calculate(expression: str) -> tuple[float | None, str | None]:
    """
    Evaluate a pure arithmetic expression.

    Automatically sanitizes common LLM formatting (``$``, ``÷``, commas, etc.)
    before evaluation.

    Returns ``(value, None)`` on success, or ``(None, error_message)``.
    """
    expr = _sanitize_expression((expression or "").strip())
    if not expr:
        return None, "empty expression"

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        return None, f"syntax error: {e}"

    if not isinstance(tree, ast.Expression):
        return None, "invalid parse tree"

    try:
        val = _eval_node(tree.body)
    except (ValueError, ZeroDivisionError, OverflowError) as e:
        return None, str(e)

    return val, None
