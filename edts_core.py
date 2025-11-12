from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Set
import re
import sys  # noqa: F401
from pathlib import Path

# Gramática desde archivo

EPS = "ε"


def load_grammar(path: str) -> Dict[str, List[List[str]]]:
    G: Dict[str, List[List[str]]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "->" not in line:
                continue
            lhs, rhs = [x.strip() for x in line.split("->", 1)]
            alts = [alt.strip() for alt in rhs.split("|")]
            prods: List[List[str]] = []
            for alt in alts:
                if alt == EPS or alt.upper() == EPS:
                    prods.append([EPS])
                else:
                    prods.append(alt.split())
            G.setdefault(lhs, []).extend(prods)
    if not G:
        raise ValueError("Gramática vacía o inválida.")
    return G

# Léxico

Token = Tuple[str, str]  # (type, lexeme)

TOKEN_SPEC = [
    ("NUM", r"\d+(?:\.\d+)?"),
    ("ID", r"[A-Za-z_]\w*"),
    ("PLUS", r"\+"),
    ("MINUS", r"-"),
    ("TIMES", r"\*"),
    ("DIV", r"/"),
    ("LPAREN", r"\("),
    ("RPAREN", r"\)"),
    ("WS", r"[ \t\r\n]+"),
]

TOKEN_RE = re.compile("|".join(f"(?P<{name}>{pat})" for name, pat in TOKEN_SPEC))


def lex(s: str) -> List[Token]:
    out: List[Token] = []
    for m in TOKEN_RE.finditer(s):
        kind = m.lastgroup
        lexeme = m.group()
        if kind == "WS":
            continue
        out.append((kind, lexeme))
    out.append(("EOF", ""))
    return out


# FIRST, FOLLOW, PREDICT


def first_of_string(
    alpha: List[str], FIRST: Dict[str, Set[str]], terminals: Set[str]
) -> Set[str]:
    out: Set[str] = set()
    nullable_prefix = True
    for X in alpha:
        if X in terminals:
            out.add(X)
            nullable_prefix = False
            break
        else:
            out |= FIRST[X] - {EPS}
            if EPS not in FIRST[X]:
                nullable_prefix = False
                break
    if nullable_prefix:
        out.add(EPS)
    return out


def compute_first(
    grammar: Dict[str, List[List[str]]], terminals: Set[str]
) -> Dict[str, Set[str]]:
    FIRST: Dict[str, Set[str]] = {A: set() for A in grammar}
    changed = True
    while changed:
        changed = False
        for A, prods in grammar.items():
            for alpha in prods:
                if alpha == [EPS]:
                    if EPS not in FIRST[A]:
                        FIRST[A].add(EPS)
                        changed = True
                    continue
                acc = first_of_string(alpha, FIRST, terminals)
                before = len(FIRST[A])
                FIRST[A] |= acc
                if len(FIRST[A]) != before:
                    changed = True
    return FIRST


def compute_follow(
    grammar: Dict[str, List[List[str]]],
    FIRST: Dict[str, Set[str]],
    start: str,
    terminals: Set[str],
) -> Dict[str, Set[str]]:
    FOLLOW: Dict[str, Set[str]] = {A: set() for A in grammar}
    FOLLOW[start].add("EOF")
    changed = True
    while changed:
        changed = False
        for A, prods in grammar.items():
            for alpha in prods:
                for i, B in enumerate(alpha):
                    if B in grammar:  # no terminal
                        beta = alpha[i + 1 :]
                        FIRST_beta = first_of_string(beta, FIRST, terminals)
                        before = len(FOLLOW[B])
                        FOLLOW[B] |= FIRST_beta - {EPS}
                        if EPS in FIRST_beta or not beta:
                            FOLLOW[B] |= FOLLOW[A]
                        if len(FOLLOW[B]) != before:
                            changed = True
    return FOLLOW


def compute_predict(
    grammar: Dict[str, List[List[str]]],
    FIRST: Dict[str, Set[str]],
    FOLLOW: Dict[str, Set[str]],
    terminals: Set[str],
) -> Dict[tuple, Set[str]]:
    P: Dict[tuple, Set[str]] = {}
    for A, prods in grammar.items():
        for alpha in prods:
            key = (A, tuple(alpha))
            if alpha == [EPS]:
                P[key] = set(FOLLOW[A])
            else:
                s = first_of_string(alpha, FIRST, terminals)
                if EPS in s:
                    s.remove(EPS)
                    s |= FOLLOW[A]
                P[key] = s
    return P


def sets_report(
    grammar: Dict[str, List[List[str]]],
    FIRST: Dict[str, Set[str]],
    FOLLOW: Dict[str, Set[str]],
    PREDICT: Dict[tuple, Set[str]],
) -> str:
    def fmt(s: Set[str]) -> str:
        return "{" + ", ".join(sorted(s)) + "}" if s else "∅"

    nts = sorted(grammar.keys())
    lines = []
    lines.append("== FIRST ==")
    for A in nts:
        lines.append(f"FIRST({A}) = {fmt(FIRST[A])}")
    lines.append("\n== FOLLOW ==")
    for A in nts:
        lines.append(f"FOLLOW({A}) = {fmt(FOLLOW[A])}")
    lines.append("\n== PREDICT ==")
    for (A, alpha), S in sorted(PREDICT.items()):
        rhs = " ".join(alpha)
        lines.append(f"PREDICT({A} -> {rhs}) = {fmt(S)}")
    return "\n".join(lines)


# AST y símbolos

from dataclasses import dataclass, field  # noqa: E402, F811
from typing import List, Optional  # noqa: E402, F811


@dataclass
class AST:
    kind: str
    lexeme: Optional[str] = None
    children: List["AST"] = field(default_factory=list)
    val: Optional[float] = None

    def pretty(self, indent: str = "", is_last: bool = True) -> str:
        branch = "└── " if is_last else "├── "
        s = f"{indent}{branch}{self.kind}"
        if self.lexeme:
            s += f"({self.lexeme})"
        if self.val is not None:
            s += f"  ⟨val={self.val}⟩"
        s += "\n"
        new_indent = indent + ("    " if is_last else "│   ")
        for i, c in enumerate(self.children):
            s += c.pretty(new_indent, i == len(self.children) - 1)
        return s


class SymbolTable:
    def __init__(self):
        self.store: Dict[str, float] = {}

    def define(self, name: str, value: float) -> None:
        self.store[name] = float(value)

    def lookup(self, name: str) -> Optional[float]:
        return self.store.get(name)

    def __str__(self) -> str:
        if not self.store:
            return "== Tabla de símbolos ==\n<vacía>"
        lines = ["== Tabla de símbolos =="]
        for k, v in sorted(self.store.items()):
            lines.append(f"{k:10s} tipo=num valor={v}")
        return "\n".join(lines)


# Parser predictivo (fijo para la gramática de expresiones)


class Parser:
    def __init__(self, tokens: List[Token], symtab: SymbolTable):
        self.toks = tokens
        self.i = 0
        self.symtab = symtab

    def look(self) -> Token:
        return self.toks[self.i]

    def match(self, kind: str) -> Token:
        t = self.look()
        if t[0] != kind:
            raise SyntaxError(f"Se esperaba {kind} y se encontró {t[0]} ('{t[1]}')")
        self.i += 1
        return t

    # F -> (E) | ID | NUM
    def F(self) -> AST:
        t = self.look()
        if t[0] == "LPAREN":
            self.match("LPAREN")
            e = self.E()
            self.match("RPAREN")
            return e
        elif t[0] == "ID":
            idt = self.match("ID")
            name = idt[1]
            v = self.symtab.lookup(name)
            if v is None:
                raise NameError(f"Variable sin valor: {name}")
            return AST("Id", name, [], v)
        elif t[0] == "NUM":
            numt = self.match("NUM")
            return AST("Num", numt[1], [], float(numt[1]))
        else:
            raise SyntaxError(f"F: token inesperado {t}")

    # T' con heredado
    def Tp(self, inh: AST) -> AST:
        t = self.look()
        if t[0] == "TIMES":
            self.match("TIMES")
            f = self.F()
            node = AST("BinOp", "*", [inh, f], inh.val * f.val)
            return self.Tp(node)
        elif t[0] == "DIV":
            self.match("DIV")
            f = self.F()
            node = AST("BinOp", "/", [inh, f], inh.val / f.val)
            return self.Tp(node)
        elif t[0] in {"PLUS", "MINUS", "RPAREN", "EOF"}:
            return inh
        else:
            raise SyntaxError(f"T': token inesperado {t}")

    # T -> F T'
    def T(self) -> AST:
        f = self.F()
        return self.Tp(f)

    # E' con heredado
    def Ep(self, inh: AST) -> AST:
        t = self.look()
        if t[0] == "PLUS":
            self.match("PLUS")
            tr = self.T()
            node = AST("BinOp", "+", [inh, tr], inh.val + tr.val)
            return self.Ep(node)
        elif t[0] == "MINUS":
            self.match("MINUS")
            tr = self.T()
            node = AST("BinOp", "-", [inh, tr], inh.val - tr.val)
            return self.Ep(node)
        elif t[0] in {"RPAREN", "EOF"}:
            return inh
        else:
            raise SyntaxError(f"E': token inesperado {t}")

    # E -> T E'
    def E(self) -> AST:
        t = self.T()
        return self.Ep(t)

    def parse(self) -> AST:
        ast = self.E()
        self.match("EOF")
        return ast


# Utilidades de I/O


def read_env_file(path: Optional[str]) -> Dict[str, float]:
    env: Dict[str, float] = {}
    if not path:
        return env
    txt = Path(path).read_text(encoding="utf-8")
    for line in txt.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = float(v.strip())
    return env


def symtab_from_env(env: Dict[str, float]) -> SymbolTable:
    σ = SymbolTable()
    for k, v in env.items():
        σ.define(k, v)
    return σ


def evaluate_expression(expr: str, env: Dict[str, float]) -> Tuple[AST, SymbolTable]:
    tokens = lex(expr)
    σ = symtab_from_env(env)
    parser = Parser(tokens, σ)
    ast = parser.parse()
    return ast, σ
