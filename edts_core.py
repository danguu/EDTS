from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Set
from pathlib import Path
import re

# Gramática (cargada desde archivo)

EPS = "ε"


def load_grammar(path: str) -> Dict[str, List[List[str]]]:
    """
    Lee una gramática externa en formato:
      A  -> X Y Z | ε
    Comentarios inician con '#'.
    Retorna: dict { NoTerminal: [ [símbolos...], [EPS] , ...] }
    """
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
            for alt in alts:
                if alt == EPS:
                    G.setdefault(lhs, []).append([EPS])
                else:
                    G.setdefault(lhs, []).append(alt.split())
    if not G:
        raise ValueError("Gramática vacía o inválida en el archivo.")
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
    """
    FIRST(alpha) para secuencia alpha de símbolos gramaticales.
    """
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
                    if B in grammar:  # B es no terminal
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


# AST y Tabla de símbolos


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


# Parser predictivo (gramática de expresiones)


class Parser:
    """
    Implementa la ETDS:
      F -> NUM           { val = float(NUM) }
      F -> ID            { val = σ[ID] (debe existir) }
      F -> ( E )         { val = E.val }
      T'-> * F T'1       { val = inh.val * F.val; T'1.inh = ese resultado }
      T'-> / F T'1       { val = inh.val / F.val; ... }
      T'-> ε             { val = inh.val }
      E'-> + T E'1       { val = inh.val + T.val; ... }
      E'-> - T E'1       { val = inh.val - T.val; ... }
      E -> T E'          { E'.inh = T.val; E.val = E'.val }
      T -> F T'          { T'.inh = F.val; T.val = T'.val }
    """

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
            return e  # F.val = E.val
        elif t[0] == "ID":
            tok = self.match("ID")
            name = tok[1]
            v = self.symtab.lookup(name)
            if v is None:
                raise NameError(f"Variable sin valor: {name}")
            return AST("Id", name, [], v)
        elif t[0] == "NUM":
            tok = self.match("NUM")
            return AST("Num", tok[1], [], float(tok[1]))
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
            return inh  # ε
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
            return inh  # ε
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
    """
    Lee variables desde archivo:
      x=2
      y=3.5
    Retorna dict {nombre: valor}
    """
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


def symtab_from_env(env: Dict[str, float]) -> "SymbolTable":
    σ = SymbolTable()
    for k, v in env.items():
        σ.define(k, v)
    return σ


def evaluate_expression(expr: str, env: Dict[str, float]) -> Tuple[AST, "SymbolTable"]:
    """
    Toma string con la expresión, y un dict de entorno {id: valor}.
    Retorna (AST decorado, tabla de símbolos).
    """
    tokens = lex(expr)
    σ = symtab_from_env(env)
    parser = Parser(tokens, σ)
    ast = parser.parse()
    return ast, σ

