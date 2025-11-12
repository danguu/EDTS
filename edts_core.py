from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Set
import re
import sys
from pathlib import Path

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
            for alt in alts:
                if alt == EPS:
                    G.setdefault(lhs, []).append([EPS])
                else:
                    G.setdefault(lhs, []).append(alt.split())
    return G

Token = Tuple[str, str]
TOKEN_SPEC = [
    ("NUM",    r'\d+(?:\.\d+)?'),
    ("ID",     r'[A-Za-z_]\w*'),
    ("PLUS",   r'\+'),
    ("MINUS",  r'-'),
    ("TIMES",  r'\*'),
    ("DIV",    r'/'),
    ("LPAREN", r'\('),
    ("RPAREN", r'\)'),
    ("WS",     r'[ \t\r\n]+'),
]
TOKEN_RE = re.compile("|".join(f"(?P<{n}>{p})" for n, p in TOKEN_SPEC))

def lex(s: str) -> List[Token]:
    out = []
    for m in TOKEN_RE.finditer(s):
        k = m.lastgroup
        if k != "WS":
            out.append((k, m.group()))
    out.append(("EOF", ""))
    return out

def first_of_string(alpha, FIRST, terminals):
    out, nullable = set(), True
    for X in alpha:
        if X in terminals:
            out.add(X)
            nullable = False
            break
        out |= (FIRST[X] - {EPS})
        if EPS not in FIRST[X]:
            nullable = False
            break
    if nullable:
        out.add(EPS)
    return out

def compute_first(G, terminals):
    F = {A: set() for A in G}
    changed = True
    while changed:
        changed = False
        for A, prods in G.items():
            for α in prods:
                acc = first_of_string(α, F, terminals)
                before = len(F[A])
                F[A] |= acc
                changed |= len(F[A]) != before
    return F

def compute_follow(G, FIRST, start, terminals):
    F = {A: set() for A in G}
    F[start].add("EOF")
    changed = True
    while changed:
        changed = False
        for A, prods in G.items():
            for α in prods:
                for i, B in enumerate(α):
                    if B in G:
                        β = α[i+1:]
                        fb = first_of_string(β, FIRST, terminals)
                        before = len(F[B])
                        F[B] |= (fb - {EPS})
                        if EPS in fb or not β:
                            F[B] |= F[A]
                        changed |= len(F[B]) != before
    return F

def compute_predict(G, FIRST, FOLLOW, terminals):
    P = {}
    for A, prods in G.items():
        for α in prods:
            key = (A, tuple(α))
            s = first_of_string(α, FIRST, terminals)
            if EPS in s:
                s.remove(EPS)
                s |= FOLLOW[A]
            P[key] = s
    return P

@dataclass
class AST:
    kind: str
    lexeme: Optional[str] = None
    children: List['AST'] = field(default_factory=list)
    val: Optional[float] = None
    def pretty(self, indent="", last=True):
        b = "└── " if last else "├── "
        s = f"{indent}{b}{self.kind}"
        if self.lexeme: s += f"({self.lexeme})"
        if self.val is not None: s += f"  ⟨val={self.val}⟩"
        s += "\n"
        for i,c in enumerate(self.children):
            s += c.pretty(indent + ("    " if last else "│   "), i == len(self.children)-1)
        return s

class SymbolTable:
    def __init__(self): self.store = {}
    def define(self, n, v): self.store[n] = float(v)
    def lookup(self, n): return self.store.get(n)
    def __str__(self):
        out = ["== Tabla de símbolos =="]
        for k,v in sorted(self.store.items()):
            out.append(f"{k:10s} tipo=num valor={v}")
        return "\n".join(out)

class Parser:
    def __init__(self, toks, σ): self.toks, self.i, self.symtab = toks,0,σ
    def look(self): return self.toks[self.i]
    def match(self,k):
        t=self.look()
        if t[0]!=k: raise SyntaxError(f"Se esperaba {k} y se encontró {t}")
        self.i+=1; return t
    def F(self):
        t=self.look()
        if t[0]=="LPAREN": self.match("LPAREN"); e=self.E(); self.match("RPAREN"); return e
        elif t[0]=="ID": id=self.match("ID")[1]; v=self.symtab.lookup(id)
        ...

