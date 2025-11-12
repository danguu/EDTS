import argparse
from pathlib import Path
from edts_core import (
    load_grammar, compute_first, compute_follow, compute_predict,
    sets_report, evaluate_expression, read_env_file
)

TERMINALS = {"PLUS","MINUS","TIMES","DIV","LPAREN","RPAREN","ID","NUM","EOF"}

ap = argparse.ArgumentParser(description="EDTS LL(1) modular (sin bloque main)")
ap.add_argument("--gram", required=True)
ap.add_argument("--expr-file", required=True)
ap.add_argument("--env-file", default="")
ap.add_argument("--out-dir", required=True)
args = ap.parse_args()

outdir = Path(args.out_dir)
outdir.mkdir(parents=True, exist_ok=True)

G = load_grammar(args.gram)
FIRST = compute_first(G, TERMINALS)
FOLLOW = compute_follow(G, FIRST, start="E", terminals=TERMINALS)
PREDICT = compute_predict(G, FIRST, FOLLOW, TERMINALS)
(outdir / "sets.txt").write_text(sets_report(G, FIRST, FOLLOW, PREDICT), encoding="utf-8")

expr = Path(args.expr_file).read_text(encoding="utf-8").strip()
env = read_env_file(args.env_file) if args.env_file else {}
try:
    ast, σ = evaluate_expression(expr, env)
    res = [f"== Expresión ==\n{expr}\n", "== AST decorado ==\n", ast.pretty(), str(σ), f"\n== Valor ==\n{ast.val}\n"]
    (outdir / "resultado.txt").write_text("".join(res), encoding="utf-8")
except Exception as e:
    (outdir / "resultado.txt").write_text(f"Error: {e}\n", encoding="utf-8")
