import argparse
from pathlib import Path
from edts_core import (
    load_grammar,
    compute_first,
    compute_follow,
    compute_predict,
    sets_report,
    evaluate_expression,
    read_env_file,
    EPS,  # noqa: F401
)

# Conjunto de terminales esperados por el lexer/parser fijo
TERMINALS = {"PLUS", "MINUS", "TIMES", "DIV", "LPAREN", "RPAREN", "ID", "NUM", "EOF"}


def main():
    ap = argparse.ArgumentParser(description="EDTS LL(1) modular")
    ap.add_argument("--gram", required=True, help="Ruta a gramatica.txt")
    ap.add_argument(
        "--expr-file", required=True, help="Ruta a archivo .txt con UNA expresión"
    )
    ap.add_argument(
        "--env-file", default="", help="Ruta a archivo con variables (k=v por línea)"
    )
    ap.add_argument("--out-dir", required=True, help="Directorio de salida")
    args = ap.parse_args()

    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Cargar gramática y conjuntos
    G = load_grammar(args.gram)
    FIRST = compute_first(G, TERMINALS)
    FOLLOW = compute_follow(G, FIRST, start="E", terminals=TERMINALS)
    PREDICT = compute_predict(G, FIRST, FOLLOW, TERMINALS)

    (outdir / "sets.txt").write_text(
        sets_report(G, FIRST, FOLLOW, PREDICT), encoding="utf-8"
    )

    # 2) Leer expresión y entorno
    expr = Path(args.expr_file).read_text(encoding="utf-8").strip()
    env = read_env_file(args.env_file) if args.env_file else {}

    # 3) Evaluar con ETDS
    try:
        ast, σ = evaluate_expression(expr, env)
    except Exception as e:
        (outdir / "resultado.txt").write_text(f"Error: {e}\n", encoding="utf-8")
        return 1

    # 4) Escritura de resultado
    res = []
    res.append("== Expresión ==")
    res.append(expr)
    res.append("\n== AST decorado ==")
    res.append(ast.pretty())
    res.append(str(σ))
    res.append(f"\n== Valor ==")  # noqa: F541
    res.append(str(ast.val))
    (outdir / "resultado.txt").write_text("\n".join(res), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
