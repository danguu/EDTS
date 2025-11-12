# EDTS

Este proyecto implementa un **Esquema de Traducción Dirigido por la Sintaxis (ETDS)** para una **Gramática Independiente de Contexto (GIC)** que permite **sumar, restar, multiplicar y dividir**.

##  Estructura de archivos

```
gramatica.txt      # Define la GIC externa
expr.txt           # Contiene la expresión a evaluar
env.txt            # Variables y sus valores opcionales
edts_core.py       # Módulo con gramática, parser, ETDS, FIRST/FOLLOW/PREDICT
edts_run.py        # Script principal de ejecución
out/
 ├── sets.txt      # Reporte de FIRST, FOLLOW y PREDICT
 └── resultado.txt # AST decorado, tabla de símbolos y valor final
```

##  Gramática (`gramatica.txt`)

```
E  -> T E'
E' -> PLUS T E' | MINUS T E' | ε
T  -> F T'
T' -> TIMES F T' | DIV F T' | ε
F  -> LPAREN E RPAREN | ID | NUM
```

Esta gramática es **LL(1)**, adecuada para un parser predictivo recursivo.
El símbolo inicial es `E`. Los terminales incluyen `PLUS`, `MINUS`, `TIMES`, `DIV`, `LPAREN`, `RPAREN`, `ID`, `NUM`, y `EOF`.


##  Atributos y ETDS

Cada nodo del **AST** contiene un atributo sintetizado `val`.
La traducción dirigida por la sintaxis (ETDS) calcula los valores durante el análisis.

**Reglas principales:**

| Producción   | Acción semántica              |
| ------------ | ----------------------------- |
| F → NUM      | `F.val = float(NUM.lex)`      |
| F → ID       | `F.val = σ[ID.lex]`           |
| F → (E)      | `F.val = E.val`               |
| T' → * F T'₁ | `T'.val = T'.inh.val * F.val` |
| T' → / F T'₁ | `T'.val = T'.inh.val / F.val` |
| E' → + T E'₁ | `E'.val = E'.inh.val + T.val` |
| E' → - T E'₁ | `E'.val = E'.inh.val - T.val` |


## Funcionalidad

* **Cálculo automático de conjuntos:** `FIRST`, `FOLLOW`, `PREDICT`
* **Parser predictivo LL(1):** sin conflictos ni retrocesos
* **AST decorado:** imprime nodos y valores (`⟨val⟩`)
* **Tabla de símbolos:** muestra variables y valores
* **Reportes en .txt:** resultados exportables

## Ejecución

1. Crea `expr.txt`:

   ```
   y + 2 * 2
   ```

2. Crea `env.txt` (opcional):

   ```
   y=2
   ```

3. Ejecuta:

   ```bash
   python3 edts_run.py --gram gramatica.txt --expr-file expr.txt --env-file env.txt --out-dir out
   ```

4. Verifica los resultados en `out/`:

   * **`sets.txt`**: contiene los conjuntos FIRST, FOLLOW y PREDICT.
   * **`resultado.txt`**: muestra AST decorado, tabla de símbolos y valor final.


##  Ejemplo de salida

```
== Expresión ==
y + 2 * 2

== AST decorado ==
└── BinOp(+)
    ├── Id(y)  ⟨val=2.0⟩
    └── BinOp(*)
        ├── Num(2)  ⟨val=2.0⟩
        └── Num(2)  ⟨val=2.0⟩

== Tabla de símbolos ==
y          tipo=num valor=2.0

== Valor ==
6.0
```
