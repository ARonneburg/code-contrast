# supports:
# diff-anywhere
# diff-atcursor
# diff-selection
# infill
# edit-chain

# doesn't support:
# highlight

# §MSG do this please
# §FILE a.h
# code
# code
# §45
# code
# code
# §34
# code
# code
# §FILE a.cpp
# code
# code
# §54
# code
# code
# §64
# code
# code<EOF>
# §CHUNK34
# -code
# -code
# §3
# +code
# +code
# §EOT

# --------------- no tpos -----------------

# §FILE a.h
# §LINE5030
# code
# code
# §SELECT       # or §CURSOR
# code
# code
# code
# §/SELECT
# code
# §LINE5050
# code
# code
# §FILE a.cpp
# §LINE3010
# code
# code
# code
# code
# code
# code
# §USER hello robot
# §ASSISTANT hello human
# §CHUNK       # two lines or less, writes "DEL#" if needs to delete more
# code
# code
# §LINE5032,DEL3
# code
# code
# code
# §/CHUNK
# §ASSISTANT hope this helps
# §TOOL console ls
# a.h
# a.cpp
# §ASSISTANT Here I used a tool for you.
# §USER Thank you robot
# §EOT

# Improvements over the previous diff:
# 1. Given code selection -> predict user instruction
#    (or cursor position)
# 2. Multi turn chat over the same changes
# 3. Tool support (run, goto definition, project search)
# 4. One special token (escape), no position tokens
# 5. Logit intrusion less needed
